"""LoRA supervised fine-tuning with TRL and PEFT (CUDA environment only).

Run with ``pixi run -e cuda train``. Training never starts automatically.

Output layout (``runs/<run_id>/``):

* ``adapter/``: final LoRA adapter weights and tokenizer files,
* ``checkpoints/``: intermediate Trainer checkpoints (resumable),
* ``training_config.json``: the resolved config and library versions,
* ``log_history.json``: every logged training and validation step,
* ``mask_check.json``: proof that only assistant tokens carry loss,
* ``training_summary.json``: the contract read by the benchmark system.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from steward import __version__
from steward.datasets.jsonl import atomic_write_text
from steward.datasets.sft import load_prepared
from steward.evaluation.records import (
    RecordStatus,
    ServingSpec,
    TrainingPoint,
    TrainingSummary,
)
from steward.generation.prompts import load_template
from steward.logs import log_event
from steward.provenance import git_state
from steward.training.config import TrainingConfig
from steward.training.data import check_completion_mask, to_prompt_completion

logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Training could not start or failed."""


def _write_json(path: Path, data: object) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")


def curve_from_history(history: list[dict[str, Any]]) -> list[TrainingPoint]:
    """Merge Trainer log entries into one point per step."""
    points: dict[int, dict[str, Any]] = {}
    for entry in history:
        step = int(entry.get("step", 0))
        point = points.setdefault(step, {"step": step})
        if "epoch" in entry:
            point["epoch"] = float(entry["epoch"])
        if "loss" in entry:
            point["train_loss"] = float(entry["loss"])
        if "eval_loss" in entry:
            point["val_loss"] = float(entry["eval_loss"])
        if "learning_rate" in entry:
            point["learning_rate"] = float(entry["learning_rate"])
        if "grad_norm" in entry:
            point["grad_norm"] = float(entry["grad_norm"])
        for key in ("gpu_allocated_mib", "gpu_peak_mib"):
            if key in entry:
                point[key] = int(entry[key])
    return [TrainingPoint.model_validate(p) for _, p in sorted(points.items())]


OOM_ADVICE = (
    "CUDA out of memory. Lower train.per_device_batch_size (and raise "
    "gradient_accumulation_steps to keep the effective batch), lower max_length, or set "
    "gradient_checkpointing = true. Saved checkpoints can be resumed with --resume."
)


def final_losses(history: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """The last logged training loss and validation loss."""
    train = [float(e["loss"]) for e in history if "loss" in e]
    val = [float(e["eval_loss"]) for e in history if "eval_loss" in e]
    return (train[-1] if train else None, val[-1] if val else None)


def best_eval(history: list[dict[str, Any]]) -> tuple[int | None, float | None]:
    """``(step, loss)`` of the lowest validation loss logged during training."""
    evals = [(int(e.get("step", 0)), float(e["eval_loss"])) for e in history if "eval_loss" in e]
    if not evals:
        return None, None
    step, loss = min(evals, key=lambda x: (x[1], x[0]))
    return step, loss


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else 0


def run_training(
    config: TrainingConfig,
    *,
    root: Path,
    prompt_dir: Path,
    resume: Path | None = None,
    config_path: Path | None = None,
) -> Path:
    """Train one LoRA adapter and return the run directory."""
    try:
        import peft
        import torch
        import transformers
        import trl
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise TrainingError(
            f"Training needs the CUDA environment ({exc}). Run: pixi run -e cuda train"
        ) from exc
    if not torch.cuda.is_available():
        raise TrainingError("No CUDA GPU is visible. Training needs a GPU.")
    if not torch.cuda.is_bf16_supported():
        raise TrainingError("This GPU does not support bfloat16.")

    dataset_dir = (
        config.dataset_dir if config.dataset_dir.is_absolute() else root / config.dataset_dir
    )
    prepared = load_prepared(dataset_dir)
    if not prepared.train or not prepared.validation:
        raise TrainingError(f"{dataset_dir}: the train or validation split is empty.")
    template = load_template(prompt_dir, prepared.manifest.prompt)

    if resume is not None:
        run_dir = resume
        run_id = run_dir.name
    else:
        run_id = f"{config.run_name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        run_dir = (root / config.output_root / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = run_dir / "checkpoints"
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.base_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = to_prompt_completion(prepared.train, template, tokenizer)
    val_rows = to_prompt_completion(prepared.validation, template, tokenizer)
    # Only prompt/completion go to TRL; row order is kept, so index i matches train_rows[i].
    train_ds = Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in train_rows]
    )
    val_ds = Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in val_rows]
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, revision=config.base_revision, dtype=torch.bfloat16
    )
    total_params = sum(p.numel() for p in model.parameters())
    lora = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    t = config.train
    args = SFTConfig(
        output_dir=str(checkpoints),
        num_train_epochs=t.epochs,
        max_steps=t.max_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler,
        warmup_steps=t.warmup_ratio,  # transformers 5: a float in [0, 1) is a ratio
        weight_decay=t.weight_decay,
        per_device_train_batch_size=t.per_device_batch_size,
        per_device_eval_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        gradient_checkpointing=t.gradient_checkpointing,
        max_length=t.max_length,
        logging_steps=t.logging_steps,
        eval_strategy="steps",
        eval_steps=t.eval_steps,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=t.select_best,
        metric_for_best_model=t.metric_for_best if t.select_best else None,
        greater_is_better=False if t.select_best and "loss" in t.metric_for_best else None,
        bf16=True,
        seed=config.seed,
        data_seed=config.seed,
        completion_only_loss=True,
        packing=False,
        report_to=[],
        dataset_num_proc=1,
        remove_unused_columns=True,
    )

    class GpuMemoryCallback(transformers.TrainerCallback):  # type: ignore[misc]
        """Add CUDA memory use to every log entry."""

        def on_log(
            self, _args: Any, _state: Any, _control: Any, logs: Any = None, **_: Any
        ) -> None:
            if logs is not None:
                logs["gpu_allocated_mib"] = int(torch.cuda.memory_allocated() / 2**20)
                logs["gpu_peak_mib"] = int(torch.cuda.max_memory_allocated() / 2**20)

    callbacks: list[Any] = [GpuMemoryCallback()]
    if t.early_stopping_patience is not None:
        callbacks.append(
            transformers.EarlyStoppingCallback(
                early_stopping_patience=t.early_stopping_patience,
                early_stopping_threshold=t.early_stopping_threshold,
            )
        )
    trainer = SFTTrainer(
        callbacks=callbacks,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )

    # Verify assistant-only loss on the processed data before spending GPU time.
    processed = trainer.train_dataset
    columns = processed.column_names
    if "completion_mask" in columns:
        mask_source = "completion_mask"
    elif "labels" in columns:
        mask_source = "labels"  # TRL >= 1.x: masked positions are -100
    else:
        raise TrainingError(
            f"TRL created neither completion_mask nor labels (columns: {columns}). "
            "Assistant-only loss cannot be guaranteed; training stopped."
        )
    mask_report = []
    for i in range(min(len(processed), 32)):
        row = processed[i]
        if mask_source == "labels":
            mask = [int(label != -100) for label in row["labels"]]
        else:
            mask = list(row["completion_mask"])
        ok, text = check_completion_mask(
            row["input_ids"], mask, train_rows[i]["completion"], tokenizer.decode
        )
        mask_report.append(
            {"index": i, "ok": ok, "unmasked_tokens": sum(mask), "unmasked_text": text[:300]}
        )
    truncated = sum(1 for row in processed if len(row["input_ids"]) >= t.max_length)
    mask_ok = all(r["ok"] for r in mask_report)
    _write_json(
        run_dir / "mask_check.json",
        {
            "mask_source": mask_source,
            "checked": len(mask_report),
            "all_ok": mask_ok,
            "truncated": truncated,
            "samples": mask_report[:5],
        },
    )
    if not mask_ok:
        raise TrainingError(
            f"Completion mask check failed; see {run_dir / 'mask_check.json'}. Training stopped."
        )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    commit, dirty = git_state(root)
    resolved = {
        "run_id": run_id,
        "config": config.model_dump(mode="json"),
        "config_sha256": config.sha256(),
        "config_path": str(config_path) if config_path else None,
        "dataset_dir": str(dataset_dir),
        "dataset_fingerprint": prepared.manifest.dataset_fingerprint,
        "dataset_label": prepared.manifest.label,
        "train_examples": len(prepared.train),
        "validation_examples": len(prepared.validation),
        "train_counts": prepared.manifest.counts,
        "prompt": template.ref.model_dump(mode="json"),
        "chat_template": "base tokenizer chat template, enable_thinking=False",
        "loss": "completion-only (assistant STE target + end-of-turn token)",
        "versions": {
            "steward": __version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "trl": trl.__version__,
            "python": platform.python_version(),
        },
        "gpu": torch.cuda.get_device_name(0),
        "git_commit": commit,
        "git_dirty": dirty,
        "total_parameters": total_params,
        "trainable_parameters": trainable,
        "started_at": datetime.now(UTC).isoformat(),
    }
    _write_json(run_dir / "training_config.json", resolved)
    log_event(
        logger,
        logging.INFO,
        "training_started",
        run_id=run_id,
        train=len(prepared.train),
        validation=len(prepared.validation),
        trainable=trainable,
        dataset=prepared.manifest.dataset_fingerprint[:12],
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    status, error = RecordStatus.COMPLETED, None
    selected_eval: float | None = None
    try:
        has_checkpoint = checkpoints.is_dir() and any(checkpoints.glob("checkpoint-*"))
        trainer.train(resume_from_checkpoint=True if resume and has_checkpoint else None)
        if t.select_best:
            # The best checkpoint is loaded; evaluate it again for the record (not logged as
            # a new curve point, so the curve keeps the values seen during training).
            selected_eval = float(trainer.evaluate()["eval_loss"])
        else:
            final_eval = trainer.evaluate()
            trainer.log({"eval_loss": final_eval["eval_loss"]})
            selected_eval = float(final_eval["eval_loss"])
    except torch.cuda.OutOfMemoryError as exc:
        status, error = RecordStatus.FAILED, f"{OOM_ADVICE} ({str(exc).splitlines()[0]})"
        log_event(logger, logging.ERROR, "training_oom", run_id=run_id, error=error)
        torch.cuda.empty_cache()
    except Exception as exc:
        status, error = RecordStatus.FAILED, f"{type(exc).__name__}: {exc}"
        log_event(logger, logging.ERROR, "training_failed", run_id=run_id, error=error)
    duration = time.perf_counter() - started
    history = trainer.state.log_history
    final_train, final_val = final_losses(history)
    best_step, best_val = best_eval(history)
    best_ckpt = trainer.state.best_model_checkpoint
    planned_steps = trainer.state.max_steps
    early_stopped = (
        t.early_stopping_patience is not None
        and status is RecordStatus.COMPLETED
        and trainer.state.global_step < planned_steps
    )
    _write_json(run_dir / "log_history.json", history)

    adapter_dir = run_dir / "adapter"
    adapter_sha = None
    if status is RecordStatus.COMPLETED:
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        from steward.evaluation.fingerprints import file_sha256

        adapter_sha = file_sha256(adapter_dir / "adapter_model.safetensors")
    adapter_bytes = (
        (adapter_dir / "adapter_model.safetensors").stat().st_size
        if adapter_sha is not None
        else None
    )
    rel_adapter = (
        adapter_dir.relative_to(root.resolve()).as_posix()
        if adapter_dir.is_relative_to(root.resolve())
        else str(adapter_dir)
    )
    summary = TrainingSummary(
        run_id=run_id,
        status=status,
        error=error,
        base_model=config.base_model,
        base_revision=config.base_revision,
        checkpoint_id=(
            f"{run_id}:{'best-step-' + str(best_step) if t.select_best else 'final'}"
            if status is RecordStatus.COMPLETED
            else f"{run_id}:none"
        ),
        adapter_id=f"lora-{adapter_sha[:16]}" if adapter_sha else None,
        checkpoint_path=rel_adapter,
        dataset_version=prepared.manifest.dataset_fingerprint,
        dataset_fingerprint=prepared.manifest.dataset_fingerprint,
        config_sha256=config.sha256(),
        prompt_versions=[template.ref.ref],
        hyperparameters={
            "method": "LoRA SFT (TRL SFTTrainer, PEFT)",
            "label": config.label,
            "lora_r": config.lora.r,
            "lora_alpha": config.lora.alpha,
            "lora_dropout": config.lora.dropout,
            "target_modules": list(config.lora.target_modules),
            "epochs": t.epochs,
            "max_steps": t.max_steps,
            "learning_rate": t.learning_rate,
            "lr_scheduler": t.lr_scheduler,
            "warmup_ratio": t.warmup_ratio,
            "effective_batch_size": t.per_device_batch_size * t.gradient_accumulation_steps,
            "max_length": t.max_length,
            "precision": "bf16",
            "seed": config.seed,
            "train_examples": len(prepared.train),
            "validation_examples": len(prepared.validation),
            "global_steps": trainer.state.global_step,
            "epochs_completed": round(float(trainer.state.epoch or 0.0), 4),
            "duration_s": round(duration, 1),
            "final_train_loss": final_train,
            "final_val_loss": final_val,
            "best_val_loss": best_val,
            "best_step": best_step,
            "selected_checkpoint": ("best-validation" if t.select_best else "final"),
            "selected_checkpoint_path": best_ckpt if t.select_best else None,
            "selected_val_loss": selected_eval,
            "planned_steps": planned_steps,
            "early_stopping_patience": t.early_stopping_patience,
            "early_stopped": early_stopped,
            "version_label": config.version_label,
            "weight_decay": t.weight_decay,
            "peak_gpu_memory_mib": int(torch.cuda.max_memory_allocated() / 2**20),
            "peak_gpu_reserved_mib": int(torch.cuda.max_memory_reserved() / 2**20),
            "adapter_size_bytes": adapter_bytes,
            "checkpoints_dir": str(checkpoints),
            "checkpoints_size_bytes": _dir_size(checkpoints),
            "gradient_checkpointing": t.gradient_checkpointing,
            "per_device_batch_size": t.per_device_batch_size,
            "gradient_accumulation_steps": t.gradient_accumulation_steps,
            "gpu": torch.cuda.get_device_name(0),
            "loss_masking": "assistant completion only (verified)",
            "truncated_train_examples": truncated,
        },
        parameter_count=total_params,
        trainable_parameters=trainable,
        curve=curve_from_history(history),
        serving=ServingSpec(
            provider="transformers",
            model=config.base_model,
            revision=config.base_revision,
            adapter_path=rel_adapter,
            prompt=template.ref.ref,
            suite=config.eval_suite,
        )
        if status is RecordStatus.COMPLETED
        else None,
    )
    _write_json(run_dir / "training_summary.json", summary.model_dump(mode="json"))
    log_event(
        logger,
        logging.INFO if status is RecordStatus.COMPLETED else logging.ERROR,
        "training_finished",
        run_id=run_id,
        status=status.value,
        steps=trainer.state.global_step,
        duration_s=round(duration, 1),
    )
    if status is RecordStatus.FAILED:
        raise TrainingError(f"Training failed: {error}. Details in {run_dir}.")
    return run_dir
