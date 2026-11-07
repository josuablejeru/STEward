"""Local Hugging Face Transformers backend (base model plus optional LoRA adapter).

Needs the CUDA environment: ``pixi run -e cuda ...``. Torch, Transformers,
and PEFT are imported lazily, so the default environment can import this
module and get a clear error message instead of an ImportError.

Generation uses the model's own chat template with thinking disabled
(``enable_thinking=False`` for Qwen3). ``temperature == 0`` means greedy
decoding.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from steward.generation.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelInfo,
    ModelNotFoundError,
    ProviderError,
)
from steward.ste.preservation import strip_reasoning


class TransformersProvider:
    """Run a causal LM locally with Transformers (and PEFT for adapters)."""

    name = "transformers"

    def __init__(
        self,
        *,
        revision: str | None = None,
        adapter_path: str | Path | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
        timeout_s: float | None = None,
        think: bool | None = False,
        **_: Any,
    ) -> None:
        """Configure the backend. The model is loaded on the first request."""
        self.revision = revision
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.device = device
        self.dtype = dtype
        self.think = bool(think)
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded: str | None = None
        self._info: ModelInfo | None = None
        del timeout_s  # local generation has no network timeout

    # -- loading ------------------------------------------------------------

    def _load(self, model: str) -> None:
        if self._loaded == model:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ProviderError(
                f"The transformers backend needs PyTorch and Transformers: {exc}.",
                hint="Run the command in the CUDA environment: pixi run -e cuda ...",
            ) from exc
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, revision=self.revision)
            lm = AutoModelForCausalLM.from_pretrained(
                model, revision=self.revision, dtype=getattr(torch, self.dtype)
            )
        except OSError as exc:
            raise ModelNotFoundError(
                f"Cannot load model {model!r} (revision {self.revision}): {exc}",
                hint="Check the model ID and that it is downloaded or reachable.",
            ) from exc
        if self.adapter_path is not None:
            if not (self.adapter_path / "adapter_config.json").is_file():
                raise ModelNotFoundError(
                    f"No LoRA adapter at {self.adapter_path} (adapter_config.json missing).",
                    hint="Train one with `pixi run -e cuda train`.",
                )
            from peft import PeftModel

            lm = PeftModel.from_pretrained(lm, str(self.adapter_path))
            lm = lm.merge_and_unload()
        lm.to(device)
        lm.eval()
        self._model, self._tokenizer, self._loaded = lm, tokenizer, model
        self._device = device

    # -- protocol -------------------------------------------------------------

    def describe_model(self, model: str) -> ModelInfo:
        """Load the model and report its identity, size, and parameter count."""
        if self._info is not None and self._loaded == model:
            return self._info
        self._load(model)
        import torch
        import transformers

        params = sum(p.numel() for p in self._model.parameters())
        snapshot = _snapshot_dir(model, self.revision)
        files = list(snapshot.glob("*.safetensors")) if snapshot.is_dir() else []
        size = sum(f.stat().st_size for f in files)
        adapter_digest = None
        adapter_size = None
        if self.adapter_path is not None:
            from steward.evaluation.fingerprints import file_sha256

            weights = self.adapter_path / "adapter_model.safetensors"
            if weights.is_file():
                adapter_digest = file_sha256(weights)
                adapter_size = weights.stat().st_size
                size += adapter_size
        revision = self.revision or _resolved_revision(snapshot)
        digest = revision if adapter_digest is None else f"{revision}+{adapter_digest[:16]}"
        self._info = ModelInfo(
            provider=self.name,
            model=model,
            digest=digest,
            provider_version=f"transformers {transformers.__version__}, torch {torch.__version__}",
            size_bytes=size or None,
            details={
                "parameter_count": params,
                "dtype": self.dtype,
                "device": self._device,
                "adapter_path": str(self.adapter_path) if self.adapter_path else None,
                "adapter_sha256": adapter_digest,
                "adapter_size_bytes": adapter_size,
                "revision": revision,
            },
        )
        return self._info

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one chat completion (``response_schema`` is ignored: text output)."""
        self._load(request.model)
        import torch

        tok, lm = self._tokenizer, self._model
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.think
        )
        inputs = tok(prompt, return_tensors="pt").to(self._device)
        sampling = request.sampling
        kwargs: dict[str, Any] = {"max_new_tokens": sampling.max_tokens}
        if sampling.temperature > 0:
            kwargs |= {
                "do_sample": True,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
            }
            if sampling.top_k:
                kwargs["top_k"] = sampling.top_k
        else:
            kwargs |= {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        if sampling.seed is not None:
            torch.manual_seed(sampling.seed)
        started = time.perf_counter()
        with torch.inference_mode():
            out = lm.generate(**inputs, **kwargs, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        elapsed = time.perf_counter() - started
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        text = strip_reasoning(tok.decode(new_tokens, skip_special_tokens=True)).strip()
        n_new = int(new_tokens.shape[0])
        finished = n_new < sampling.max_tokens
        return GenerationResult(
            text=text,
            model=request.model,
            latency_s=round(elapsed, 4),
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=n_new,
            finish_reason="stop" if finished else "length",
            metadata={"eval_duration": int(elapsed * 1e9)},
        )

    def loaded_vram_mib(self, model: str) -> int | None:
        """Peak memory allocated by PyTorch in this process."""
        del model
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated() / 2**20)

    def close(self) -> None:
        """Release the model."""
        self._model = self._tokenizer = self._loaded = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _snapshot_dir(model: str, revision: str | None) -> Path:
    """Local directory of ``model``: the path itself, or its Hugging Face cache snapshot."""
    if Path(model).is_dir():
        return Path(model)
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model, revision=revision, local_files_only=True))
    except Exception:  # not cached locally: size and revision stay unknown
        return Path(model)


def _resolved_revision(snapshot: Path) -> str | None:
    """Hugging Face cache snapshots are named after the commit hash."""
    name = snapshot.name
    return name if len(name) == 40 and all(c in "0123456789abcdef" for c in name) else None
