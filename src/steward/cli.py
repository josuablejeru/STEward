"""Command-line entry point for STEward."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from steward import __version__
from steward.environment import collect_environment_info
from steward.logs import configure_logging, log_event

DISCLAIMER = (
    "STEward output is STE-oriented. It is not certified or approved ASD-STE100 compliance."
)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

logger = logging.getLogger("steward.cli")


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {number}")
    return number


def default_ollama_url() -> str:
    """Return the Ollama URL from ``OLLAMA_HOST``, or the local default."""
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if not host:
        return "http://localhost:11434"
    return host if "://" in host else f"http://{host}"


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="steward",
        description="Tools for STE-oriented technical documentation.",
        epilog=DISCLAIMER,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    info = subparsers.add_parser("info", help="Show version and runtime information.")
    info.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    info.set_defaults(handler=_cmd_info)

    gen = subparsers.add_parser(
        "generate",
        help="Generate synthetic STE-oriented rewrite candidates with a teacher model.",
        description=(
            "Generate fictional plain-English passages and STE-oriented rewrite candidates. "
            "Accepted examples, rejected examples, and a run manifest are written next to "
            "--output. " + DISCLAIMER
        ),
    )
    gen.add_argument(
        "--model",
        required=True,
        help="Teacher model name, e.g. 'qwen3:8b'. With --input: a run directory, adapter "
        "directory, or Hugging Face model ID.",
    )
    gen.add_argument(
        "--input",
        default=None,
        help="STE-only inference mode: convert this text with the model and the output guard "
        "(CUDA environment) instead of generating a dataset.",
    )
    gen.add_argument(
        "--language", choices=["en", "de", "fr", "ro"], default=None, help="With --input."
    )
    gen.add_argument("--strict", action="store_true", help="With --input: reject rule errors.")
    gen.add_argument("--json", action="store_true", help="With --input: print the full result.")
    gen.add_argument(
        "--count", type=_positive_int, default=100, help="Target number of accepted examples."
    )
    gen.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/examples.jsonl"),
        help="Accepted-examples JSONL file (default: data/raw/examples.jsonl).",
    )
    gen.add_argument(
        "--config",
        type=Path,
        default=Path("configs/generation/default.toml"),
        help="Generation config TOML.",
    )
    gen.add_argument("--prompt-dir", type=Path, default=Path("prompts"), help="Prompt directory.")
    gen.add_argument("--provider", default="ollama", help="Provider name (default: ollama).")
    gen.add_argument(
        "--base-url",
        default=None,
        help="Provider URL. Default: $OLLAMA_HOST or http://localhost:11434.",
    )
    gen.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    gen.add_argument(
        "--timeout", type=_positive_float, default=None, help="Per-call timeout in seconds."
    )
    gen.add_argument(
        "--max-attempts", type=_positive_int, default=None, help="Attempts per provider call."
    )
    gen.add_argument(
        "--think",
        choices=["on", "off"],
        default=None,
        help="Turn model 'thinking' on or off (reasoning models only).",
    )
    gen.add_argument(
        "--checkpoint-every",
        type=_positive_int,
        default=5,
        help="Write files after this many scenarios (default: 5).",
    )
    gen.add_argument(
        "--max-scenarios",
        type=_positive_int,
        default=None,
        help="Stop after this many scenarios in this run (default: 3 x remaining + 5).",
    )
    mode = gen.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Continue an existing output.")
    mode.add_argument("--overwrite", action="store_true", help="Replace an existing output.")
    gen.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    gen.add_argument("--log-format", default="text", choices=["text", "json"])
    gen.set_defaults(handler=_cmd_generate)

    from steward.evaluation.cli import (
        add_benchmark_parser,
        add_evaluate_parser,
        add_sweep_select_parser,
        run_benchmark_command,
    )

    add_benchmark_parser(subparsers)
    add_evaluate_parser(subparsers)
    add_sweep_select_parser(subparsers)
    _add_ml_parsers(subparsers)
    subparsers.choices["benchmark"].set_defaults(dispatch=run_benchmark_command)
    return parser


def _cmd_info(args: argparse.Namespace) -> int:
    info = collect_environment_info()
    if args.json:
        print(json.dumps(info.to_dict(), indent=2, sort_keys=True))
    else:
        for key, value in info.to_dict().items():
            print(f"{key}: {value}")
        print()
        print(DISCLAIMER)
    return EXIT_OK


def _cmd_generate(args: argparse.Namespace) -> int:
    # Imported here so `steward info` and `--version` stay fast.
    from steward.datasets.jsonl import DatasetFileError
    from steward.generation.config import ConfigError, load_config
    from steward.generation.pipeline import (
        DatasetGenerator,
        GenerationAbortedError,
        OutputExistsError,
        RunStatus,
    )
    from steward.generation.prompts import PromptTemplateError
    from steward.generation.providers import ProviderError, create_provider
    from steward.provenance import is_git_tracked_location

    configure_logging(args.log_level, args.log_format)
    if args.input is not None:
        return _infer(args.model, args.input, args.language, None, args.strict, args.json)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        return _fail(str(exc))
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})
    request_updates: dict[str, object] = {}
    if args.timeout is not None:
        request_updates["timeout_s"] = args.timeout
    if args.max_attempts is not None:
        request_updates["max_attempts"] = args.max_attempts
    if request_updates:
        config = config.model_copy(
            update={"request": config.request.model_copy(update=request_updates)}
        )

    if is_git_tracked_location(args.output):
        log_event(
            logger,
            logging.WARNING,
            "output_not_git_ignored",
            output=str(args.output),
            hint="Generated data must stay out of Git. Write under data/ instead.",
        )

    provider_kwargs: dict[str, object] = {"timeout_s": config.request.timeout_s}
    if args.provider == "ollama":
        provider_kwargs["base_url"] = args.base_url or default_ollama_url()
        if args.think is not None:
            provider_kwargs["think"] = args.think == "on"
    elif args.base_url:
        provider_kwargs["base_url"] = args.base_url
    try:
        provider = create_provider(args.provider, **provider_kwargs)
    except ValueError as exc:
        return _fail(str(exc))

    try:
        generator = DatasetGenerator(
            provider=provider,
            model=args.model,
            config=config,
            prompt_dir=args.prompt_dir,
            output=args.output,
            resume=args.resume,
            overwrite=args.overwrite,
            checkpoint_every=args.checkpoint_every,
        )
        record = generator.run(args.count, max_scenarios=args.max_scenarios)
    except (OutputExistsError, PromptTemplateError, DatasetFileError) as exc:
        return _fail(str(exc))
    except GenerationAbortedError as exc:
        return _fail(f"Generation stopped: {exc}\nProgress was saved. Use --resume to continue.")
    except ProviderError as exc:
        return _fail(str(exc))
    except KeyboardInterrupt:
        print("\nInterrupted. Progress was saved. Use --resume to continue.", file=sys.stderr)
        return EXIT_INTERRUPTED
    finally:
        provider.close()

    paths = generator.paths
    print(
        f"\nStatus:      {record.status.value}\n"
        f"Accepted:    {record.accepted} this run, {len(generator.examples)} total "
        f"(target {args.count})\n"
        f"Rejected:    {record.rejected} this run {record.rejection_reasons or ''}\n"
        f"Time:        {record.elapsed_s:.1f} s, "
        f"{record.accepted_per_minute or 0:.2f} accepted/min, "
        f"{record.completion_tokens_per_s or 0:.1f} output tokens/s\n"
        f"Warnings:    {record.warning_checks_failed or 'none'}\n"
        f"Examples:    {paths.examples}\n"
        f"Rejected:    {paths.rejected}\n"
        f"Manifest:    {paths.manifest}\n\n"
        f"All examples have review_state 'candidate'. {DISCLAIMER}"
    )
    if record.status is not RunStatus.COMPLETED:
        print(f"Run did not complete: {record.error}", file=sys.stderr)
        return EXIT_FAILURE
    return EXIT_OK


def _add_ml_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    dataset = subparsers.add_parser("dataset", help="Prepare training data.")
    dsub = dataset.add_subparsers(dest="dataset_command", metavar="<action>")
    prep = dsub.add_parser(
        "prepare",
        help="Build a deterministic SFT train/validation split from accepted examples.",
    )
    prep.add_argument(
        "--input", action="append", required=True, help="Accepted-examples JSONL (glob allowed)."
    )
    prep.add_argument("--config", type=Path, default=Path("configs/sft/default.toml"))
    prep.add_argument("--output", type=Path, default=Path("data/sft/v1"))
    prep.add_argument(
        "--label",
        choices=["experimental", "pipeline-validation"],
        default=None,
        help="Default: pipeline-validation below 300 training examples, else experimental.",
    )
    prep.set_defaults(handler=_cmd_dataset_prepare)

    rep = dsub.add_parser(
        "repair",
        help="Linter-guided repair of candidates with STE rule errors (needs Ollama).",
    )
    rep.add_argument("--input", action="append", required=True, help="Accepted JSONL (glob).")
    rep.add_argument("--output", type=Path, required=True, help="Output JSONL (all examples).")
    rep.add_argument("--model", required=True, help="Teacher model, e.g. qwen3:8b.")
    rep.add_argument("--url", default=None, help="Ollama URL (default: OLLAMA_HOST or local).")
    rep.add_argument("--workers", type=_positive_int, default=4)
    rep.add_argument("--rounds", type=_positive_int, default=2)
    rep.add_argument("--seed", type=int, default=20261109)
    rep.add_argument("--config", type=Path, default=Path("configs/generation/multilingual.toml"))
    rep.set_defaults(handler=_cmd_dataset_repair)

    crit = dsub.add_parser(
        "critique",
        help="Independent LLM fidelity critique and critique-guided revision (needs Ollama).",
        description=(
            "A critic model from another family than the teacher compares each source text "
            "with its English target. Critical findings are sent to the reviser (the teacher) "
            "for a revision, which is kept only if every check passes and a new critique "
            "has no critical finding. Results are LLM judgements, not proof of correctness."
        ),
    )
    crit.add_argument("--input", action="append", required=True, help="Examples JSONL (glob).")
    crit.add_argument("--output", type=Path, required=True, help="Output JSONL (all examples).")
    crit.add_argument("--critic", required=True, help="Critic model, e.g. gemma3:12b.")
    crit.add_argument("--reviser", default=None, help="Reviser model (default: no revision).")
    crit.add_argument("--rounds", type=int, default=1, help="Revision rounds (default 1).")
    crit.add_argument("--url", default=None, help="Ollama URL (default: OLLAMA_HOST or local).")
    crit.add_argument("--workers", type=_positive_int, default=4)
    crit.add_argument("--seed", type=int, default=20261121)
    crit.add_argument(
        "--only-eligible",
        action="store_true",
        help="Critique only examples that pass prelint-3 and the STE rule check (the others "
        "are written unchanged and are excluded at preparation anyway).",
    )
    crit.add_argument("--config", type=Path, default=Path("configs/generation/fidelity.toml"))
    crit.set_defaults(handler=_cmd_dataset_critique)

    qr = dsub.add_parser(
        "report",
        help="Write the data-quality report (counts only, no example text).",
    )
    qr.add_argument("--generated", action="append", required=True, help="Accepted JSONL (glob).")
    qr.add_argument("--reviewed", action="append", default=[], help="Critiqued JSONL (glob).")
    qr.add_argument("--prepared", type=Path, required=True, help="Prepared SFT directory.")
    qr.add_argument("--output-md", type=Path, default=Path("docs/dataset-quality.md"))
    qr.set_defaults(handler=_cmd_dataset_report)

    train = subparsers.add_parser(
        "train",
        help="Train a LoRA adapter (CUDA environment; never runs automatically).",
        description="LoRA SFT with TRL and PEFT. Run: pixi run -e cuda train",
    )
    train.add_argument("--config", type=Path, default=Path("configs/training/qwen3-0.6b-lora.toml"))
    train.add_argument("--resume", type=Path, default=None, help="Resume a run directory.")
    train.add_argument("--max-steps", type=int, default=None, help="Override max_steps (smoke).")
    train.add_argument("--log-format", choices=["text", "json"], default="text")
    train.set_defaults(handler=_cmd_train)

    sweep = subparsers.add_parser(
        "train-sweep",
        help="Run the trials of a hyperparameter sweep (CUDA environment; never automatic).",
        description="Train each trial of a sweep file one after another. Failed trials are "
        "recorded and the next trial starts. State: runs/sweeps/<sweep_id>/sweep.json.",
    )
    sweep.add_argument("--sweep", type=Path, required=True, help="Sweep TOML file.")
    sweep.add_argument("--trial", action="append", default=[], help="Only this trial (repeat).")
    sweep.add_argument("--log-format", choices=["text", "json"], default="text")
    sweep.set_defaults(handler=_cmd_train_sweep)

    ste = subparsers.add_parser(
        "ste",
        help="STE-only inference with the output guard (CUDA environment).",
        description=(
            "Convert technical input (English, German, French, Romanian) to STE-oriented "
            "English. Prints only the text on success; prints the failure status otherwise. "
            + DISCLAIMER
        ),
    )
    ste.add_argument("text", nargs="?", help="Input text (default: read standard input).")
    ste.add_argument(
        "--model",
        default="latest",
        help=(
            "'latest' (newest completed run in runs/, the default), a run directory, "
            "an adapter directory, or a Hugging Face model ID such as Qwen/Qwen3-0.6B."
        ),
    )
    ste.add_argument("--revision", default=None)
    ste.add_argument("--adapter", type=Path, default=None, help="LoRA adapter directory.")
    ste.add_argument("--language", choices=["en", "de", "fr", "ro"], default=None)
    ste.add_argument("--strict", action="store_true", help="Reject on heuristic rule errors.")
    ste.add_argument("--json", action="store_true", help="Print the full response as JSON.")
    ste.set_defaults(handler=_cmd_ste)

    exp = subparsers.add_parser(
        "export-ollama",
        help="Merge a LoRA run and write an Ollama Modelfile (CUDA environment).",
        description=(
            "Merge the adapter into the base model, convert it to GGUF, and write "
            "<output>/model.gguf, <output>/Modelfile, and <output>/export.json. Import it with "
            "`ollama create <name> -f <output>/Modelfile` (or `pixi run ollama-create`). "
            "Ollama output is not checked by the STEward output guard. " + DISCLAIMER
        ),
    )
    exp.add_argument(
        "model",
        nargs="?",
        default="latest",
        help="'latest' (default), a run directory, or a LoRA adapter directory.",
    )
    exp.add_argument(
        "--output",
        type=Path,
        default=Path("runs/ollama-export"),
        help="Output directory (default: runs/ollama-export; replaced on each export).",
    )
    exp.add_argument("--revision", default=None)
    exp.add_argument(
        "--gguf-type",
        choices=["bf16", "f16", "q8_0"],
        default="bf16",
        help="GGUF weight type (default bf16, lossless for this model; q8_0 is half the size).",
    )
    exp.add_argument(
        "--keep-merged", action="store_true", help="Keep the merged safetensors directory."
    )
    exp.set_defaults(handler=_cmd_export_ollama)


def _cmd_dataset_prepare(args: argparse.Namespace) -> int:
    import glob

    from steward.datasets.sft import SftConfigError, load_sft_config, prepare, write_prepared
    from steward.evaluation.project import BenchmarkProject
    from steward.evaluation.suite import BenchmarkConfigError

    paths = sorted({Path(p) for pattern in args.input for p in glob.glob(pattern)})
    paths = [p for p in paths if not p.name.endswith(".rejected.jsonl")]
    if not paths:
        return _fail(f"No input files match {args.input}.")
    try:
        config = load_sft_config(args.config)
        project = BenchmarkProject.at(Path.cwd())
        golden: dict[str, list[str]] = {}
        for gdir in sorted(project.paths.suite_dir.parent.iterdir()):
            if (gdir / "suite.toml").is_file():
                suite = project.suite_named(gdir.name)
                golden[gdir.name] = [i.source_text for i in suite.items] + [
                    i.reference for i in suite.items if i.reference
                ]
        prepared = prepare(paths, config, project.rules, golden_texts=golden, label="experimental")
        label = args.label or (
            "pipeline-validation" if len(prepared.train) < 300 else "experimental"
        )
        if label != prepared.manifest.label:
            prepared = prepare(paths, config, project.rules, golden_texts=golden, label=label)
        written = write_prepared(prepared, args.output)
    except (SftConfigError, BenchmarkConfigError) as exc:
        return _fail(str(exc))
    m = prepared.manifest
    print(
        f"Train:       {len(prepared.train)} {m.counts['train']}\n"
        f"Validation:  {len(prepared.validation)} {m.counts['validation']}\n"
        f"Languages:   train {m.languages['train']}, validation {m.languages['validation']}\n"
        f"Mix target:  {m.mix_target}\nMix actual:  {m.mix_actual}\n"
        f"Excluded:    {m.exclusions}\nLabel:       {m.label}\n"
        f"Fingerprint: {m.dataset_fingerprint}\n"
        f"Written:     {', '.join(map(str, written.values()))}"
    )
    return EXIT_OK


def _cmd_dataset_repair(args: argparse.Namespace) -> int:
    import glob

    from steward.datasets.jsonl import read_jsonl, write_jsonl_atomic
    from steward.datasets.schema import DatasetExample, SamplingParameters
    from steward.generation.config import ConfigError, load_config
    from steward.generation.prompts import load_template
    from steward.generation.providers.ollama import OllamaProvider
    from steward.generation.repair import repair_all
    from steward.ste.rules import load_rule_config

    configure_logging(level="INFO")
    paths = sorted({Path(p) for pattern in args.input for p in glob.glob(pattern)})
    paths = [p for p in paths if not p.name.endswith(".rejected.jsonl") and p != args.output]
    if not paths:
        return _fail(f"No input files match {args.input}.")
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        return _fail(str(exc))
    examples = [e for p in paths for e in read_jsonl(p, DatasetExample)]
    provider = OllamaProvider(
        base_url=args.url or default_ollama_url(), timeout_s=config.request.timeout_s, think=False
    )
    done = 0

    def progress(_: object) -> None:
        nonlocal done
        done += 1
        if done % 50 == 0:
            log_event(logger, logging.INFO, "repair_progress", done=done, total=len(examples))

    try:
        repaired, counts = repair_all(
            examples,
            workers=args.workers,
            on_done=progress,
            provider=provider,
            model=args.model,
            template=load_template(Path("prompts"), "ste_repair@1"),
            rules=load_rule_config(Path("configs/ste/terms.toml")),
            settings=config.validation,
            sampling=SamplingParameters(temperature=0.2, top_p=0.9, max_tokens=1000),
            rounds=args.rounds,
            seed=args.seed,
        )
    finally:
        provider.close()
    write_jsonl_atomic(args.output, repaired)
    print(f"Examples: {len(repaired)} {dict(counts)}\nWritten:  {args.output}")
    return EXIT_OK


def _cmd_dataset_critique(args: argparse.Namespace) -> int:
    import glob
    from collections import Counter

    from steward.datasets.jsonl import read_jsonl, write_jsonl_atomic
    from steward.datasets.schema import DatasetExample, SamplingParameters
    from steward.generation.config import ConfigError, load_config
    from steward.generation.critique import review_all
    from steward.generation.prompts import load_template
    from steward.generation.providers.ollama import OllamaProvider
    from steward.ste.rules import load_rule_config

    configure_logging(level="INFO")
    paths = sorted({Path(p) for pattern in args.input for p in glob.glob(pattern)})
    paths = [p for p in paths if not p.name.endswith(".rejected.jsonl") and p != args.output]
    if not paths:
        return _fail(f"No input files match {args.input}.")
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        return _fail(str(exc))
    examples = [e for p in paths for e in read_jsonl(p, DatasetExample)]
    url = args.url or default_ollama_url()
    critic = OllamaProvider(base_url=url, timeout_s=config.request.timeout_s)
    reviser = (
        OllamaProvider(base_url=url, timeout_s=config.request.timeout_s, think=False)
        if args.reviser
        else None
    )
    done = 0

    def progress(_: object) -> None:
        nonlocal done
        done += 1
        if done % 50 == 0:
            log_event(logger, logging.INFO, "critique_progress", done=done, total=len(examples))

    revise_template = load_template(Path("prompts"), "ste_revise@1") if args.reviser else None
    rules = load_rule_config(Path("configs/ste/terms.toml"))
    todo = list(range(len(examples)))
    if args.only_eligible:
        from steward.generation.repair import rule_errors
        from steward.generation.validation import validate_multitask

        todo = [
            i
            for i, e in enumerate(examples)
            if validate_multitask(
                e.task_type,
                e.source_language,
                e.task_category,
                e.source_text,
                e.ste_candidate,
                e.technical_entities,
                config.validation,
                required_terms=e.provenance.scenario.required_terms,
            ).passed
            and not rule_errors(e.ste_candidate, e.task_category, rules)
        ]
        print(f"Eligible for critique: {len(todo)} of {len(examples)}")
    try:
        reviewed_subset, counts = review_all(
            [examples[i] for i in todo],
            workers=args.workers,
            on_done=progress,
            critic=critic,
            critic_model=args.critic,
            critique_template=load_template(Path("prompts"), "fidelity_critique@1"),
            reviser=reviser,
            reviser_model=args.reviser,
            revise_template=revise_template,
            rules=rules,
            settings=config.validation,
            critic_sampling=SamplingParameters(
                temperature=0.0, top_p=1.0, seed=args.seed, max_tokens=900
            ),
            revise_sampling=SamplingParameters(
                temperature=0.2, top_p=0.9, seed=args.seed, max_tokens=1000
            ),
            rounds=args.rounds,
        )
    finally:
        critic.close()
        if reviser is not None:
            reviser.close()
    reviewed = list(examples)
    for i, example in zip(todo, reviewed_subset, strict=True):
        reviewed[i] = example
    write_jsonl_atomic(args.output, reviewed)
    categories = Counter(
        p.category
        for e in reviewed
        if e.llm_review
        for p in e.llm_review.initial_problems
        if p.severity == "critical"
    )
    print(f"Examples: {len(reviewed)} {dict(counts)}")
    print(f"Critical findings of the first critique by category: {dict(categories)}")
    print(f"Written:  {args.output}")
    return EXIT_OK


def _cmd_dataset_report(args: argparse.Namespace) -> int:
    import glob

    from steward.datasets.jsonl import atomic_write_text
    from steward.datasets.quality import build_report, render_markdown

    def expand(patterns: list[str]) -> list[Path]:
        found = sorted({Path(p) for pattern in patterns for p in glob.glob(pattern)})
        return [p for p in found if not p.name.endswith(".rejected.jsonl")]

    generated = expand(args.generated)
    if not generated:
        return _fail(f"No files match {args.generated}.")
    report = build_report(generated, expand(args.reviewed), args.prepared)
    json_path = args.prepared / "quality_report.json"
    atomic_write_text(json_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    atomic_write_text(args.output_md, render_markdown(report))
    g, p = report["generation"], report["prepared"]
    print(
        f"Candidates {g['candidates']} (accepted {g['accepted']}, rejected {g['rejected']}); "
        f"train {p['train']}, validation {p['validation']}; "
        f"fingerprint {p['dataset_fingerprint']}"
    )
    print(f"Written: {json_path}, {args.output_md}")
    return EXIT_OK


def _cmd_train_sweep(args: argparse.Namespace) -> int:
    from steward.training.config import TrainingConfigError
    from steward.training.sweep import load_sweep, run_sweep, state_path
    from steward.training.trainer import run_training

    configure_logging(level="INFO", fmt=args.log_format)
    root = Path.cwd()
    try:
        sweep = load_sweep(args.sweep)
        state = run_sweep(
            sweep,
            args.sweep,
            root=root,
            only=args.trial or None,
            train=lambda config: run_training(
                config, root=root, prompt_dir=Path("prompts"), config_path=args.sweep
            ),
        )
    except TrainingConfigError as exc:
        return _fail(str(exc))
    except KeyboardInterrupt:
        print("Interrupted. Run the command again to continue the sweep.", file=sys.stderr)
        return EXIT_INTERRUPTED
    for t in state.trials:
        print(
            f"{t.name:<4} {t.status:<10} best val {t.best_val_loss} @ step {t.best_step} "
            f"({t.global_steps} steps) {t.run_dir or t.error or ''}"
        )
    print(f"State: {state_path(root, state.sweep_id)}")
    return EXIT_OK if all(t.status == "completed" for t in state.trials) else EXIT_FAILURE


def _cmd_train(args: argparse.Namespace) -> int:
    from steward.training.config import TrainingConfigError, load_training_config
    from steward.training.trainer import TrainingError, run_training

    configure_logging(level="INFO", fmt=args.log_format)
    try:
        config = load_training_config(args.config)
        if args.max_steps is not None:
            config = config.model_copy(
                update={"train": config.train.model_copy(update={"max_steps": args.max_steps})}
            )
        run_dir = run_training(
            config,
            root=Path.cwd(),
            prompt_dir=Path("prompts"),
            resume=args.resume,
            config_path=args.config,
        )
    except (TrainingConfigError, TrainingError) as exc:
        return _fail(str(exc))
    except KeyboardInterrupt:
        print("Interrupted. Resume with --resume <run dir>.", file=sys.stderr)
        return EXIT_INTERRUPTED
    print(f"Run directory: {run_dir}\nAdapter:       {run_dir / 'adapter'}")
    print("Next: pixi run -e cuda benchmark-gpu (evaluates the new adapter). " + DISCLAIMER)
    return EXIT_OK


def _cmd_ste(args: argparse.Namespace) -> int:
    text = args.text if args.text is not None else sys.stdin.read()
    model = str(args.adapter) if args.adapter is not None else args.model
    return _infer(model, text, args.language, args.revision, args.strict, args.json)


def _cmd_export_ollama(args: argparse.Namespace) -> int:
    from steward.generation.prompts import PromptTemplateError, load_template
    from steward.generation.providers.base import ProviderError
    from steward.inference.api import DEFAULT_PROMPT
    from steward.inference.models import ModelRefError, resolve_model
    from steward.inference.ollama_export import MODELFILE, OllamaExportError, export_for_ollama

    try:
        ref = resolve_model(args.model, args.revision)
        template = load_template(Path("prompts"), DEFAULT_PROMPT)
    except (ModelRefError, PromptTemplateError) as exc:
        return _fail(str(exc))
    if ref.adapter_path is None:
        return _fail(f"{args.model} is not a training run or LoRA adapter directory.")
    out_dir: Path = args.output
    print(f"Exporting {ref.run_dir or ref.adapter_path}", file=sys.stderr)
    try:
        export_for_ollama(
            ref, template, out_dir, gguf_type=args.gguf_type, keep_merged=args.keep_merged
        )
    except (OllamaExportError, ProviderError, OSError) as exc:
        return _fail(str(exc))
    print(f"Wrote {out_dir / MODELFILE}")
    print(f"Import: ollama create steward-ste -f {out_dir / MODELFILE}")
    return EXIT_OK


def _infer(
    model_ref: str,
    text: str,
    language: str | None,
    revision: str | None,
    strict: bool,
    as_json: bool,
) -> int:
    """STE-only inference with the output guard; prints text or the failure."""
    from steward.generation.providers.base import ProviderError
    from steward.inference.api import SteOnlyService
    from steward.inference.guard import GuardPolicy
    from steward.inference.models import ModelRefError, resolve_model
    from steward.inference.transformers_provider import TransformersProvider
    from steward.ste.rules import load_rule_config

    try:
        ref = resolve_model(model_ref, revision)
    except ModelRefError as exc:
        return _fail(str(exc))
    print(f"Model: {ref.run_dir or ref.adapter_path or ref.model}", file=sys.stderr)
    provider = TransformersProvider(revision=ref.revision, adapter_path=ref.adapter_path)
    service = SteOnlyService(
        provider,
        ref.model,
        load_rule_config(Path("configs/ste/terms.toml")),
        prompt_dir=Path("prompts"),
        policy=GuardPolicy(strict_ste=strict),
    )
    try:
        response = service.generate(text, source_language=language)
    except ProviderError as exc:
        return _fail(str(exc))
    finally:
        provider.close()
    if as_json:
        print(json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False))
    elif response.text is not None:
        print(response.text)
        if response.validation.warnings:
            print(f"[flagged: {', '.join(response.validation.warnings)}]", file=sys.stderr)
    else:
        print(
            f"status: {response.status.value}; {response.failure_reason or 'no text returned'}",
            file=sys.stderr,
        )
    return EXIT_OK if response.text is not None else EXIT_FAILURE


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_FAILURE


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch = getattr(args, "dispatch", None)
    if dispatch is not None:
        code: int = dispatch(args)
        return code
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_USAGE
    result: int = handler(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
