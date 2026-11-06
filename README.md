# STEward

STEward is an open-source research and engineering project for generating,
rewriting, and evaluating aerospace technical documentation in
ASD-STE100 Simplified Technical English (STE). The reference is
ASD-STE100 Issue 9 (January 2025).

> **Disclaimer:** STEward output is *STE-oriented*. A linter result, model
> output, or benchmark score from this project is not a certification of
> ASD-STE100 compliance. All synthetic scenarios are fictional and are not
> approved aircraft instructions.

## Status

Early foundation. The package layout, CLI entry point, and quality tooling
exist. Generation, linting, evaluation, training, and publishing are not
implemented yet.

## Quick start

STEward uses [Pixi](https://pixi.sh) for all environment and dependency
management. Do not use pip, venv, Poetry, or uv directly.

```bash
pixi install            # create the default (dev) environment
pixi run steward info   # show version and runtime information
pixi run check          # format check, lint, type check, tests
```

## Pixi environments

| Environment | Features     | Purpose                                         |
|-------------|--------------|-------------------------------------------------|
| `default`   | `dev`        | CPU-only development: Ruff, mypy, pytest.       |
| `cuda`      | `dev`,`cuda` | Linux GPU stack (PyTorch, CUDA >= 12.8) for future training. |

The `cuda` environment targets an NVIDIA RTX 5090 (Blackwell), which needs
CUDA 12.8 or later. CI and unit tests never require a GPU.

```bash
pixi run -e cuda cuda-info   # confirm PyTorch can see the GPU
```

## Tasks

| Task           | What it does                                  |
|----------------|-----------------------------------------------|
| `format`       | Format code and sort imports with Ruff.       |
| `format-check` | Check formatting without changes.             |
| `lint`         | Lint with Ruff.                               |
| `typecheck`    | Strict type check with mypy.                  |
| `test`         | Run the CPU-only pytest suite.                |
| `test-cov`     | Run tests with a coverage report.             |
| `check`        | Run all of the checks above (used in CI).     |

Tests that need a GPU or network access must be marked `gpu` or `network`.

## Layout

```text
src/steward/
├── cli.py          # `steward` command
├── environment.py  # runtime info for provenance records
├── generation/     # teacher-model adapters and prompts
├── datasets/       # schemas, validation, splits
├── ste/            # STE-oriented rule checks
├── evaluation/     # benchmarks and metrics
├── training/       # LoRA/QLoRA fine-tuning (future)
└── publishing/     # Hugging Face cards and opt-in upload
```

See [AGENTS.md](AGENTS.md) for the full project rules.

## License

Code is licensed under the [Apache License 2.0](LICENSE). The ASD-STE100
standard itself is copyrighted by ASD and is not included in this repository.
