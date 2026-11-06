# STEward — Agent Instructions

## Mission

STEward is an open-source research and engineering project for generating, rewriting, and evaluating aerospace technical documentation in ASD-STE100 Simplified Technical English (STE). The intended deliverables are a reproducible synthetic-data pipeline, an independently testable STE-oriented static linter, a small fine-tuned language model, evaluation artifacts, and publishable Hugging Face repositories.

The current reference is ASD-STE100 Issue 9 (January 2025). Always distinguish *STE-oriented* output from verified or certified STE compliance. An LLM, linter, or benchmark score alone must never be represented as certification.

## Non-negotiable development rules

1. **Use Pixi exclusively for project environment and dependency management.** Maintain `pixi.toml` and commit `pixi.lock`. Prefer `pixi add <package>` for conda packages, `pixi add --pypi <package>` for PyPI packages, and `pixi remove` for removal. Run Python, tests, linters, tools, and scripts through `pixi run ...`. Define repeatable commands under Pixi tasks. Do not introduce Poetry, Pipenv, `uv`, `requirements.txt` installation flows, manual `venv` setup, or `pip install` instructions. A Pixi-managed PyPI dependency is permitted; invoking pip outside Pixi is not. Do not modify the global Python environment.
2. **Use Python 3.12 as the default baseline**, unless a documented dependency or GPU compatibility issue demands a different supported version. Keep code typed, testable, and straightforward; prefer standard library solutions where possible.
3. **Favor reproducibility.** Fix random seeds where meaningful, version configurations and prompt templates, record tool/model versions, record dataset generation provenance, and commit the Pixi lockfile. Keep heavyweight model weights, caches, and generated datasets out of Git.
4. **Make offline/local execution possible.** Hosted teacher providers may be optional adapters; the core workflow must support a locally running model and must not require paid APIs.
5. **No silent compliance claims.** Label outputs `candidate`, `linted`, `reviewed`, or `validated_against_available_checks` as appropriate. Never label text `ASD-approved`, `certified`, or `fully compliant` without independent substantiation.
6. **Never publish secrets or unlicensed materials.** Respect model licenses, teacher-provider terms, dataset provenance, third-party documentation rights, and privacy. Do not ingest real maintenance manuals, proprietary aircraft data, or the full copyrighted STE standard into a public dataset without permission.
7. **Avoid fabricated technical authority.** All synthetic scenarios are fictional and explicitly labeled. Never imply fabricated torque values, clearances, safety steps, or maintenance actions are approved aircraft instructions.

## Technical scope and design

Implement the project as composable Python packages with clear interfaces:

- `generation`: teacher-model adapters (local first; hosted optional), prompt templates, structured generation, retries, batch processing, and provenance.
- `datasets`: schemas, validation, deduplication, splits, versioning, and serialization to JSONL/Parquet or Hugging Face Datasets.
- `ste`: deterministic rule checks, vocabulary and terminology interfaces, sentence analysis, severity levels, and machine-readable diagnostics.
- `evaluation`: held-out benchmark suite, baseline comparison, rule-violation metrics, meaning-preservation checks, error analysis, and human-review workflow.
- `training`: LoRA/QLoRA supervised fine-tuning, configurations, checkpointing, metrics, and deterministic seeds where possible.
- `publishing`: Hugging Face model/dataset cards, licensing/provenance checks, packaging, and upload tooling (opt-in only).
- `cli`: ergonomic commands connecting the modules.

Preferred source layout:

```text
steward/
├── AGENTS.md
├── pixi.toml
├── pixi.lock
├── README.md
├── LICENSE
├── src/steward/
│   ├── cli.py
│   ├── generation/
│   ├── datasets/
│   ├── ste/
│   ├── evaluation/
│   ├── training/
│   └── publishing/
├── tests/
├── configs/
├── prompts/
├── examples/
├── docs/
└── .github/workflows/
```

Use absolute imports from `steward`. Keep training, inference, and validator backends behind small, replaceable interfaces. Do not add dependencies or services without a clear justification.

## Pixi requirements

- Bootstrap a project using Pixi and use `pixi.toml` as the main environment/task manifest; `pyproject.toml` may contain Python build metadata if needed, but it must not introduce a competing environment manager.
- Define Pixi tasks for `format`, `lint`, `typecheck`, `test`, `generate`, `validate`, `evaluate`, and `train` when those entry points exist. Add `publish` only with explicit opt-in controls.
- Prefer lightweight default/dev environments and a separate Linux CUDA training environment/feature when GPU dependencies require it. Do not make CPU-only lint/test runs install large CUDA packages unnecessarily.
- Default commands for contributors should look like `pixi install`, `pixi run test`, `pixi run lint`, and `pixi run -e cuda train` where configured.
- CI must check the lockfile with `pixi lock --check`, and run appropriate Pixi tasks. Never claim a task succeeds unless it actually ran.
- If a dependency needs a nonstandard installation method, investigate Pixi/conda/PyPI compatibility first and document any unavoidable exception before changing the policy.

## STE language and validation rules

- Treat the official ASD-STE100 Issue 9 as the authoritative human reference, not an automatically redistributable data source.
- Separate deterministic checks (e.g., sentence length, explicitly configured vocabulary, certain patterns) from heuristic NLP checks (e.g., passive voice, part-of-speech, semantic ambiguity). Report false-positive/false-negative limitations.
- Support project-specific technical terminology through configurable, traceable dictionaries. Do not assume that every unfamiliar technical noun violates STE.
- Include rule identifiers, text spans, actionable explanations, severity, and confidence/limitations in linter reports when available.
- Preserve meaning, negation, numerical values, units, ordering, warnings, conditions, and prerequisites when rewriting. If preservation cannot be assessed confidently, flag the result for human review.
- Do not claim that an existing open-source STE checker implements all of Issue 9. Audit candidate implementations and licenses before adopting them.
- Never substitute readability statistics, an LLM-as-judge, or successful syntax checks for standards verification.

## Synthetic dataset policy

- Generate original and fictional technical-writing tasks; do not paraphrase copyrighted manuals into a public dataset without confirmed permission.
- Support tasks such as plain-English-to-STE rewriting, error correction, procedural text generation, technical descriptions, and rule explanation.
- Store at minimum: example ID; task type; source prompt; target response; teacher provider and model ID; generator prompt version; seed or sampling parameters where available; timestamp; linter results; review state; provenance and license metadata.
- Retain rejected examples separately for analysis; do not train on unreviewed outputs by default.
- Use schema validation, exact/near-duplicate removal, and stable split assignments. Keep related prompt families together across splits to prevent leakage.
- Use a multi-stage filter: schema check → deterministic linter → semantic/technical fidelity check → sampled human review. Do not assume teacher-generated examples are correct.
- Track acceptance rates and error categories. Prefer a smaller high-quality corpus to an unverified large corpus.
- Keep a fixed, independently authored holdout set; never recycle it as teacher prompt material or training data.

## Model strategy and hardware

- Target a single NVIDIA RTX 5090 with 32 GB VRAM for local prototyping. Confirm supported CUDA/PyTorch versions before choosing a training stack.
- Benchmark quantized local teacher candidates before generating a large corpus; record throughput, peak VRAM, output quality, and review burden. Avoid hardcoding a single model before evidence supports it.
- Start with a small permissively licensed instruction-tuned student (approximately 3B–8B parameters). Use LoRA/QLoRA rather than full pretraining for the first release.
- Baseline-test the untouched student, prompted student, and fine-tuned student on the same held-out evaluation set.
- Make all train/generate jobs resumable with explicit output directories. Do not assume GPU availability in unit tests or CI.

## Evaluation and publication

Track separate measures for rule violations, approved-vocabulary use where legally/configurably available, semantic fidelity, instruction following, human acceptance, latency, and resource usage. Report denominators and confidence/limitations, not just a single aggregate compliance score.

Hugging Face releases should include:

- Reproducible configuration and documented base model and adaptation method.
- Model card: intended uses, limitations, risks, standard reference, dataset sources, benchmark setup, and exact non-certification disclaimer.
- Dataset card: synthesis method, provider/model provenance, quality gates, rights review, potential bias, and split methodology.
- License compatible with all source weights, training data, and usage terms; do not assume synthetic data is automatically free of restrictions.
- An explicit manual release gate. Do not run `huggingface-cli upload`, `hf upload`, or push assets without a direct request to publish.

## Coding standards and workflow for agents

- Before implementing, inspect the repository, Pixi configuration, tests, and existing interfaces. Prefer minimal focused changes over broad rewrites.
- For nontrivial changes, state a short plan, then implement, test, and summarize results.
- Use Ruff for formatting/linting, pytest for tests, and a type checker such as Pyright or mypy when configured. All checks run via Pixi tasks.
- Add unit tests for new logic and regression fixtures for discovered linter errors; make external model/API tests optional and explicitly marked.
- Use typed data models (e.g., Pydantic) for dataset and API boundaries. Avoid hand-built, loosely structured JSON records.
- Handle model/API failures, rate limits, malformed outputs, and interrupted generation safely. Log enough for reproducibility but never secrets or restricted input text.
- Avoid making network calls in default unit tests. Mock provider adapters where practical.
- Add useful docstrings and concise README instructions; prefer simple tools and maintainable abstractions over premature orchestration.
- Never invent test results, benchmark figures, licensing approvals, or standards certifications. Explicitly state what was not run or verified.

## Definition of done

A change is complete only when it has: (1) a clear purpose, (2) implementation and relevant tests, (3) Pixi-managed dependencies/tasks, (4) documented input/output contracts where appropriate, (5) a reproducible local verification command, and (6) explicit limitations for validation, data rights, or safety where relevant.

When requirements are underspecified, make a conservative, reversible engineering decision, document the assumption, and continue unless doing so could expose restricted material or publish external assets.
