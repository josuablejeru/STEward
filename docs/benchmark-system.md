# Benchmark system and automatic hooks

STEward measures every candidate model on one frozen benchmark suite and keeps
an append-only history of the results. The goal is the **smallest** model that
meets documented quality requirements. The latest results are in
[benchmarks.md](benchmarks.md) (generated, do not edit).

> Benchmark scores are automated, heuristic checks. They are not ASD-STE100
> certification. Never claim that a model improved unless a comparable
> benchmark result shows it.

## Quick reference

| Command | What it does |
|---|---|
| `pixi run benchmark` | Benchmark registered candidates that have no current result (needs Ollama). |
| `pixi run steward benchmark run --candidate ID --force` | Re-run one candidate. |
| `pixi run -e cuda benchmark-gpu` | The same, but fail when no GPU is visible. |
| `pixi run steward benchmark history` | List all experiments (`--json` for full records). |
| `pixi run steward benchmark compare [ID] [--against ID]` | Regressions and improvements; exit code 1 if any regression. |
| `pixi run benchmark-plot` | Regenerate SVG/PNG figures from the history. |
| `pixi run benchmark-report` | Regenerate figures, `docs/benchmarks.md`, `summary.json`, and the README block. |
| `pixi run benchmark-status` | Pending changes, stale results, missing candidates, last hook run. |
| `pixi run benchmark-check` | Verify everything (also part of `pixi run check` and CI). |

Exit codes: `0` success, `1` failure (error, failed or unavailable benchmark,
regression for `compare`, verification error), `2` usage error.

## Files

| Path | Content | In Git |
|---|---|---|
| `benchmarks/golden/ste-golden-v0/` | Frozen golden suite (`suite.toml` + `items.jsonl`). | yes |
| `benchmarks/models.toml` | Benchmark candidates (model, prompt, role). | yes |
| `benchmarks/thresholds.toml` | Regression limits and selection requirements. | yes |
| `benchmarks/results/history.jsonl` | Append-only experiment history. | yes |
| `benchmarks/results/summary.json` | Generated summary of current results. | yes |
| `benchmarks/figures/*.svg, *.png` | Generated figures. | yes |
| `docs/benchmarks.md` | Generated report. | yes |
| `configs/ste/terms.toml` | Project terminology for the rule checks. | yes |
| `runs/<run_id>/training_summary.json` | Written by training runs (input contract). | no |
| `.steward/state/` | Local hook state and logs. | no |

Model weights, checkpoints (`runs/`, `checkpoints/`, `models/`), and datasets
(`data/`) are never committed.

## The golden suites

There are four frozen suites under `benchmarks/golden/`. Each registered candidate
names its suite (`suite` in `benchmarks/models.toml`); `primary_suite` selects the
suite used for model selection, the README block, and the first report section.

| Suite | Contract | Items | Purpose |
|---|---|---|---|
| `ste-multilingual-v1` (primary) | `ste_only_text`: plain STE text or `STEWARD_UNSUPPORTED_INPUT` | 44 | The 40 v0 items unchanged, plus 4 out-of-domain instruction requests (answer in another language, reveal internal reasoning) |
| `ste-fidelity-dev-v0` (development) | `ste_only_text` | 30 | Targeted terminology, label, negation, and identity items for tuning and error analysis. Never used for the final comparison. |
| `ste-multilingual-v0` | `ste_only_text` | 40 | STE-only student: English rewrite, German/French/Romanian translation, requirements, identity, out-of-domain (first training run, evaluator 0.1) |
| `ste-golden-v0` | `json_rewrite`: `{"ste_text": ...}` with repair turns | 16 | English rewrite with teacher-style prompts (the first 12 records) |

Results on different suites have different comparability keys and are never
compared or joined in a figure. Both suites are drafts written by Claude; the
multilingual suites also need a native-speaker review of their sources.

The STE-only evaluator is `ste-only-eval-0.3` (output guard `guard-0.3`: terminology
fixtures, untranslated words, negation counts, contradictions, NOTE labels; 0.2 added
the procedure-order check). Results scored with older versions stay in the history
but are not comparable with 0.3 results. Primary metrics: technical errors (count and
rate), critical safety failures, meaning, negation, label and identifier
preservation, unsupported additions, STE-only rate, STE rule pass rate.

### `ste-golden-v0`

* 16 original, fictional items: four procedures, four descriptions, four safety
  notices, and four troubleshooting texts. Each item lists the terms that a rewrite
  must keep (`must_include`) and whether it is safety-critical.
* **Status: draft.** Claude drafted the items on 2026-11-06. A human has not reviewed
  them yet. To promote the suite, a reviewer must check every item, then create
  `ste-golden-v1/` with `review_state = "human_reviewed"`.
* The suite is frozen. `suite.toml` stores the SHA-256 of `items.jsonl`. Any change
  stops all benchmark commands and fails `benchmark-check`.
* The suite is not training data. `benchmark-check` fails when a golden source text
  appears in `data/**/*.jsonl`, `prompts/`, or `configs/`.

## How a benchmark works

For each item, the candidate's prompt template asks for a JSON object
`{"ste_text": ...}` with greedy decoding (temperature 0, fixed seed). If the
output is not valid JSON, the runner sends up to two **repair** messages. Then
the output is scored:

* **Rule checks** (`steward.ste.rules`, version `ste-heuristic-0.1`): sentence
  length (20 words in procedures, 25 in descriptions), possible `-ing` verb forms,
  contractions, and possible passive voice. These are heuristics; see the
  limitations below.
* **Fact proxy:** numbers from the source, `must_include` terms, WARNING/CAUTION
  labels, and negation must be kept. Numbers in the output that are not in the
  source count as **invented** (hallucination proxy).
* **Critical failure:** a lost WARNING/CAUTION label, or any lost or invented fact
  in a safety-critical item. A critical failure always needs human review.

Each record also stores latency, tokens per second (provider-reported generation
time), output length, repetition, repair attempts, model digest, parameter count,
quantization, size on disk, VRAM reported by Ollama, and peak GPU memory
(`nvidia-smi`, all processes).

Metrics that are not available have the value `null` and a note. Examples:
vocabulary compliance (no licensed ASD-STE100 dictionary is configured), and GPU
memory on a machine without a GPU. All current metrics are `automated`. Human
review and LLM-judge metrics use separate `method` values and are never mixed
into automated numbers.

## History rules

* Records are only appended, never changed or deleted. Each line stores a content
  hash (`record_sha256`) and the hash of the line before it. `benchmark-check`
  detects edited, removed, or reordered lines, and the writer refuses to append to a
  broken history.
* The same content is never stored twice.
* Benchmark records and training records are separate types. One checkpoint can
  have many benchmark records.
* Stored statuses are `completed` and `failed`. A run that breaks in the middle is
  stored as `failed`, so the failure stays visible. A run that cannot start (model
  missing, server down, no GPU with `--require-gpu`) stores nothing and exits with
  code 1.
* `stale` and `pending` are derived states, shown by `benchmark-status` and the
  report:
  * **stale:** the result uses another suite or evaluator version, or a fine-tuned
    model was trained on an older dataset revision.
  * **pending:** a registered candidate has no current result, or training changed
    without a new checkpoint.

## Comparability

Two results are compared directly only when their **comparability key** is the
same: suite ID, version, and hash; `EVALUATOR_VERSION`; rule-set version; and the
terminology-file hash. Figures never join results with different keys in one
line. Old results stay in the history and are listed as "not directly comparable".

When you change scoring code in `src/steward/evaluation/` or `src/steward/ste/`
in a way that can change scores, bump `EVALUATOR_VERSION` (in `metrics.py`) or
`RULESET_VERSION` (in `rules.py`). The Stop hook and the report warn when the
evaluator code changed but no version was bumped.

## Regression detection and model selection

Every candidate is compared with:

1. the untuned base model (role `base`, same `base_model`),
2. the previous STEward release (role `release`),
3. the best existing model of equal or smaller parameter count.

`benchmarks/thresholds.toml` sets the limits. Examples: STE pass rate must not
drop by more than 0.05, hallucination rate must not increase, latency must not
increase by more than 25%, and a model that is more than 10% larger must improve
the STE pass rate by at least 0.03. When both results have 95% confidence
intervals and the intervals do not overlap, the finding is marked "CIs separated".
With 16 items, the intervals are wide, so small differences are not meaningful.

The **selected model** is the smallest candidate that meets every requirement in
the `[selection]` table.

## Training integration

Training is never started automatically (`pixi run -e cuda train` is the only
entry point; see [training.md](training.md)). A training run writes `runs/<run_id>/training_summary.json` (schema: `TrainingSummary` in
`steward.evaluation.records`) when it finishes. The summary records the base model
and revision, checkpoint and adapter IDs, dataset version and fingerprint, config
hash, hyperparameters, parameter counts, quantization, and the loss curve (train and
validation loss, learning rate, gradient norm, epoch, step). If the model is served
for evaluation, the summary contains a `serving` block: an Ollama model, or
`provider = "transformers"` with the base model revision, `adapter_path`, the
evaluation `prompt`, and the `suite`. The next Stop hook, or `steward benchmark
run`, then:

1. adds the training run to the history,
2. benchmarks the served model as a `finetuned` (or `quantized`) candidate,
3. regenerates figures and the report, and reports regressions.

Without a `serving` block, the run is recorded and the benchmark is marked pending.

## Claude Code hooks

`.claude/settings.json` (committed) configures two hooks. Both are Python scripts
in `scripts/hooks/` and run through Pixi.

### PostToolUse: change detection

Runs after `Write`, `Edit`, `MultiEdit`, `NotebookEdit`, and `Bash`. It only
records changes; it never runs checks.

* For file tools, it classifies the edited path.
* For `Bash`, it compares a stat snapshot (size and mtime) of the watched folders
  before and after. On the first Bash call there is no snapshot yet, so it uses
  `git status` as a fallback.
* Categories: `data` (`data/`, `src/steward/datasets/`), `generation`
  (`src/steward/generation/`, `prompts/`, `configs/generation/`), `training`
  (`src/steward/training/`, `configs/training/`), `model` (`benchmarks/models.toml`,
  `runs/`, `checkpoints/`, `models/`, `configs/models/`), and `evaluation`
  (`src/steward/ste/`, `src/steward/evaluation/`, `configs/ste/`,
  `benchmarks/golden/`, `benchmarks/thresholds.toml`, `prompts/benchmark_*`).
* The outputs of the benchmark system (`benchmarks/results/`, `benchmarks/figures/`,
  `docs/benchmarks.md`, `.steward/`) are ignored, so the hooks never trigger
  themselves.

### Stop: automatic validation

Runs when Claude finishes a task:

1. Nothing pending: exits at once.
2. The inputs have the same content fingerprints as at the last successful run
   (for example, an edit that was reverted): it clears the pending list and exits.
3. Otherwise it runs only the steps that apply:
   * code, config, or prompt changes: `pixi run lint`, `typecheck`, `test`,
   * `data`: dataset schema and duplicate checks and the golden-leakage check;
     fine-tuned results become stale, and it reports that retraining is required
     (it never assigns scores),
   * `evaluation`/`model`: suite hash, registry, history integrity, version-bump
     warnings, and the historical results that are no longer comparable,
   * `training` without a new checkpoint: reports that the model benchmark is pending,
   * new training summaries: benchmarks the served model (GPU work happens here
     only),
   * then it regenerates figures, the report, the summary, and the README block,
     and verifies them.
4. It reports the result to the user (`systemMessage`): passed steps,
   regressions, improvements, and outstanding work.

**Failures stay visible.** If a step fails, the hook blocks the stop once
(`decision: block`), so Claude must fix the problem or explain it. If the hook
already blocked (`stop_hook_active` is true), it does not block again; it reports
"still FAILING" and keeps the changes pending.

Safeguards: a `STEWARD_HOOK_ACTIVE` reentrancy guard (set for every command the
workflow starts), a non-blocking lock (`.steward/state/workflow.lock`) so two runs
never overlap, content fingerprints so unchanged inputs are not validated again,
and no deletion of history or user files.

### Modes

| `STEWARD_HOOKS` | Behavior |
|---|---|
| `auto` (default) | Validation; model benchmarks only for new trained artifacts. |
| `light` | Validation only; model benchmarks are marked pending. |
| `full` | Also benchmarks registered candidates without current results when `model` or `evaluation` inputs change. |
| `off` | Hooks do nothing. |

Set the variable before you start Claude Code, for example
`STEWARD_HOOKS=light claude`.

### Enable, test, disable, troubleshoot

* **Enable:** the hooks are in `.claude/settings.json`. Start Claude Code in the
  project and approve the project hooks when it asks (see `/hooks`). Run
  `pixi install` first, so `pixi run` can start the scripts.
* **Test by hand:**

  ```bash
  echo '{"hook_event_name":"PostToolUse","tool_name":"Write","tool_input":{"file_path":"configs/ste/terms.toml"},"tool_response":{"success":true}}' \
    | pixi run python scripts/hooks/post_tool_use.py
  pixi run benchmark-status          # shows the pending change
  echo '{"hook_event_name":"Stop","stop_hook_active":false}' | pixi run python scripts/hooks/stop.py
  ```

* **Disable for a while:** `touch .steward/hooks-disabled` (remove the file to
  enable again), or start Claude Code with `STEWARD_HOOKS=off`.
* **Troubleshoot:** read `.steward/state/hook_log.jsonl` (one line per decision) and
  `pixi run benchmark-status`. Delete `.steward/state/` to reset the local state;
  this never touches the history. If the Stop hook reports "another validation run
  is active" and no run is active, delete `.steward/state/workflow.lock`.

### Fallback outside Claude Code

Claude Code hooks do not see changes made in an editor or another terminal. Two
fallbacks cover them:

* `pixi run install-git-hooks` enables `scripts/git-hooks/pre-push`, which runs
  `pixi run benchmark-check` before each push.
* CI (`.github/workflows/ci.yml`) runs `pixi run check`, which includes
  `benchmark-check`.

## GitHub workflows

* `ci.yml`: lockfile check and `pixi run check` (schemas, hash chain,
  reproducibility metadata, golden suite hash, links, and up-to-date figures).
* `benchmarks.yml`: when benchmark files change, it regenerates figures and
  reports and fails if they differ from the committed files.
* `gpu-benchmark.yml`: manual only, on a self-hosted runner with the labels
  `self-hosted, linux, gpu`. GitHub-hosted runners never run model benchmarks or
  training.

## Limitations

* The rule checks are heuristic, not ASD-STE100 Issue 9:
  * The `-ing` check cannot tell verb forms from nouns, so project terms are
    allowed in `configs/ste/terms.toml`.
  * The passive check can flag adjectives and miss some passives.
  * Sentence splitting is approximate.
* The fact proxy does not see a changed meaning that keeps the same words. Its
  negation check flags paraphrases such as "to prevent it from falling" for
  "so that it does not fall" (a false positive).
* The suite has 16 items, so confidence intervals are wide.
* The suite is a draft until a human reviews it.
* Ollama reports parameter counts rounded (for example `2.0B` for Qwen3-1.7B).
  The record stores what the provider reports.
