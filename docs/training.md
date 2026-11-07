# STE-only student: data, training, inference, evaluation

STEward's student model is **not** a general assistant. It takes technical
input in English, German, French, or Romanian and returns only STE-oriented
English, or a controlled failure for out-of-domain input. The output guard
checks every answer; the model alone never decides that text is acceptable.

> Output is STE-oriented. Nothing here is ASD-STE100 certification.

## Pipeline

```text
generate (teacher, Ollama)  ->  repair (linter-guided, teacher)  ->  prepare (filter, split)
   -> train (TRL SFT + PEFT LoRA, GPU)  ->  benchmark (STE-only suite, guard)  ->  history, figures
```

| Step | Command | Output (git-ignored) |
|---|---|---|
| Generate | `pixi run steward generate --model qwen3:8b --think off --config configs/generation/multilingual.toml --seed <n> --count 250 --output data/raw/multilingual/shard-<n>.jsonl` | accepted, rejected, manifest |
| Repair | `pixi run steward dataset repair --model qwen3:8b --rounds 3 --input 'data/raw/multilingual/shard-*.jsonl' --output data/raw/multilingual/repaired/all.jsonl` | all examples, repaired where possible |
| Prepare | `pixi run prepare-sft` (= `steward dataset prepare --input 'data/raw/multilingual/repaired/*.jsonl' --output data/sft/v2`) | `train.jsonl`, `validation.jsonl`, `manifest.json` |
| Train | `pixi run -e cuda train` (= `steward train --config configs/training/qwen3-0.6b-lora.toml`) | `runs/<run_id>/` |
| Evaluate | `pixi run -e cuda evaluate runs/<run_id>` (= `steward evaluate --model runs/<run_id> --baseline --require-gpu`) | appended to `benchmarks/results/history.jsonl`, `runs/<run_id>/benchmark_refs.json` |
| Use | `pixi run -e cuda infer runs/<run_id> --input "text"` (= `steward generate --model runs/<run_id> --input ...`) | text, or a failure status |
| Figures and report | `pixi run benchmark-report`, then `pixi run benchmark-verify` | `benchmarks/figures/`, `docs/benchmarks.md`, README block |

`steward generate` without `--input` is still the synthetic-data generator.
`--model` accepts a run directory, an adapter directory, or a Hugging Face model ID.

Training never starts automatically. The Claude Code Stop hook only records a
finished run (`runs/<id>/training_summary.json`) and benchmarks it.

## Tasks and languages

| Task (`task_type`) | Input | Target | Default share |
|---|---|---|---|
| `plain_to_ste_rewrite` | ordinary English | STE-oriented English | 40% |
| `multilingual_to_ste_translation` | German, French, Romanian prose | STE-oriented English | 30% |
| `requirements_to_ste` | structured `key: value` information, any of the 4 languages | STE-oriented English | 20% |
| `ste_identity` | text that already passes all checks | the same text | 10% |
| `out_of_domain` | hand-written non-technical requests | `STEWARD_UNSUPPORTED_INPUT` | +6% of in-domain |

The shares are set in `configs/sft/default.toml` (`[mix]`,
`out_of_domain_fraction`). The task brief defines only the four in-domain
shares; the out-of-domain share is an assumption of this project. Teacher task
weights are in `configs/generation/multilingual.toml`.

Each record keeps: source language, task type, original input, STE target,
identifier codes found in the source (`technical_entities`), safety
constraints reported by the teacher (`safety_constraints`), prompt versions,
validation results, teacher model, and provenance (run, config hash, seed).

## Quality gates (prepare)

An example is used for training only if all of these hold:

1. Every multitask check passes (`prelint-2`, `src/steward/generation/validation.py`):
   target is English, numbers and identifier codes are kept, no invented
   measurements, WARNING/CAUTION labels match the source (in any of the 4
   languages), negation is kept, no commentary or Markdown.
2. No heuristic STE rule **error** in the target (`require_rule_pass = true`).
3. Not an exact or near duplicate (word-trigram Jaccard >= 0.85) of an
   earlier example, by source or by target.
4. Not an exact or near duplicate of any golden benchmark item or reference.

Known critical failures are never trained on: gate 1 excludes them. Rejected
examples stay in the `*.rejected.jsonl` files for analysis.

The split is by **scenario topic**: all examples about the same fictional
component are in the same split, and identity examples stay with their
parent. `load_prepared` refuses a set whose content hash does not match
its manifest, or whose groups cross the split.

## Training

`configs/training/qwen3-0.6b-lora.toml`: Qwen/Qwen3-0.6B at a pinned
revision, LoRA (r 16, alpha 32, dropout 0.05, all attention and MLP
projections), BF16, learning rate 2e-4, cosine schedule with 5% warmup,
3 epochs, batch 4 x gradient accumulation 4, max length 1024, seed 42,
optional gradient checkpointing. All values are configurable.

Changes from the Milestone 3 brief, and why:

* **Validation split:** `validation_fraction = 0.10`, but the split is by
  scenario topic (no leakage), so the share of *examples* is not exact
  (data/sft/v2: 78 of 689, 11.3%).
* **`warmup_ratio`** is passed as `warmup_steps` (a float below 1 is a ratio
  in transformers 5).
* **Loss masking** uses TRL `completion_only_loss` with prompt/completion
  data; TRL 1.17 stores the mask as `labels = -100`, and the pre-training
  check reads that.
* The first run (`runs/qwen3-0.6b-ste-lora-20261106T193155Z`) used batch 8 x 2,
  max length 1536, and data/sft/v1 (validation fraction 0.12). It is kept
  for comparison.

Failure handling: CUDA out-of-memory is caught, the run is recorded as
`failed` with advice (smaller batch, gradient checkpointing, shorter max
length), and saved checkpoints can be resumed with `--resume runs/<run_id>`.
GPU memory (allocated and peak) is added to every log entry.

* **Chat template:** the base tokenizer's own template, with
  `enable_thinking=False`. For Qwen3 the prompt ends with an empty
  `<think></think>` block, which is the template's non-thinking format.
  No reasoning traces are trained.
* **Loss:** prompt/completion format with TRL `completion_only_loss`. Only
  the STE target and the end-of-turn token carry loss. Before training starts,
  `mask_check.json` proves on 32 processed examples that exactly the target
  tokens are unmasked; training stops if not.
* **Saved per run** (`runs/<run_id>/`): `adapter/` (LoRA weights and
  tokenizer), `checkpoints/` (resumable with `--resume runs/<run_id>`),
  `training_config.json` (resolved config, library versions, dataset
  fingerprint, GPU), `log_history.json` (training and validation losses),
  `mask_check.json`, `training_summary.json` (the benchmark contract: run ID,
  base model and revision, dataset fingerprint, seed, parameters, epochs,
  optimizer steps, duration, peak GPU memory, final training and validation
  loss, adapter size, checkpoint location, status), and after evaluation
  `benchmark_refs.json` (the history records of this run).

## Milestone 4: data v3, best checkpoint, and the sweep

| Step | Command |
|---|---|
| Targeted candidates | `pixi run steward generate --model qwen3:8b --think off --config configs/generation/fidelity.toml --seed <n> --count 500 --max-scenarios 500 --output data/raw/fidelity/shard-<i>.jsonl` (4 shards) |
| Repair | `pixi run steward dataset repair --model qwen3:8b --rounds 3 --config configs/generation/fidelity.toml --input 'data/raw/fidelity/shard-*.jsonl' --output data/raw/fidelity/repaired/all.jsonl` |
| Critique | `pixi run steward dataset critique --critic mistral-nemo:12b --reviser qwen3:8b --only-eligible --input <repaired> --output <dir>/reviewed/all.jsonl` (for the Milestone 3 data and the new data) |
| Prepare v3 | `pixi run prepare-sft-v3` |
| Quality report | `pixi run dataset-report` (writes `data/sft/v3/quality_report.json` and [dataset-quality.md](dataset-quality.md)) |
| Sweep | `pixi run -e cuda train-sweep` |
| Select | `pixi run -e cuda sweep-select` |
| Final comparison | `pixi run -e cuda evaluate runs/<selected>` |

`configs/sft/v3.toml` adds: `require_llm_review` (exclude examples without a passed
critique), `max_warning_caution_ratio = 1.5` (v2 had about 3.6 WARNING targets per
CAUTION target), and `prefer_focus` (when a task is down-sampled, targeted examples
and identity examples with labels, negations, or several lines are kept first).

**Best checkpoint.** `select_best = true` evaluates every `eval_steps`, saves every
`save_steps` (a multiple), loads the checkpoint with the lowest validation loss at
the end, and saves it as `adapter/`. `early_stopping_patience` stops after that many
evaluations without improvement. The summary records `best_step`, `best_val_loss`,
`selected_checkpoint`, `selected_val_loss`, `planned_steps`, and `early_stopped`.

**Sweep.** `configs/training/sweeps/m4-v0.2.toml` has trials A (lr 2e-4, r 16,
2 epochs), B (lr 1e-4, r 16, 3 epochs), and C (lr 1e-4, r 32, alpha 64, 3 epochs).
Each trial overrides only these values of `configs/training/qwen3-0.6b-lora-v0.2.toml`,
so data, split, seed, batch, and length are the same. A failed trial is recorded in
`runs/sweeps/m4-v0.2/sweep.json` and the next trial starts. `sweep-select` evaluates
each trial on the **development** suite `ste-fidelity-dev-v0` and ranks by critical
failures, then technical-error rate, then meaning preservation, then best validation
loss. Training loss is never used, and the final golden suite `ste-multilingual-v1`
is used only once, for the selected trial.

### Milestone 4 results (experimental, heuristic metrics, draft suites)

Data: 2,000 new targeted candidates (1,396 accepted, 604 rejected) plus the 1,412
Milestone 3 candidates; after repair, `prelint-3`, the rule check, and the
independent critique, `data/sft/v3` has 868 training and 111 validation examples
(fingerprint `556cf4912f06…`). See [dataset-quality.md](dataset-quality.md).

| Trial | lr | r / alpha | Epochs (run) | Steps | Best val loss (step) | Dev: tech. errors | Dev: critical | Rank |
|---|---|---|---|---|---|---|---|---|
| A | 2e-4 | 16 / 32 | 2 (2.0) | 110 | 0.268 (110) | 7 / 27 | 14 | 3 |
| B | 1e-4 | 16 / 32 | 3 (2.74, early stop) | 150 | 0.271 (110) | 6 / 27 | 14 | 2 |
| C | 1e-4 | 32 / 64 | 3 (2.74, early stop) | 150 | 0.261 (110) | 4 / 27 | 13 | **1** |

Final comparison of the selected trial C on `ste-multilingual-v1` (evaluator
`ste-only-eval-0.3`) is in [benchmarks.md](benchmarks.md): technical errors 10 → 6
items and critical safety failures 11 → 6 against v0.1, with no metric regressing.
With 34 in-domain items the 95% intervals overlap, so this is a measured but not
statistically established improvement. No candidate meets the selection
requirements yet (critical failures must be 0).

Remaining failure types of v0.2-C: Romanian input copied untranslated, French
ATTENTION (CAUTION) written as WARNING, the German *22-mm-Schlüssel* as "key",
lost identifiers in structured requirements, and an added safety label.

Next experiment (recommended): more Romanian and requirements data (the binding
task; 369 English rewrites were dropped to keep the mix), contrastive
ATTENTION/AVERTISSEMENT pairs, a requirements prompt that lists every identifier,
a native-speaker review of the dev and golden suites, and then the same sweep for
Qwen3-1.7B (prompted but not fine-tuned, it already has 7 critical failures against 9
for the untuned 0.6B model).

## Inference and the output guard

Quick test with the newest completed run (no path needed):

```bash
pixi run -e cuda ste "Make sure that the pressure is released before you remove the pump."
pixi run -e cuda ste --model runs/<run_id> "..."   # a specific run
```

`--model latest` (the default) picks the newest completed run in `runs/` by the
UTC timestamp at the end of the run ID. That is the newest run, not
necessarily the sweep-selected one. The command prints which run it used.

### Local Ollama model (for interactive testing)

```bash
pixi run -e ollama ollama-serve      # in its own terminal
pixi run ollama-create               # latest run -> Ollama model "steward-ste"
pixi run ollama-chat                 # or: pixi run ollama-create runs/<run_id> my-name
```

`ollama-create` merges the LoRA adapter into the base model, converts it to
GGUF (bf16) with llama.cpp's converter (tag `b12004`, pinned in `pixi.lock`), and
writes `runs/ollama-export/{model.gguf,Modelfile,export.json}`. Each export
replaces the previous one. `export.json` records the run, adapter, base revision,
and prompt version. The Modelfile contains the exact `ste_only@1` system prompt,
the Qwen3 chat format with thinking disabled, and greedy sampling (seed 1234).
Only the latest chat message is sent, because the model was trained on
single-turn examples.

Limitations: Ollama output is **not** checked by the output guard. It is an
unchecked candidate. Use `pixi run -e cuda ste` for checked output and the
benchmarks for metrics. On three test inputs, Ollama output was identical to the
Transformers backend, but that is not a general guarantee.

`steward.inference.api.SteOnlyService.generate(text, source_language=None)`
returns an `SteResponse` (the API's generation result):

* `status`: `accepted`, `flagged`, `rejected`, `unsupported_input`, or `invalid_input`,
* `text`: the STE-oriented text, only for `accepted` and `flagged`,
* `violations`: heuristic STE rule violations of the output,
* `validation_version`: guard and rule-set versions (for example `guard-0.2+ste-heuristic-0.1`),
* `model_id`: base model, revision, and adapter hash,
* `failure_reason`: why no text was returned,
* `generation`: raw model output, model and adapter identity, prompt, latency,
* `validation`: every guard check with its result, kind (deterministic or heuristic), and details.

The guard (`src/steward/inference/guard.py`) never edits the output. Critical
checks: output is English, no commentary/Markdown/reasoning, all numbers,
measurements (value and unit), identifier codes, WARNING/CAUTION labels, and
negation are kept, nothing is added (numbers, codes, labels), facts from
different steps or sentences keep their order (`sequence_preserved`; not
applied to `key: value` requirements input, which is not a sequence), and the
output is not much shorter than the input. A failed critical check means `rejected`:
no text is returned. Heuristic STE rule errors mean `flagged` (or `rejected`
with `--strict`). A truncated answer is rejected.

`guard-0.3` adds: exact negation counts, terminology fixtures (a known wrong
rendering is critical), untranslated source-language words, contradictory
instructions, NOTE labels, and a non-critical warning when an English source verb has
no equivalent in the output.

Limitations: the checks compare literal tokens and word lists. They do not
detect a reversed instruction that keeps the same words ("Exceed 3.0 psi"
instead of "Do not exceed 3.0 psi" is caught only through the negation
check), a wrong translation of a part name, or a step in the wrong order.
Human review is still necessary.

## Evaluation

The STE-only suite `benchmarks/golden/ste-multilingual-v1` (44 draft items,
contract `ste_only_text`; the 40 items of v0 plus 4 requests to answer in
another language or to reveal reasoning) is the primary suite. Untuned and fine-tuned models
use the same prompt (`ste_only@1`), decoding (greedy, max 768 tokens), and
scoring. Metrics: STE-only output rate, non-English rate, commentary rate,
out-of-domain controlled-failure rate, false refusals, guard pass rate,
unnecessary rewriting of valid STE, rule violations per 100 words, meaning
preservation (fact proxy), numerical accuracy, identifier preservation,
negation preservation, procedure order, unsupported additions, missing
information, chrF against draft references, critical safety failures,
latency, tokens per second, peak GPU memory, parameters, size on disk, and
adapter size, per task and per source language. Vocabulary compliance and
instruction-structure violations are reported as unavailable (no licensed
dictionary; no such rule yet). All are automated
heuristics; no human-reviewed metric exists yet.

See [benchmarks.md](benchmarks.md) for the current results and
[benchmark-system.md](benchmark-system.md) for the history and hooks.
