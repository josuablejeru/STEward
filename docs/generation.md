# Synthetic dataset generation

`steward generate` uses a local teacher model to create **fictional**
plain-English aerospace maintenance passages, then rewrite candidates in an
STE-oriented style. Every record has `review_state: "candidate"`. A
generated record is **not** verified or certified ASD-STE100 text.

## Quick start

```bash
# Terminal 1: local Ollama server (CUDA build from conda-forge, via Pixi)
pixi run -e ollama ollama-serve

# Terminal 2
pixi run -e ollama ollama-pull qwen3:8b
pixi run steward generate --model qwen3:8b --count 100 --output data/raw/examples.jsonl --think off
```

`--think off` turns off the "thinking" mode of reasoning models such as
Qwen3. It makes generation faster. Leave it out for other models.

## How one example is made

1. **Scenario.** A category, fictional subject, reader, and length are
   sampled from `configs/generation/default.toml`. The choice depends only
   on `seed` and the example index.
2. **Source stage.** Prompt `prompts/source_passage/v1.toml` asks for an
   original, fictional, unedited passage (JSON: `title`, `text`).
3. **Rewrite stage.** Prompt `prompts/ste_rewrite/v1.toml` asks for an
   STE-oriented rewrite (JSON: `ste_text`).
4. **Pre-lint sanity checks** (`steward.generation.validation`). These are
   not an STE linter:

   | Check                           | Severity | Kind          |
   |---------------------------------|----------|---------------|
   | `candidate_nonempty`            | error    | deterministic |
   | `numbers_preserved`             | error    | deterministic |
   | `candidate_differs_from_source` | warning  | deterministic |
   | `safety_labels_preserved`       | warning  | deterministic |
   | `negation_preserved`            | warning  | heuristic     |
   | `length_ratio`                  | warning  | heuristic     |
   | `sentence_length`               | warning  | heuristic     |

   A failed *error* check rejects the example. Failed warnings are stored on
   the accepted record for later review.
5. **Duplicate check.** An example is a duplicate when its source or
   candidate is identical to an earlier one after normalization (case,
   punctuation, spacing). Paraphrases are *not* detected yet.

## Multitask, multilingual generation (schema 1.1)

`configs/generation/multilingual.toml` adds a `[tasks]` table. Each scenario
then also gets a task type and a source language:

* `plain_to_ste_rewrite`: English prose to STE-oriented English,
* `multilingual_to_ste_translation`: German, French, or Romanian prose to English,
* `requirements_to_ste`: structured `key: value` information in any of the four
  languages to STE-oriented English.

The source stage uses `source_passage@2` (it also returns identifier codes and
a short English summary of the safety constraints), and the rewrite stage
uses `ste_rewrite@2`. The per-task format guidance and task instructions are
in the config file, so they are part of `config_sha256`. Identifier codes
reported by the teacher are kept only if they occur in the source text.

These examples use the stricter `prelint-2` checks: a lost number,
identifier code, WARNING/CAUTION label, or negation, an invented
measurement, commentary, or a non-English target is an `error` and the
example is rejected. Use `--think off` with Qwen3 teachers: thinking mode is
slower and its long outputs can be cut off.

Generation runs one request at a time. For more throughput, start the
Ollama server with `OLLAMA_NUM_PARALLEL=4` and run several generators with
different `--seed` values and output files (shards); `steward dataset
prepare` removes duplicates across shards.

### Linter-guided repair

`steward dataset repair` sends each accepted candidate with heuristic STE
rule errors back to the teacher with the exact rule problems
(`ste_repair@1`). A repaired text replaces the candidate only if it still
passes every `prelint-2` check and has fewer rule errors. The old text is kept
in `pre_repair_candidate`; the repair prompt and sampling are recorded.
The output file contains every input example, repaired or not.

### Targeted (fidelity-focused) generation

`configs/generation/fidelity.toml` adds a `[focus]` mix that oversamples the failure
modes found in the Milestone 3 benchmark. Each scenario gets one focus (own random
stream, so it is reproducible):

| Focus | Applies to | What the prompts add |
|---|---|---|
| `terminology` | German, French, Romanian translation and requirements | Use a context-dependent term (for example *Mutter* = nut, *à mains nues* = with bare hands, *panou* = panel). The intended English term is given to the rewrite stage and is required in the target (`required_terms_present`). |
| `safety_label` | all tasks | An exact labelling (CAUTION, NOTE, WARNING, or two of them), with CAUTION oversampled. The rewrite must keep each label exactly. |
| `negation_condition` | all tasks | A prohibition (do not, must not, never) and a condition (before, after, only when, unless, if, while). |

The focus text is appended to the existing `source_passage@2` and `ste_rewrite@2`
prompts through the config (so it is part of `config_sha256`), and the sampled
focus, option, and required terms are stored in `provenance.scenario`. The
mappings are context dependent; they are not universal translations.

### Stricter checks (`prelint-3`)

`prelint-3` adds: exact WARNING/CAUTION/NOTE counts, negation counts (French and
Romanian two-part negations count once), terminology fixtures
(`configs/ste/terminology.toml`: a known wrong rendering such as *mother* for
*Mutter* is an error), untranslated source-language words (word frequencies from
`wordfreq`), contradictory instructions, the required focus terms, and that the
focus term occurs in the source. The same checks run in the inference output guard
(`guard-0.3`).

| Kind | Checks |
|---|---|
| deterministic | numbers, invented measurements, identifiers, safety-label counts, fixture terms, required terms, contradictions |
| heuristic | target language, negation counts, untranslated words, sentence length |
| llm_judged | independent critique (below) |
| human_reviewed | none yet |

### Independent critique and revision

`steward dataset critique --critic mistral-nemo:12b --reviser qwen3:8b` compares each
source text (in its original language) with the English target. The critic is from
another model family than the teacher. Critical findings (mistranslated term,
untranslated word, negation, condition, label, value, missing or added step, order,
contradiction) go to the reviser once per round. A revision is kept only if every
`prelint-3` check passes, it has no more STE rule errors, and a new critique has no
critical finding. The result is stored in `llm_review` with `method = "llm_judged"`.
It is never proof of correctness and never a human review. `--only-eligible`
skips examples that fail `prelint-3` or the rule check (they are excluded later anyway).

Note: the Pixi Ollama build (0.17.0) cannot download `gemma3`; `mistral-nemo:12b`
(Apache-2.0) is used as the critic.

See [training.md](training.md) for preparation of the training set.

## Output files

For `--output data/raw/examples.jsonl`:

| File                       | Content                                                   |
|----------------------------|-----------------------------------------------------------|
| `examples.jsonl`           | Accepted `DatasetExample` records (schema `1.0`).         |
| `examples.rejected.jsonl`  | `RejectedExample` records with a reason (for analysis).   |
| `examples.manifest.json`   | One record per run: counts, rejection reasons, speed, teacher digest, config, git commit. |

All three files are written atomically (temporary file + rename) every
`--checkpoint-every` scenarios and at the end. `data/` is git-ignored, and
the CLI warns if `--output` is in a location that Git would track.

Each record stores: example ID (derived from the source text), task type and
category, source text, STE candidate, teacher provider/model/digest, prompt
template IDs, versions, and SHA-256, sampling parameters with the actual
seed of each stage, attempts, latency, token counts, validation results,
timestamp, run ID, config hash, steward version, git commit, and
license/rights status (`pending-rights-review`).

## Reproducibility

* The scenario and seed for example *n* come from `seed` and *n* only.
* A retry uses `seed + attempt - 1`. The record stores the seed that worked.
* The teacher model digest is stored, so a changed model is visible.
* Outputs from GPU inference are not always bit-identical between runs,
  even with the same seed.

## Resume, overwrite, and failures

* An existing output is never replaced by mistake. Use `--resume` to continue
  to `--count` total, or `--overwrite` to start again.
* Retryable errors (connection, timeout, HTTP 408/429/5xx, malformed or cut-off
  output) are retried with exponential backoff (`[request]` in the config).
* After `max_consecutive_failures` provider failures in a row, the run stops
  and saves its progress. A missing model stops the run before any calls.
* A run stops as `incomplete` after `3 x remaining + 5` scenarios
  (`--max-scenarios`) when too many examples are rejected.
* Exit codes: `0` completed, `1` error or incomplete, `2` usage error,
  `130` interrupted (progress saved).

Logs go to stderr. Use `--log-format json` for one JSON object per line.
Logs do not contain generated text.

## Adding a provider

Implement `TextGenerationProvider` (`generate`, `describe_model`, `close`)
in `src/steward/generation/providers/` and register it in `PROVIDERS`. Hosted
providers must stay optional.

## Changing prompts

Never edit a published template version. Copy `v1.toml` to `v2.toml`,
change it, and point the config at `name@2`.

## Tests

```bash
pixi run test                                      # mocked Ollama, no network
STEWARD_OLLAMA_MODEL=qwen3:8b pixi run test-ollama # opt-in, needs a live server
```

## Limitations

* The teacher writes the source and the rewrite. Both can contain errors.
  Nothing is human-reviewed yet.
* The sanity checks do not detect wrong meaning, invented steps, or
  vocabulary outside the ASD-STE100 dictionary. The dictionary is not
  included and is not redistributed.
* Sentence splitting is approximate.
