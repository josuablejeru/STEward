# Testing the Model Locally

This page explains how to try a trained STEward model on your own machine:
from the command line with the output guard, or interactively in Ollama.

> **Not certification.** STEward output is STE-oriented. It is not certified
> or approved ASD-STE100 compliance. Ollama output is also **not** checked by
> the STEward output guard. Treat every result as an unchecked `candidate`.

## Contents

- [Background: base model and adapter](#background-base-model-and-adapter)
- [Requirements](#requirements)
- [Option 1: command line with the output guard](#option-1-command-line-with-the-output-guard)
- [Option 2: interactive chat in Ollama](#option-2-interactive-chat-in-ollama)
- [Which run is used](#which-run-is-used)
- [How the Ollama export works](#how-the-ollama-export-works)
- [Command reference](#command-reference)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

## Background: base model and adapter

A STEward training run does not change the base model. LoRA (Low-Rank
Adaptation) freezes the base model, `Qwen/Qwen3-0.6B`, and trains a small set
of extra weights: the **adapter**.

| Part | Where | Size |
|------|-------|------|
| Base model | Hugging Face cache (`Qwen/Qwen3-0.6B`, fixed revision) | about 1.2 GB |
| Adapter | `runs/<run_id>/adapter/adapter_model.safetensors` | tens of MB |

To use the model, STEward loads the base model and applies the adapter. The
`ste` command does this automatically. Ollama cannot apply a Qwen3 adapter, so
the Ollama export **merges** the adapter into the base model and converts the
result to one GGUF file.

## Requirements

- Linux or WSL2 with an NVIDIA GPU (the CUDA environment needs CUDA 12.8 or later).
- [Pixi](https://pixi.sh) installed.
- At least one completed training run in `runs/`. To train one:
  `pixi run -e cuda train` (it never runs automatically).

Install the environments once:

```bash
pixi install -e cuda     # PyTorch, Transformers, PEFT, GGUF converter
pixi install -e ollama   # local Ollama server
```

## Option 1: command line with the output guard

This is the recommended way to check output. The guard checks every result
and labels it, for example `accepted`, `flagged`, or `rejected`.

```bash
pixi run -e cuda ste "Before you start working on the hydraulic pump, you should make sure that the system pressure has been released, otherwise there is a risk of injury."
```

Example output (run `qwen3-0.6b-ste-lora-v0.2-m4-v0.2-c-20261107T011144Z`):

```text
Model: .../runs/qwen3-0.6b-ste-lora-v0.2-m4-v0.2-c-20261107T011144Z
[flagged: actions_present]
Before you start working on the hydraulic pump, release the system pressure. Otherwise, there is a risk of injury.
```

The text goes to standard output. The model line and the guard status go to
standard error.

Useful options:

| Option | Effect |
|--------|--------|
| `--model runs/<run_id>` | Use a specific run instead of the latest. |
| `--model Qwen/Qwen3-0.6B` | Use the untuned base model (for comparison). |
| `--language de` | Set the source language (`en`, `de`, `fr`, `ro`). |
| `--strict` | Reject output with heuristic STE rule errors. |
| `--json` | Print the full response, with every guard check. |

Text can also come from standard input:

```bash
cat input.txt | pixi run -e cuda ste
```

Each call loads the model again, so one call takes about one minute.

## Option 2: interactive chat in Ollama

Use this to try many inputs quickly. The model stays loaded between messages,
so you do not wait for a model load on each input.

**Step 1.** Start the Ollama server in its own terminal:

```bash
pixi run -e ollama ollama-serve
```

**Step 2.** Export the latest run and import it into Ollama (about 2–3 minutes):

```bash
pixi run ollama-create
```

This creates the Ollama model `steward-ste`. The command ends with `success`.

**Step 3.** Chat with the model:

```bash
pixi run ollama-chat
```

Type or paste a technical text. Each message is converted on its own: the
model does not see earlier messages. Off-topic input returns
`STEWARD_UNSUPPORTED_INPUT`. Type `/bye` to exit.

You can also use any Ollama client, such as the HTTP API:

```bash
curl -s localhost:11434/api/chat -d '{
  "model": "steward-ste",
  "stream": false,
  "messages": [{"role": "user", "content": "Ventil V-12 nicht öffnen, solange der Druck über 2 bar liegt."}]
}'
```

**After the next training run**, run `pixi run ollama-create` again. It replaces
`steward-ste` with the new latest run.

To keep several versions side by side, give each one a name:

```bash
pixi run ollama-create runs/<run_id> steward-ste-trial-a
pixi run ollama-chat steward-ste-trial-a
```

## Which run is used

All commands use `latest` by default: the newest **completed** training run in
`runs/`. "Newest" means the UTC timestamp at the end of the run ID, for
example `...-20261107T011144Z`. Failed runs are skipped.

- The latest run is not always the best run. After a sweep, check which trial
  was selected in `runs/sweeps/<sweep_id>/sweep.json` (`"selected"`), and pass
  that run with `--model` or as the first argument of `ollama-create`.
- Every command prints the run it used (`Model: ...` or `Exporting ...`).
- `runs/ollama-export/export.json` records which run is in the current Ollama export.

## How the Ollama export works

`pixi run ollama-create [run] [name]` runs two steps:

1. `export-ollama` (CUDA environment):
   1. Load the base model at the revision recorded in the run, and apply the adapter.
   2. Merge the adapter into the weights (on the CPU; about 2 GB RAM).
   3. Convert the merged model to GGUF (bf16, lossless for this model) with
      llama.cpp's converter, pinned to tag `b12004` in `pixi.lock`.
   4. Write the Ollama `Modelfile` and `export.json`, and delete the merged
      intermediate files.
2. `ollama create <name> -f runs/ollama-export/Modelfile` (Ollama environment).

Output in `runs/ollama-export/` (not in Git; each export replaces the last one):

| File | Contents |
|------|----------|
| `model.gguf` | The merged model (about 1.2 GB). |
| `Modelfile` | Prompt template and sampling parameters for Ollama. |
| `export.json` | Provenance: run, adapter, base revision, prompt version, GGUF type, converter. |

The `Modelfile` reproduces the training and inference contract:

- The exact `ste_only@1` system prompt from `prompts/ste_only/v1.toml`. It is
  fixed in the template, so `/set system` has no effect.
- The Qwen3 chat format with thinking disabled (an empty `<think>` block),
  the same as during training.
- Single-turn use: only the latest user message goes to the model.
- Greedy decoding: `temperature 0`, `top_p 1`, `seed 1234`, at most 768 new tokens.

### Export options

Run the export alone with `pixi run -e cuda steward export-ollama [run] [options]`:

| Option | Default | Effect |
|--------|---------|--------|
| `--output DIR` | `runs/ollama-export` | Output directory. |
| `--gguf-type` | `bf16` | `bf16`, `f16`, or `q8_0` (half the size, slightly lossy). |
| `--keep-merged` | off | Keep the merged safetensors in `DIR/merged`. |
| `--revision` | from the run | Override the base model revision. |

Then import with `pixi run -e ollama ollama create <name> -f DIR/Modelfile`.

## Command reference

| Command | Environment | What it does |
|---------|-------------|--------------|
| `pixi run -e cuda ste "text"` | cuda | Convert text with the output guard (latest run). |
| `pixi run -e cuda export-ollama [run]` | cuda | Merge, convert to GGUF, and write the Modelfile. |
| `pixi run ollama-create [run] [name]` | cuda + ollama | Export and import into Ollama (default name `steward-ste`). |
| `pixi run ollama-chat [name]` | ollama | Interactive chat with the imported model. |
| `pixi run -e ollama ollama-serve` | ollama | Start the Ollama server on port 11434. |
| `pixi run -e ollama ollama list` | ollama | List the imported models. |
| `pixi run -e ollama ollama rm <name>` | ollama | Remove an imported model. |

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `could not connect to a running Ollama instance` | The server is not running. Start `pixi run -e ollama ollama-serve` in another terminal. |
| `No completed training run in runs/` | Train a model first, or pass a run directory explicitly. |
| `llama-convert-hf-to-gguf not found` | The export ran outside the CUDA environment. Use `pixi run -e cuda export-ollama` or `pixi run ollama-create`. |
| `unsupported architecture "Qwen3ForCausalLM"` | An old Modelfile points at safetensors. Run `pixi run ollama-create` again; the current export uses GGUF. |
| Ollama output differs from `ste` output | Check that the Ollama model comes from the same run (`export.json`), and that it was exported after the last training run. |
| Ollama server on another host or port | Set `OLLAMA_HOST`, for example `OLLAMA_HOST=127.0.0.1:11500 pixi run ollama-chat`. |

## Limitations

- **No output guard in Ollama.** Ollama returns the raw model output. It does
  not check numbers, units, negations, warnings, or step order. Use
  `pixi run -e cuda ste` for checked output, and the benchmarks for metrics.
- **Backend differences.** On three test inputs, Ollama output was identical to
  the Transformers backend used by the benchmarks. That is a spot check, not a
  general guarantee.
- **Experimental model.** The current model is a small (0.6B) experimental
  student. It makes meaning errors. Example: it translates German *Ausbau der
  Pumpe* (removal of the pump) as "installing the pump". Human review is
  always necessary.
- **Fictional training data.** The training data is synthetic and fictional.
  Do not use output as approved maintenance instructions.
