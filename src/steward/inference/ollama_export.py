"""Export a trained LoRA run as a local Ollama model, for interactive testing.

The export merges the adapter into the base model, converts the merged
weights to GGUF with llama.cpp's converter (Ollama cannot import Qwen3
safetensors), and writes an Ollama ``Modelfile``. ``ollama create`` then
imports it (``pixi run ollama-create``).

The Modelfile reproduces the training and inference contract:

* the exact ``ste_only`` system prompt (fixed in the template, so it cannot
  drift from the version the model was trained on),
* the Qwen3 chat format with thinking disabled (the empty ``<think>`` block
  that ``enable_thinking=False`` adds),
* single-turn use: only the latest user message is sent, because the model
  was trained on single-turn examples. Each chat message is converted on its own,
* the default greedy sampling of :data:`steward.inference.api.DEFAULT_SAMPLING`.

Limitations: Ollama output does **not** pass through the STEward output guard.
It is an unchecked ``candidate``. Use ``pixi run -e cuda ste`` for guarded
output. Ollama's own weight conversion and runtime can produce slightly
different output from the Transformers backend that the benchmarks use.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from steward.generation.prompts import PromptTemplate
from steward.generation.providers.base import ProviderError
from steward.inference.api import DEFAULT_PROMPT, DEFAULT_SAMPLING
from steward.inference.models import ModelRef

EXPORT_VERSION = "ollama-export-0.2"
MERGED_DIR = "merged"
GGUF_FILE = "model.gguf"
GGUF_TYPES = ("bf16", "f16", "q8_0")
CONVERTER = "llama-convert-hf-to-gguf"
MODELFILE = "Modelfile"
MANIFEST = "export.json"
DEFAULT_NUM_CTX = 4096

NOTICE = (
    "Experimental STEward model export for local testing. Output is an unchecked, "
    "STE-oriented candidate: it is not checked by the STEward output guard and it is "
    "not certified or approved ASD-STE100 compliance."
)


class OllamaExportError(Exception):
    """The export cannot be completed."""


class ExportManifest(BaseModel):
    """Provenance for one export (written as ``export.json``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    export_version: str
    created_at: datetime
    base_model: str
    base_revision: str | None
    adapter_path: str | None
    run_dir: str | None
    prompt: str
    prompt_sha256: str
    gguf_type: str
    converter: str
    label: str = "candidate"
    notice: str = NOTICE


def render_modelfile(template: PromptTemplate, *, num_ctx: int = DEFAULT_NUM_CTX) -> str:
    """Return the Modelfile text for ``./model.gguf`` next to the Modelfile."""
    system = template.render(source_text="").system
    if "{{" in system or '"""' in system:
        raise OllamaExportError("The system prompt cannot be embedded in an Ollama template.")
    sampling = DEFAULT_SAMPLING
    # Go template: find the latest user message and render one Qwen3 turn,
    # byte-identical to apply_chat_template(..., enable_thinking=False).
    chat = (
        '{{- $prompt := "" }}'
        "{{- range .Messages }}"
        '{{- if eq .Role "user" }}{{ $prompt = .Content }}{{ end }}'
        "{{- end }}"
        f"<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{{ $prompt }}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    lines = [
        f"# {NOTICE}",
        f"# Prompt: {template.id}@{template.version} (sha256 {template.sha256})",
        f"FROM ./{GGUF_FILE}",
        f'TEMPLATE """{chat}"""',
        f"PARAMETER temperature {sampling.temperature}",
        f"PARAMETER top_p {sampling.top_p}",
    ]
    if sampling.seed is not None:
        lines.append(f"PARAMETER seed {sampling.seed}")
    if sampling.max_tokens is not None:
        lines.append(f"PARAMETER num_predict {sampling.max_tokens}")
    lines += [
        f"PARAMETER num_ctx {num_ctx}",
        'PARAMETER stop "<|im_end|>"',
        f'LICENSE """{NOTICE}\nBase model license: see the base model card."""',
    ]
    return "\n".join(lines) + "\n"


def merge_adapter(ref: ModelRef, out_dir: Path, *, dtype: str = "bfloat16") -> None:
    """Merge ``ref``'s LoRA adapter into its base model and save it to ``out_dir``.

    Runs on the CPU (a 0.6B model needs about 2 GB of RAM). Needs the CUDA
    environment for Torch, Transformers, and PEFT.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ProviderError(
            f"The export needs PyTorch and Transformers: {exc}.",
            hint="Run the command in the CUDA environment: pixi run -e cuda ...",
        ) from exc
    lm = AutoModelForCausalLM.from_pretrained(
        ref.model, revision=ref.revision, dtype=getattr(torch, dtype)
    )
    tokenizer_source = ref.model
    if ref.adapter_path is not None:
        from peft import PeftModel

        lm = PeftModel.from_pretrained(lm, str(ref.adapter_path)).merge_and_unload()
        tokenizer_source = str(ref.adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=None if ref.adapter_path else ref.revision
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    lm.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)


def convert_to_gguf(merged_dir: Path, outfile: Path, gguf_type: str = "bf16") -> str:
    """Convert a Hugging Face model directory to GGUF; return the converter path."""
    if gguf_type not in GGUF_TYPES:
        raise OllamaExportError(f"Unsupported GGUF type {gguf_type!r}; use one of {GGUF_TYPES}.")
    converter = shutil.which(CONVERTER)
    if converter is None:
        raise OllamaExportError(
            f"{CONVERTER} not found. Run the export in the CUDA environment: "
            "pixi run -e cuda export-ollama"
        )
    cmd = [converter, str(merged_dir), "--outfile", str(outfile), "--outtype", gguf_type]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not outfile.is_file():
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        raise OllamaExportError(f"GGUF conversion failed (exit {result.returncode}):\n{tail}")
    return converter


def export_for_ollama(
    ref: ModelRef,
    template: PromptTemplate,
    out_dir: Path,
    *,
    gguf_type: str = "bf16",
    num_ctx: int = DEFAULT_NUM_CTX,
    keep_merged: bool = False,
) -> ExportManifest:
    """Write ``model.gguf``, ``Modelfile``, and ``export.json`` to ``out_dir``.

    The merged safetensors in ``out_dir/merged`` are deleted after the
    conversion unless ``keep_merged`` is set. An earlier export in
    ``out_dir`` is replaced.
    """
    if f"{template.id}@{template.version}" != DEFAULT_PROMPT:
        raise OllamaExportError(f"Expected the {DEFAULT_PROMPT} prompt, got {template.id}.")
    modelfile = render_modelfile(template, num_ctx=num_ctx)
    merged = out_dir / MERGED_DIR
    for stale in (merged, out_dir / GGUF_FILE, out_dir / MODELFILE, out_dir / MANIFEST):
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()
    merge_adapter(ref, merged)
    converter = convert_to_gguf(merged, out_dir / GGUF_FILE, gguf_type)
    if not keep_merged:
        shutil.rmtree(merged)
    (out_dir / MODELFILE).write_text(modelfile, encoding="utf-8")
    manifest = ExportManifest(
        export_version=EXPORT_VERSION,
        created_at=datetime.now(UTC),
        base_model=ref.model,
        base_revision=ref.revision,
        adapter_path=str(ref.adapter_path) if ref.adapter_path else None,
        run_dir=str(ref.run_dir) if ref.run_dir else None,
        prompt=f"{template.id}@{template.version}",
        prompt_sha256=template.sha256,
        gguf_type=gguf_type,
        converter=f"{Path(converter).name} (llama.cpp b12004, see pixi.lock)",
    )
    (out_dir / MANIFEST).write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    return manifest
