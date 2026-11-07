"""Reproducible data-quality report (``steward dataset report``).

Reads only files that the pipeline wrote (generation manifests, accepted and rejected
JSONL files, the reviewed file, and a prepared SFT set) and counts what is in them.
No example text is copied into the report, so it can be committed.

Method labels in the report:

* deterministic: exact checks (numbers, identifiers, labels, fixture terms),
* heuristic: word-list and frequency checks (language, negation, untranslated words),
* llm_judged: the independent critic model,
* human_reviewed: none yet.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from steward.datasets.jsonl import read_jsonl
from steward.datasets.schema import DatasetExample
from steward.datasets.sft import load_prepared
from steward.ste import preservation as pv

DETERMINISTIC = {
    "candidate_nonempty",
    "numbers_preserved",
    "no_invented_measurements",
    "identifiers_preserved",
    "safety_labels_preserved",
    "terminology_forbidden",
    "required_terms_present",
    "focus_term_in_source",
    "no_contradiction",
}


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def build_report(
    generated: list[Path],
    reviewed: list[Path],
    prepared_dir: Path,
) -> dict[str, Any]:
    """Count candidates, rejections, reviews, and the prepared split."""
    gen_runs: list[dict[str, Any]] = []
    accepted = rejected = 0
    reasons: Counter[str] = Counter()
    failed_checks: Counter[str] = Counter()
    focus_gen: Counter[str] = Counter()
    focus_rej: Counter[str] = Counter()
    lang_gen: Counter[str] = Counter()
    for path in generated:
        manifest = path.with_suffix("").with_suffix(".manifest.json")
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for run in data.get("runs", []):
                gen_runs.append(
                    {
                        "file": path.as_posix(),
                        "run_id": run.get("run_id"),
                        "status": run.get("status"),
                        "scenarios": run.get("scenarios_attempted"),
                        "accepted": run.get("accepted"),
                        "rejected": run.get("rejected"),
                        "config_sha256": run.get("config_sha256"),
                        "teacher": (run.get("teacher") or {}).get("model")
                        if isinstance(run.get("teacher"), dict)
                        else None,
                    }
                )
        for ex in read_jsonl(path, DatasetExample):
            accepted += 1
            sc = ex.provenance.scenario
            focus_gen[sc.focus or "none"] += 1
            lang_gen[sc.source_language] += 1
        rej_path = path.with_suffix("").with_suffix(".rejected.jsonl")
        if rej_path.is_file():
            for line in rej_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rej = json.loads(line)
                rejected += 1
                reasons[rej.get("reason", "unknown")] += 1
                rej_sc: dict[str, Any] = rej.get("scenario") or {}
                focus_gen[rej_sc.get("focus") or "none"] += 1
                focus_rej[rej_sc.get("focus") or "none"] += 1
                lang_gen[rej_sc.get("source_language", "en")] += 1
                for check in (rej.get("validation") or {}).get("checks", []):
                    if not check.get("passed") and check.get("severity") == "error":
                        failed_checks[check["check_id"]] += 1

    review_status: Counter[str] = Counter()
    critic_first: Counter[str] = Counter()
    critic_final: Counter[str] = Counter()
    critic_models: Counter[str] = Counter()
    reviewed_total = 0
    for path in reviewed:
        for ex in read_jsonl(path, DatasetExample):
            reviewed_total += 1
            r = ex.llm_review
            if r is None:
                review_status["none"] += 1
                continue
            review_status[r.status] += 1
            critic_models[r.critic_model] += 1
            for p in r.initial_problems:
                if p.severity == "critical":
                    critic_first[p.category] += 1
            for p in r.critical:
                critic_final[p.category] += 1

    prepared = load_prepared(prepared_dir)
    m = prepared.manifest
    everything = prepared.train + prepared.validation
    labels: Counter[str] = Counter()
    for rec in everything:
        found = pv.safety_labels(rec.target, english_only=True)
        for kind in ("WARNING", "CAUTION", "NOTE"):
            if found[kind]:
                labels[kind] += 1
        if not found:
            labels["none"] += 1
    excl = m.exclusions
    dup = excl.get("duplicate", 0)
    considered = sum(v for k, v in excl.items() if ":" not in k and not k.startswith("mix_"))
    considered += sum(v for k, v in excl.items() if k.startswith("mix_downsampled"))
    return {
        "method_labels": {
            "deterministic": sorted(DETERMINISTIC),
            "heuristic": "language detection, negation counts, untranslated words (wordfreq), "
            "sentence length, STE rule checks",
            "llm_judged": dict(critic_models) or "none",
            "human_reviewed": "none",
        },
        "generation": {
            "files": [p.as_posix() for p in generated],
            "runs": gen_runs,
            "candidates": accepted + rejected,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate_pct": _pct(accepted, accepted + rejected),
            "rejection_reasons": dict(reasons.most_common()),
            "failed_error_checks": dict(failed_checks.most_common()),
            "focus_candidates": dict(sorted(focus_gen.items())),
            "focus_rejected": dict(sorted(focus_rej.items())),
            "source_languages": dict(sorted(lang_gen.items())),
        },
        "critique": {
            "files": [p.as_posix() for p in reviewed],
            "examples": reviewed_total,
            "status": dict(sorted(review_status.items())),
            "critical_findings_first_pass": dict(critic_first.most_common()),
            "critical_findings_final": dict(critic_final.most_common()),
        },
        "prepared": {
            "dir": prepared_dir.as_posix(),
            "dataset_fingerprint": m.dataset_fingerprint,
            "label": m.label,
            "train": len(prepared.train),
            "validation": len(prepared.validation),
            "validation_share_pct": _pct(len(prepared.validation), len(everything)),
            "tasks": m.counts,
            "languages": m.languages,
            "mix_target": m.mix_target,
            "mix_actual": m.mix_actual,
            "mix_deviation_pct_points": {
                t: round(100 * (m.mix_actual.get(t, 0.0) - v), 2) for t, v in m.mix_target.items()
            },
            "label_examples": dict(sorted(labels.items())),
            "distributions": m.distributions or {},
            "exclusions": excl,
            "duplicate_rate_pct": _pct(dup, considered),
        },
    }


def _table(rows: dict[str, Any], head: tuple[str, str] = ("Item", "Count")) -> str:
    lines = [f"| {head[0]} | {head[1]} |", "|---|---:|"]
    lines += [f"| `{k}` | {v} |" for k, v in rows.items()]
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    """The report as Markdown (no example text)."""
    g, c, p = report["generation"], report["critique"], report["prepared"]
    ml = report["method_labels"]
    out = [
        "# Dataset quality report",
        "",
        "Generated by `steward dataset report` from pipeline files only. Counts, not "
        "examples. All examples are synthetic and fictional, with `review_state: "
        "candidate`. Nothing here is human reviewed or ASD-STE100 compliance.",
        "",
        "Method labels: deterministic checks: "
        + ", ".join(f"`{x}`" for x in ml["deterministic"])
        + f". Heuristic checks: {ml['heuristic']}. LLM-judged: {ml['llm_judged']}. "
        f"Human-reviewed: {ml['human_reviewed']}.",
        "",
        "## Generation",
        "",
        f"* Candidates: **{g['candidates']}** (accepted {g['accepted']}, rejected "
        f"{g['rejected']}, acceptance {g['acceptance_rate_pct']}%).",
        f"* Files: {', '.join(f'`{f}`' for f in g['files'])}",
        "",
        "Rejection reasons:",
        "",
        _table(g["rejection_reasons"], ("Reason", "Candidates")),
        "",
        "Failed error-level checks in rejected candidates (a candidate can fail several):",
        "",
        _table(g["failed_error_checks"], ("Check", "Candidates")),
        "",
        "Focus (targeted failure mode) of all candidates and of rejected candidates:",
        "",
        "| Focus | Candidates | Rejected |",
        "|---|---:|---:|",
        *[
            f"| `{k}` | {v} | {g['focus_rejected'].get(k, 0)} |"
            for k, v in g["focus_candidates"].items()
        ],
        "",
        "Source languages of candidates: "
        + ", ".join(f"{k} {v}" for k, v in g["source_languages"].items())
        + ".",
        "",
        "## Independent critique (LLM-judged)",
        "",
        f"* Examples reviewed: {c['examples']}; status: "
        + ", ".join(f"{k} {v}" for k, v in c["status"].items())
        + ".",
        "",
        "Critical findings of the first critique, and findings left after revision:",
        "",
        "| Category | First critique | After revision |",
        "|---|---:|---:|",
        *[
            f"| `{k}` | {v} | {c['critical_findings_final'].get(k, 0)} |"
            for k, v in c["critical_findings_first_pass"].items()
        ],
        "",
        "## Prepared SFT set",
        "",
        f"* Directory: `{p['dir']}`; fingerprint `{p['dataset_fingerprint']}`; label "
        f"`{p['label']}`.",
        f"* Train **{p['train']}**, validation **{p['validation']}** "
        f"({p['validation_share_pct']}%; split by scenario topic).",
        f"* Duplicate rate (exact and near duplicates removed): {p['duplicate_rate_pct']}%.",
        "",
        "Task mix (in-domain): target, actual, deviation in percentage points:",
        "",
        "| Task | Target | Actual | Deviation |",
        "|---|---:|---:|---:|",
        *[
            f"| `{t}` | {v} | {p['mix_actual'].get(t)} | {p['mix_deviation_pct_points'][t]} |"
            for t, v in p["mix_target"].items()
        ],
        "",
        "Examples per split, task, and language:",
        "",
        "| Split | Tasks | Languages |",
        "|---|---|---|",
        *[
            f"| {s} | {', '.join(f'{k} {v}' for k, v in p['tasks'][s].items())} | "
            f"{', '.join(f'{k} {v}' for k, v in p['languages'][s].items())} |"
            for s in p["tasks"]
        ],
        "",
        "Safety labels in targets (examples that contain the label):",
        "",
        _table(p["label_examples"], ("Label", "Examples")),
        "",
        "Focus of selected in-domain examples: "
        + ", ".join(f"{k} {v}" for k, v in p["distributions"].get("focus", {}).items())
        + ".",
        "",
        "Exclusions during preparation:",
        "",
        _table(p["exclusions"], ("Reason", "Examples")),
        "",
    ]
    return "\n".join(out)
