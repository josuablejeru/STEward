"""Benchmark figures (SVG and PNG) generated only from stored history.

Rules:

* No data, no numbers: a figure without results shows an empty-state
  message, never placeholder values.
* Results with different comparability keys are never joined by one line.
  They get separate series and a different marker style.
* Output is deterministic (fixed SVG hash salt, no timestamps in metadata),
  so CI can check that committed figures match the history.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter

from steward.evaluation.compare import latest_by_candidate
from steward.evaluation.records import (
    BenchmarkRecord,
    CandidateRole,
    RecordStatus,
    TrainingRecord,
)

# Validated categorical palette (light mode), fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
ROLE_STYLE: dict[CandidateRole, tuple[str, str]] = {
    CandidateRole.BASE: (PALETTE[0], "o"),
    CandidateRole.PROMPTED: (PALETTE[1], "s"),
    CandidateRole.FINETUNED: (PALETTE[2], "D"),
    CandidateRole.QUANTIZED: (PALETTE[6], "v"),
    CandidateRole.RELEASE: (PALETTE[5], "*"),
}
KEY_MARKERS = ["o", "s", "^", "D", "v"]
KEY_LINESTYLES = ["-", "--", ":", "-."]

FIGURES = [
    "ste_compliance_by_model",
    "violations_over_time",
    "training_loss",
    "quality_vs_parameters",
    "quality_vs_disk_size",
    "pareto_frontier",
    "inference_performance",
    "violation_categories",
    "meaning_vs_compliance",
    "output_length_repetition",
    "baseline_finetuned_quantized",
    # STE-only multilingual suite.
    "ste_only_output_rate",
    "non_english_output_rate",
    "translation_quality_by_language",
    "technical_meaning_preservation",
    "out_of_domain_handling",
    "unnecessary_rewriting",
    "ste_violations_per_100_words",
    "quality_vs_size",
    # Milestone 3 additions.
    "validation_loss",
    "model_size",
    "model_improvement_history",
    # Milestone 4: technical fidelity.
    "safety_label_preservation",
    "negation_preservation",
    "identifier_preservation",
    "version_comparison",
    "compliance_vs_critical_errors",
    "hyperparameter_trials",
    "error_distribution",
]

_PRIMARY_KEY: str | None = None
"""Comparability key of the primary suite (set by render_all); figures prefer it."""


def _style() -> None:
    plt.rcParams.update(
        {
            "svg.hashsalt": "steward",
            "svg.fonttype": "path",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 7,
        }
    )


def _label(record: BenchmarkRecord) -> str:
    return record.candidate.release_version or record.candidate.candidate_id


def _key_short(key: str) -> str:
    suite, evaluator, rules = key.split("|")
    return f"{suite.split(':')[0]}, {evaluator}, {rules.split(':')[0]}"


def _metric(record: BenchmarkRecord, name: str) -> float | None:
    metric = record.metrics.get(name)
    return metric.value if metric else None


def _si(value: float, _pos: object = None) -> str:
    """Format large counts as 2B, 750M, 4.5G-style labels."""
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


def _gb(value: float, _pos: object = None) -> str:
    """Format bytes as gigabytes."""
    return f"{value / 1e9:g} GB"


def _nan(value: float | None) -> float:
    """Missing values become NaN (a gap in the line), never zero."""
    return float("nan") if value is None else value


def _empty(ax: Axes, message: str) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        color=TEXT_SECONDARY,
        fontsize=10,
        wrap=True,
        transform=ax.transAxes,
    )


def _footer(fig: Figure, note: str) -> None:
    # Below the figure area; bbox_inches="tight" grows the canvas to fit it.
    # Wrapped to the figure width so a long note never widens the canvas.
    width = max(60, int(fig.get_figwidth() * 17))
    text = textwrap.fill(note, width=width)
    fig.text(0.01, -0.03, text, fontsize=7, color=TEXT_SECONDARY, ha="left", va="top")


DISCLAIMER = "Automated heuristic checks. Not ASD-STE100 certification."


def _latest_groups(records: list[BenchmarkRecord]) -> dict[str, list[BenchmarkRecord]]:
    keys = sorted({r.comparability_key for r in records if r.status is RecordStatus.COMPLETED})
    groups = {}
    for key in keys:
        latest = latest_by_candidate(records, key)
        groups[key] = sorted(
            latest.values(), key=lambda r: (r.candidate.parameter_count or 0, _label(r))
        )
    return groups


def _current_group(records: list[BenchmarkRecord]) -> tuple[str | None, list[BenchmarkRecord]]:
    """The comparability group of the newest completed record."""
    completed = [r for r in records if r.status is RecordStatus.COMPLETED]
    if not completed:
        return None, []
    keys = {r.comparability_key for r in completed}
    key = (
        _PRIMARY_KEY
        if _PRIMARY_KEY in keys
        else max(completed, key=lambda r: r.created_at).comparability_key
    )
    return key, _latest_groups(records)[key]


EMPTY_BENCH = (
    "No completed benchmark results yet.\nRun `pixi run benchmark` to create real results."
)


def _hbar(
    ax: Axes,
    rows: list[BenchmarkRecord],
    metric: str,
    xlabel: str,
    color: str,
    *,
    fmt: str = "{:.2f}",
    xlim: tuple[float, float] | None = None,
    invert: bool = True,
) -> None:
    values = [_metric(r, metric) for r in rows]
    labels = [_label(r) for r in rows]
    ys = list(range(len(rows)))
    plotted = [v if v is not None else 0.0 for v in values]
    ax.barh(ys, plotted, color=color, height=0.6, edgecolor=SURFACE, linewidth=2)
    for y, r, v in zip(ys, rows, values, strict=True):
        m = r.metrics.get(metric)
        if m and m.ci_low is not None and m.ci_high is not None and v is not None:
            ax.errorbar(
                v,
                y,
                # Rounding can put a bound a hair past the point estimate; never negative.
                xerr=[[max(v - m.ci_low, 0.0)], [max(m.ci_high - v, 0.0)]],
                fmt="none",
                ecolor=TEXT_SECONDARY,
                elinewidth=1,
                capsize=3,
            )
        label_x = max(v or 0.0, m.ci_high if m and m.ci_high is not None else 0.0)
        ax.text(
            label_x,
            y,
            "  n/a" if v is None else "  " + fmt.format(v),
            va="center",
            ha="left",
            fontsize=8,
            color=TEXT,
        )
    ax.set_yticks(ys, labels)
    if invert:
        ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)


def fig_ste_compliance(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """1. STE rule-check pass rate per model (latest comparable results)."""
    key, rows = _current_group(records)
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.45 * len(rows) + 1.5)))
    ax.set_title("STE rule-check pass rate by model")
    if not rows:
        _empty(ax, EMPTY_BENCH)
    else:
        _hbar(
            ax,
            rows,
            "ste_rule_pass_rate",
            "Share of items with no error-level violation (95% Wilson CI)",
            PALETTE[0],
            xlim=(0, 1.15),
        )
        _footer(fig, f"Suite/evaluator: {_key_short(key or '')}. {DISCLAIMER}")
    return fig


def fig_violations_over_time(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """2. Violations per 100 words over time; one line per candidate and comparability key."""
    completed = sorted(
        (r for r in records if r.status is RecordStatus.COMPLETED), key=lambda r: r.created_at
    )
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.set_title("Violations per 100 words over time")
    if not completed:
        _empty(ax, EMPTY_BENCH)
        return fig
    keys = sorted({r.comparability_key for r in completed})
    candidates = sorted({_label(r) for r in completed})
    for ci, cand in enumerate(candidates[: len(PALETTE)]):
        for ki, key in enumerate(keys):
            pts = [r for r in completed if _label(r) == cand and r.comparability_key == key]
            if not pts:
                continue
            ax.plot(
                mdates.date2num([r.created_at for r in pts]),  # type: ignore[no-untyped-call]
                [_nan(_metric(r, "violations_per_100_words")) for r in pts],
                color=PALETTE[ci],
                marker=KEY_MARKERS[ki % len(KEY_MARKERS)],
                linestyle=KEY_LINESTYLES[ki % len(KEY_LINESTYLES)],
                label=cand if len(keys) == 1 else f"{cand} [{_key_short(key)}]",
            )
    ax.set_ylabel("Violations per 100 output words (lower is better)")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # type: ignore[no-untyped-call]
    fig.autofmt_xdate()
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    note = "Different suite/evaluator versions use separate line styles and are not joined."
    _footer(fig, note if len(keys) > 1 else DISCLAIMER)
    fig.tight_layout()
    return fig


def _loss_figure(trainings: list[TrainingRecord], column: str, title: str) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.set_title(title)
    runs = [t for t in trainings if t.summary.curve]
    if not runs:
        _empty(ax, "No training runs recorded yet.\nTraining is an explicit experiment step.")
        return fig
    plotted = False
    for i, run in enumerate(runs[: len(PALETTE)]):
        df = pd.DataFrame([p.model_dump() for p in run.summary.curve]).dropna(subset=[column])
        if df.empty:
            continue
        plotted = True
        hp = run.summary.hyperparameters
        label = (
            f"{run.summary.run_id} (effective batch {hp.get('effective_batch_size', '?')}, "
            f"max len {hp.get('max_length', '?')}, {hp.get('train_examples', '?')} examples)"
        )
        marker = "o" if column == "val_loss" else None
        ax.plot(df["step"], df[column], color=PALETTE[i], marker=marker, label=label)
        if column == "val_loss":
            steps, values = list(df["step"]), list(df[column])
            k = values.index(min(values))
            ax.annotate(
                f"min {values[k]:.3f} @ {int(steps[k])}",
                (steps[k], values[k]),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=7,
                color=TEXT,
            )
    if not plotted:
        _empty(ax, f"No {column.replace('_', ' ')} values were logged.")
        return fig
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Validation loss" if column == "val_loss" else "Training loss")
    ax.legend(fontsize=7, loc="upper right")
    _footer(
        fig,
        "Assistant-only (completion) cross-entropy. Runs with different datasets or "
        "validation splits are not directly comparable.",
    )
    return fig


def fig_training_loss(_: list[BenchmarkRecord], trainings: list[TrainingRecord]) -> Figure:
    """3. Training loss by optimizer step."""
    return _loss_figure(trainings, "train_loss", "Training loss by optimizer step")


def fig_validation_loss(_: list[BenchmarkRecord], trainings: list[TrainingRecord]) -> Figure:
    """Validation loss by optimizer step (minimum marked)."""
    return _loss_figure(trainings, "val_loss", "Validation loss by optimizer step")


def _pareto(points: list[tuple[float, float]]) -> list[int]:
    """Indices of points not dominated (smaller x and higher y are better)."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    frontier, best = [], float("-inf")
    for i in order:
        if points[i][1] > best:
            frontier.append(i)
            best = points[i][1]
    return frontier


def _size_scatter(
    records: list[BenchmarkRecord],
    size_metric: str,
    xlabel: str,
    title: str,
    *,
    annotate_frontier: bool,
) -> Figure:
    key, rows = _current_group(records)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.set_title(title)
    rows = [
        r for r in rows if _metric(r, size_metric) and _metric(r, "ste_rule_pass_rate") is not None
    ]
    if not rows:
        _empty(ax, EMPTY_BENCH if not key else f"No results with a known {xlabel.lower()}.")
        return fig
    pts = [
        (float(_metric(r, size_metric) or 0), float(_metric(r, "ste_rule_pass_rate") or 0))
        for r in rows
    ]
    for role in CandidateRole:
        sel = [(p, r) for p, r in zip(pts, rows, strict=True) if r.candidate.role is role]
        if sel:
            color, marker = ROLE_STYLE[role]
            ax.scatter(
                [p[0] for p, _ in sel],
                [p[1] for p, _ in sel],
                color=color,
                marker=marker,
                s=60,
                edgecolors=SURFACE,
                linewidths=2,
                label=role.value,
                zorder=3,
            )
    frontier = _pareto(pts)
    fx = [pts[i][0] for i in frontier]
    fy = [pts[i][1] for i in frontier]
    ax.step(
        fx,
        fy,
        where="post",
        color=TEXT_SECONDARY,
        linewidth=1.2,
        linestyle="--",
        label="Pareto frontier",
        zorder=2,
    )
    for i, (p, r) in enumerate(zip(pts, rows, strict=True)):
        if annotate_frontier or i in frontier:
            ax.annotate(
                _label(r), p, xytext=(5, 5), textcoords="offset points", fontsize=7, color=TEXT
            )
    ax.set_xscale("log")
    fmt = FuncFormatter(_si if size_metric == "parameter_count" else _gb)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(fmt)
    ax.set_xlabel(xlabel + " (log scale)")
    ax.set_ylabel("STE rule-check pass rate")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=7, loc="lower right")
    _footer(
        fig,
        f"Suite/evaluator: {_key_short(key or '')}. Frontier: no other model is smaller "
        "and better. " + DISCLAIMER,
    )
    return fig


def fig_quality_vs_parameters(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """4. Quality versus parameter count."""
    return _size_scatter(
        records,
        "parameter_count",
        "Parameters",
        "Quality vs. parameter count",
        annotate_frontier=True,
    )


def fig_quality_vs_disk(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """5. Quality versus model size on disk."""
    return _size_scatter(
        records,
        "disk_size_bytes",
        "Size on disk",
        "Quality vs. size on disk",
        annotate_frontier=True,
    )


def fig_pareto(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """Pareto frontier: best quality for model size (frontier models labeled)."""
    return _size_scatter(
        records,
        "parameter_count",
        "Parameters",
        "Pareto frontier: quality vs. size",
        annotate_frontier=False,
    )


def fig_inference(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """6. Inference performance: speed and latency (two charts, no dual axis)."""
    key, rows = _current_group(records)
    fig, axes = plt.subplots(1, 2, figsize=(10, max(2.8, 0.45 * len(rows) + 1.6)), sharey=True)
    fig.suptitle("Inference performance", x=0.01, ha="left", fontweight="bold", color=TEXT)
    if not rows:
        for ax in axes:
            _empty(ax, EMPTY_BENCH)
        return fig
    _hbar(
        axes[0],
        rows,
        "tokens_per_s",
        "Generation speed (tokens/s, higher is better)",
        PALETTE[0],
        fmt="{:.0f}",
    )
    _hbar(
        axes[1],
        rows,
        "latency_mean_s",
        "Mean latency per item (s, lower is better)",
        PALETTE[1],
        invert=False,
    )
    axes[1].tick_params(labelleft=False)
    _footer(fig, f"Suite/evaluator: {_key_short(key or '')}. Hardware is in each history record.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_violation_categories(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """7. Violation counts per rule ID and model (small multiples would hide totals)."""
    key, rows = _current_group(records)
    rules = sorted({rule for r in rows for rule in r.rule_counts})
    fig, ax = plt.subplots(figsize=(8, max(2.8, 0.5 * len(rows) + 1.8)))
    ax.set_title("Violations by rule")
    if not rows:
        _empty(ax, EMPTY_BENCH)
        return fig
    if not rules:
        _empty(ax, "No rule violations in the latest results.")
        return fig
    ys = list(range(len(rows)))
    left = [0.0] * len(rows)
    for i, rule in enumerate(rules[: len(PALETTE)]):
        counts = [float(r.rule_counts.get(rule, 0)) for r in rows]
        ax.barh(
            ys,
            counts,
            left=left,
            color=PALETTE[i],
            height=0.6,
            edgecolor=SURFACE,
            linewidth=2,
            label=rule,
        )
        left = [a + b for a, b in zip(left, counts, strict=True)]
    for y, total in zip(ys, left, strict=True):
        ax.text(total, y, f"  {total:.0f}", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(ys, [_label(r) for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Violations in all suite outputs")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _footer(
        fig, f"Suite/evaluator: {_key_short(key or '')}. Heuristic rules, see docs/benchmarks.md."
    )
    fig.tight_layout()
    return fig


def fig_meaning_vs_compliance(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """8. Meaning preservation versus STE compliance."""
    rows = _current_group(records)[1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Meaning preservation vs. STE compliance")
    rows = [r for r in rows if _metric(r, "meaning_preservation") is not None]
    if not rows:
        _empty(ax, EMPTY_BENCH)
        return fig
    for role in CandidateRole:
        sel = [r for r in rows if r.candidate.role is role]
        if sel:
            color, marker = ROLE_STYLE[role]
            ax.scatter(
                [_metric(r, "ste_rule_pass_rate") or 0.0 for r in sel],
                [_metric(r, "meaning_preservation") or 0.0 for r in sel],
                color=color,
                marker=marker,
                s=60,
                edgecolors=SURFACE,
                linewidths=2,
                label=role.value,
                zorder=3,
            )
    for r in rows:
        ax.annotate(
            _label(r),
            (_metric(r, "ste_rule_pass_rate") or 0, _metric(r, "meaning_preservation") or 0),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
            color=TEXT,
        )
    ax.set_xlabel("STE rule-check pass rate")
    lowest = min(_metric(r, "meaning_preservation") or 0.0 for r in rows)
    bottom = max(0.0, round(lowest - 0.05, 2))
    ax.set_ylabel(
        "Meaning preservation (fact proxy)" + (f", axis from {bottom:g}" if bottom else "")
    )
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(bottom, 1.01)
    ax.legend(fontsize=7, loc="lower left")
    _footer(fig, "Top right is best. " + DISCLAIMER)
    return fig


def fig_length_repetition(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """9. Output length and repetition distributions (two charts)."""
    key, rows = _current_group(records)
    fig, axes = plt.subplots(1, 2, figsize=(10, max(2.8, 0.45 * len(rows) + 1.6)), sharey=True)
    fig.suptitle("Output length and repetition", x=0.01, ha="left", fontweight="bold", color=TEXT)
    if not rows:
        for ax in axes:
            _empty(ax, EMPTY_BENCH)
        return fig
    labels = [_label(r) for r in rows]
    lengths = [[i.words for i in r.items if i.valid_output] or [0] for r in rows]
    reps = [[i.repetition_rate for i in r.items if i.valid_output] or [0.0] for r in rows]
    for ax, data, xlabel, color in (
        (axes[0], lengths, "Output words per item", PALETTE[0]),
        (axes[1], reps, "Repeated trigram rate per item", PALETTE[1]),
    ):
        ax.boxplot(
            data,
            orientation="horizontal",
            tick_labels=labels,
            widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": color, "edgecolor": TEXT_SECONDARY},
            medianprops={"color": TEXT},
            whiskerprops={"color": TEXT_SECONDARY},
            capprops={"color": TEXT_SECONDARY},
            flierprops={"markeredgecolor": TEXT_SECONDARY},
        )
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    _footer(fig, f"Suite/evaluator: {_key_short(key or '')}. Valid outputs only.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_roles(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """10. Baseline vs prompted vs fine-tuned vs quantized, per base model."""
    rows = _current_group(records)[1]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.set_title("Baseline vs. fine-tuned vs. quantized")
    if not rows:
        _empty(ax, EMPTY_BENCH)
        return fig
    bases = sorted({r.candidate.base_model for r in rows})
    roles = [role for role in CandidateRole if any(r.candidate.role is role for r in rows)]
    width = 0.8 / max(len(roles), 1)
    for ri, role in enumerate(roles):
        xs, ys = [], []
        for bi, base in enumerate(bases):
            match = [r for r in rows if r.candidate.base_model == base and r.candidate.role is role]
            if match:
                xs.append(bi + (ri - (len(roles) - 1) / 2) * width)
                ys.append(_metric(match[-1], "ste_rule_pass_rate") or 0.0)
        color = ROLE_STYLE[role][0]
        ax.bar(xs, ys, width=width, color=color, edgecolor=SURFACE, linewidth=2, label=role.value)
        for x, y in zip(xs, ys, strict=True):
            ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=7, color=TEXT)
    ax.set_xticks(range(len(bases)), bases, fontsize=8)
    ax.set_ylabel("STE rule-check pass rate")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, ncols=len(roles), loc="upper left")
    missing = {CandidateRole.FINETUNED, CandidateRole.QUANTIZED} - set(roles)
    note = (
        f"No {' or '.join(sorted(m.value for m in missing))} results yet. " if missing else ""
    ) + DISCLAIMER
    _footer(fig, note)
    return fig


# -- STE-only suite figures -------------------------------------------------------

STE_DISCLAIMER = (
    "Method: automated heuristic checks on a draft suite; no human-reviewed results yet. "
    "Not ASD-STE100 certification."
)
EMPTY_STE = (
    "No STE-only benchmark results yet.\nRun `pixi run -e cuda benchmark-gpu` to create real "
    "results."
)


def _ste_group(records: list[BenchmarkRecord]) -> tuple[str | None, list[BenchmarkRecord]]:
    """Latest results of the newest comparability group with STE-only metrics."""
    ste = [
        r
        for r in records
        if r.status is RecordStatus.COMPLETED and "ste_only_output_rate" in r.metrics
    ]
    if not ste:
        return None, []
    keys = {r.comparability_key for r in ste}
    key = (
        _PRIMARY_KEY
        if _PRIMARY_KEY in keys
        else max(ste, key=lambda r: r.created_at).comparability_key
    )
    return key, _latest_groups(records)[key]


def _role_color(record: BenchmarkRecord) -> str:
    return ROLE_STYLE.get(record.candidate.role, (PALETTE[7], "o"))[0]


def _ste_hbar_figure(
    records: list[BenchmarkRecord],
    panels: list[tuple[str, str, str, tuple[float, float] | None]],
    title: str,
    note: str,
) -> Figure:
    key, rows = _ste_group(records)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(4.2 * len(panels) + 1.6, 1.0 + 0.45 * max(len(rows), 3)),
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=TEXT)
    if not rows:
        for ax in axes[0]:
            ax.axis("off")
        _empty(axes[0][0], EMPTY_STE)
        _footer(fig, STE_DISCLAIMER)
        return fig
    for i, (ax, (metric, xlabel, fmt, xlim)) in enumerate(zip(axes[0], panels, strict=True)):
        _hbar(ax, rows, metric, xlabel, PALETTE[0], fmt=fmt, xlim=xlim, invert=i == 0)
        for patch, r in zip(ax.patches, rows, strict=False):
            patch.set_facecolor(_role_color(r))
    handles = [
        Rectangle((0, 0), 1, 1, color=c)
        for role, (c, _) in ROLE_STYLE.items()
        if any(r.candidate.role is role for r in rows)
    ]
    labels = [role.value for role in ROLE_STYLE if any(r.candidate.role is role for r in rows)]
    fig.legend(handles, labels, loc="upper right", ncols=len(labels), fontsize=7, frameon=False)
    fig.tight_layout()
    _footer(fig, f"{note} Suite/evaluator: {_key_short(key or '')}. {STE_DISCLAIMER}")
    return fig


def fig_ste_only_rate(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """STE-only output rate (English text without commentary) per model version."""
    return _ste_hbar_figure(
        records,
        [("ste_only_output_rate", "STE-only output rate (in-domain items)", "{:.2f}", (0, 1.18))],
        "STE-only output rate by model version",
        "STE-only = non-empty English text, no commentary, Markdown, or reasoning; 95% Wilson CI.",
    )


def fig_non_english(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """Non-English output rate and commentary rate."""
    return _ste_hbar_figure(
        records,
        [
            ("non_english_output_rate", "Non-English output rate", "{:.2f}", (0, 1.18)),
            ("extra_commentary_rate", "Commentary / Markdown rate", "{:.2f}", (0, 1.18)),
        ],
        "Non-English output and extra commentary (lower is better)",
        "Language identification uses stopword and diacritic counts (heuristic).",
    )


def fig_translation_by_language(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """Meaning preservation and chrF per source language (small multiples)."""
    key, rows = _ste_group(records)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle(
        "Multilingual quality by source language",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=TEXT,
    )
    if not rows:
        axes[1].axis("off")
        _empty(axes[0], EMPTY_STE)
        _footer(fig, STE_DISCLAIMER)
        return fig
    langs = ["en", "de", "fr", "ro"]
    rows = rows[: len(PALETTE)]
    width = 0.8 / len(rows)
    for ax, metric, ylabel, ylim in (
        (axes[0], "meaning_preservation", "Meaning preservation (fact proxy)", (0, 1.15)),
        (axes[1], "chrf_mean", "chrF2 vs. draft reference", (0, 105)),
    ):
        for i, r in enumerate(rows):
            xs = [j + (i - (len(rows) - 1) / 2) * width for j in range(len(langs))]
            vals = []
            for lang in langs:
                m = r.by_category.get(f"lang:{lang}", {}).get(metric)
                vals.append(_nan(m.value if m else None))
            ax.bar(
                xs,
                vals,
                width=width,
                color=PALETTE[i],
                edgecolor=SURFACE,
                linewidth=1.5,
                label=_label(r),
            )
        ax.set_xticks(range(len(langs)), ["English", "German", "French", "Romanian"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=min(len(labels), 3),
        fontsize=7,
        bbox_to_anchor=(0.5, 0.0),
    )
    _footer(
        fig,
        "English = English rewrite, requirements, and identity items; other languages = "
        f"translation and requirements items. Suite/evaluator: {_key_short(key or '')}. "
        + STE_DISCLAIMER,
    )
    return fig


def fig_meaning(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """Technical meaning preservation and critical safety failures."""
    return _ste_hbar_figure(
        records,
        [
            ("meaning_preservation", "Meaning preservation (fact proxy)", "{:.2f}", (0, 1.18)),
            ("critical_failures", "Critical safety failures (items)", "{:.0f}", None),
        ],
        "Technical meaning preservation",
        "Facts: numbers, measurements, identifiers, WARNING/CAUTION, required terms, negation; "
        "95% bootstrap CI.",
    )


def fig_ood(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """Controlled failure on out-of-domain input, and false refusals on valid input."""
    return _ste_hbar_figure(
        records,
        [
            (
                "controlled_failure_reliability",
                "Out-of-domain: controlled failure",
                "{:.2f}",
                (0, 1.18),
            ),
            ("false_refusal_rate", "In-domain: refused (lower is better)", "{:.2f}", (0, 1.18)),
        ],
        "Out-of-domain handling",
        "Controlled failure = the model returns STEWARD_UNSUPPORTED_INPUT; 95% Wilson CI.",
    )


def fig_identity(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """Unnecessary rewriting of already-valid STE input."""
    return _ste_hbar_figure(
        records,
        [
            (
                "unnecessary_rewrite_rate",
                "Valid STE inputs rewritten (lower is better)",
                "{:.2f}",
                (0, 1.18),
            ),
            ("identity_edit_similarity", "Word similarity to the input", "{:.2f}", (0, 1.18)),
        ],
        "Unnecessary rewriting of already-valid STE",
        "Identity items: the correct output is the unchanged input.",
    )


def fig_ste_violations(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """STE rule violations per 100 words on the STE-only suite."""
    return _ste_hbar_figure(
        records,
        [
            (
                "violations_per_100_words",
                "Rule violations per 100 words (lower is better)",
                "{:.1f}",
                None,
            ),
            ("ste_rule_pass_rate", "Items with no rule error", "{:.2f}", (0, 1.18)),
        ],
        "STE rule violations per 100 words",
        "Heuristic rules ste-heuristic-0.1 (sentence length, -ing, contractions, passive).",
    )


def fig_quality_vs_size(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """Model quality versus parameter count on the STE-only suite."""
    key, rows = _ste_group(records)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle(
        "Model quality versus model size",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=TEXT,
    )
    sized = [r for r in rows if r.candidate.parameter_count]
    if not sized:
        axes[1].axis("off")
        _empty(axes[0], EMPTY_STE if not rows else "No results with a known parameter count.")
        _footer(fig, STE_DISCLAIMER)
        return fig
    for ax, metric, ylabel in (
        (axes[0], "guard_pass_rate", "Output guard pass rate (no critical failure)"),
        (axes[1], "ste_rule_pass_rate", "STE rule-check pass rate"),
    ):
        for r in sized:
            color, marker = ROLE_STYLE.get(r.candidate.role, (PALETTE[7], "o"))
            y = _metric(r, metric)
            if y is None:
                continue
            ax.scatter(
                r.candidate.parameter_count,
                y,
                color=color,
                marker=marker,
                s=60,
                edgecolors=SURFACE,
                linewidths=2,
                zorder=3,
            )
            ax.annotate(
                _label(r),
                (r.candidate.parameter_count, y),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color=TEXT,
            )
        ax.set_xscale("log")
        ax.set_xticks([3e8, 1e9, 3e9, 1e10])
        ax.xaxis.set_major_formatter(FuncFormatter(_si))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(3e8, 2e10)
        ax.set_xlabel("Parameters (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.1)
    roles = [role for role in ROLE_STYLE if any(r.candidate.role is role for r in sized)]
    handles = [
        Line2D([], [], color=ROLE_STYLE[ro][0], marker=ROLE_STYLE[ro][1], ls="") for ro in roles
    ]
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(
        handles,
        [ro.value for ro in roles],
        fontsize=7,
        loc="lower center",
        ncols=len(roles),
        bbox_to_anchor=(0.5, 0.0),
    )
    _footer(fig, f"Suite/evaluator: {_key_short(key or '')}. {STE_DISCLAIMER}")
    return fig


def fig_model_size(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """Parameters and size on disk per model (two charts, no dual axis)."""
    key, rows = _ste_group(records)
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 1.2 + 0.45 * max(len(rows), 3)), sharey=True, squeeze=False
    )
    fig.suptitle("Model size", x=0.01, ha="left", fontsize=11, fontweight="bold", color=TEXT)
    if not rows:
        for ax in axes[0]:
            ax.axis("off")
        _empty(axes[0][0], EMPTY_STE)
        return fig
    ys = list(range(len(rows)))
    labels = [_label(r) for r in rows]
    panels = [
        ("parameter_count", "Parameters", _si),
        (
            "disk_size_bytes",
            "Size on disk (base + adapter)",
            lambda v, _p=None: f"{v / 1e9:.2f} GB",
        ),
    ]
    for ax, (metric, xlabel, formatter) in zip(axes[0], panels, strict=True):
        values = [_metric(r, metric) for r in rows]
        ax.barh(
            ys,
            [v if v else float("nan") for v in values],
            color=[_role_color(r) for r in rows],
            height=0.6,
            edgecolor=SURFACE,
            linewidth=2,
        )
        for y, r, v in zip(ys, rows, values, strict=True):
            text = "n/a" if v is None else formatter(v)
            adapter = _metric(r, "adapter_size_bytes")
            if metric == "disk_size_bytes" and adapter:
                text += f" (adapter {adapter / 1e6:.0f} MB)"
            ax.text(v or 0.0, y, "  " + text, va="center", fontsize=8, color=TEXT)
        if not any(values):
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_xlabel(f"{xlabel}: not reported")
            ax.grid(axis="y", visible=False)
            continue
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(formatter))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(f"{xlabel} (log scale)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(right=max((v or 1.0) for v in values) * 6)
    axes[0][0].set_yticks(ys, labels)
    axes[0][0].invert_yaxis()
    fig.tight_layout()
    _footer(
        fig,
        f"Ollama sizes are quantized GGUF files; Transformers sizes are BF16 safetensors. "
        f"Suite/evaluator: {_key_short(key or '')}.",
    )
    return fig


def fig_improvement_history(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """STE pass rate and critical failures over time for STE-only results."""
    ste = sorted(
        (
            r
            for r in records
            if r.status is RecordStatus.COMPLETED and "ste_only_output_rate" in r.metrics
        ),
        key=lambda r: r.created_at,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), squeeze=False)
    fig.suptitle(
        "Model improvement history (STE-only suites)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=TEXT,
    )
    if not ste:
        for ax in axes[0]:
            ax.axis("off")
        _empty(axes[0][0], EMPTY_STE)
        return fig
    keys = sorted({r.comparability_key for r in ste})
    candidates = sorted({_label(r) for r in ste})
    panels = [
        ("ste_rule_pass_rate", "STE rule-check pass rate (higher is better)"),
        ("critical_failures", "Critical safety failures (lower is better)"),
    ]
    for ax, (metric, ylabel) in zip(axes[0], panels, strict=True):
        for ci, cand in enumerate(candidates[: len(PALETTE)]):
            labelled = False
            for ki, key in enumerate(keys):
                pts = [r for r in ste if _label(r) == cand and r.comparability_key == key]
                if not pts:
                    continue
                first, labelled = not labelled, True
                ax.plot(
                    mdates.date2num([r.created_at for r in pts]),  # type: ignore[no-untyped-call]
                    [_nan(_metric(r, metric)) for r in pts],
                    color=PALETTE[ci],
                    marker=KEY_MARKERS[ki % len(KEY_MARKERS)],
                    linestyle=KEY_LINESTYLES[ki % len(KEY_LINESTYLES)],
                    label=cand if first else None,
                )
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))  # type: ignore[no-untyped-call]
        ax.tick_params(axis="x", labelrotation=30)
    axes[0][1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    _footer(
        fig,
        "Marker and line style show the suite/evaluator version; points with different "
        "versions are not comparable and are not joined. " + STE_DISCLAIMER,
    )
    return fig


def fig_label_preservation(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """WARNING/CAUTION/NOTE labels kept exactly."""
    return _ste_hbar_figure(
        records,
        [
            (
                "safety_label_preservation",
                "Items with labels: all kept, none changed",
                "{:.2f}",
                (0, 1.18),
            )
        ],
        "Safety-label preservation (higher is better)",
        "Deterministic label count per kind (WARNING, CAUTION, NOTE) in source and output.",
    )


def fig_negation_preservation(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """Negations kept (count-based heuristic)."""
    return _ste_hbar_figure(
        records,
        [("negation_preservation", "Items with negation: negation kept", "{:.2f}", (0, 1.18))],
        "Negation preservation (higher is better)",
        "Heuristic: negation words counted per language; it does not parse scope.",
    )


def fig_identifier_preservation(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """Identifier codes and fixture terms."""
    return _ste_hbar_figure(
        records,
        [
            ("terminology_preservation", "Identifier codes kept unchanged", "{:.2f}", (0, 1.18)),
            (
                "terminology_accuracy",
                "Fixture terms: no known wrong rendering",
                "{:.2f}",
                (0, 1.18),
            ),
        ],
        "Identifier and terminology preservation (higher is better)",
        "Deterministic string checks; fixture terms are in configs/ste/terminology.toml.",
    )


_VERSION_METRICS = [
    ("technical_error_rate", "Technical errors", False),
    ("meaning_preservation", "Meaning kept", True),
    ("negation_preservation", "Negation kept", True),
    ("safety_label_preservation", "Labels kept", True),
    ("unsupported_addition_rate", "Added content", False),
    ("ste_only_output_rate", "STE-only output", True),
    ("ste_rule_pass_rate", "STE rule pass", True),
]


def fig_version_comparison(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """Untuned 0.6B vs fine-tuned versions on the primary suite, primary metrics first."""
    key, rows = _ste_group(records)
    rows = [r for r in rows if r.candidate.base_model == "Qwen/Qwen3-0.6B"]
    rows.sort(key=lambda r: (r.candidate.role is not CandidateRole.BASE, r.created_at))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_title("Qwen3-0.6B: untuned vs. fine-tuned versions")
    if not rows:
        _empty(ax, EMPTY_STE)
        return fig
    n = len(rows)
    width = 0.8 / n
    for i, r in enumerate(rows[: len(PALETTE)]):
        xs = [k + (i - (n - 1) / 2) * width for k in range(len(_VERSION_METRICS))]
        ys = [_metric(r, m) or 0.0 for m, _, _ in _VERSION_METRICS]
        ax.bar(
            xs, ys, width=width, color=PALETTE[i], edgecolor=SURFACE, linewidth=2, label=_label(r)
        )
        for x, y in zip(xs, ys, strict=True):
            ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=6, color=TEXT)
    ax.set_xticks(
        range(len(_VERSION_METRICS)),
        [
            f"{label}\n({'higher' if up else 'lower'} is better)"
            for _, label, up in _VERSION_METRICS
        ],
        fontsize=7,
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    _footer(
        fig,
        "Safety and technical fidelity (left) take priority over STE style (right). "
        f"Suite/evaluator: {_key_short(key or '')}. {STE_DISCLAIMER}",
    )
    return fig


def fig_compliance_vs_critical(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """Trade-off: STE rule-check pass rate vs. critical technical errors."""
    key, rows = _ste_group(records)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("STE compliance vs. critical technical errors")
    rows = [r for r in rows if _metric(r, "ste_rule_pass_rate") is not None]
    if not rows:
        _empty(ax, EMPTY_STE)
        return fig
    for r in rows:
        x = _metric(r, "ste_rule_pass_rate") or 0.0
        y = _metric(r, "critical_technical_errors")
        if y is None:
            y = _metric(r, "critical_failures") or 0.0
        color, marker = ROLE_STYLE.get(r.candidate.role, (PALETTE[7], "o"))
        ax.scatter(
            [x], [y], color=color, marker=marker, s=80, edgecolor=SURFACE, linewidth=2, zorder=3
        )
        ax.annotate(
            _label(r), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7, color=TEXT
        )
    ax.set_xlabel("STE rule-check pass rate (higher is better)")
    ax.set_ylabel("Items with a critical technical error (lower is better)")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(bottom=0)
    ax.text(1.05, 0.2, "best corner", ha="right", fontsize=7, color=TEXT_SECONDARY)
    handles = [
        Line2D([], [], color=c, marker=m, linestyle="", label=role.value)
        for role, (c, m) in ROLE_STYLE.items()
        if any(r.candidate.role is role for r in rows)
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    _footer(
        fig,
        "A model that passes more style checks but has more critical errors is not better. "
        f"Suite/evaluator: {_key_short(key or '')}. {STE_DISCLAIMER}",
    )
    return fig


def fig_hyperparameter_trials(
    records: list[BenchmarkRecord], trainings: list[TrainingRecord]
) -> Figure:
    """Sweep trials: best validation loss and development-suite results."""
    runs = [
        t
        for t in trainings
        if isinstance(t.summary.hyperparameters.get("version_label"), str)
        and t.summary.hyperparameters.get("best_val_loss") is not None
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 1.4 + 0.5 * max(len(runs), 3)), sharey=True, squeeze=False
    )
    fig.suptitle(
        "Hyperparameter trials", x=0.01, ha="left", fontsize=11, fontweight="bold", color=TEXT
    )
    if not runs:
        for ax in axes[0]:
            ax.axis("off")
        _empty(axes[0][0], "No sweep trials recorded yet.")
        return fig
    runs.sort(key=lambda t: str(t.summary.hyperparameters.get("version_label")))
    dev: dict[str, BenchmarkRecord] = {}
    for r in sorted(records, key=lambda r: r.created_at):
        if r.status is RecordStatus.COMPLETED and r.suite.suite_id == "ste-fidelity-dev":
            dev[str(r.candidate.training_run_id)] = r
    labels = []
    for t in runs:
        hp = t.summary.hyperparameters
        labels.append(
            f"{hp.get('version_label')}\nlr {hp.get('learning_rate')}, r {hp.get('lora_r')}, "
            f"{hp.get('epochs')} ep"
        )
    ys = list(range(len(runs)))
    panels = [
        ("best_val_loss", "Best validation loss (lower is better)", "{:.3f}"),
        ("critical_technical_errors", "Dev suite: items with technical errors", "{:.0f}"),
        ("meaning_preservation", "Dev suite: meaning kept", "{:.2f}"),
    ]
    for ax, (metric, xlabel, fmt) in zip(axes[0], panels, strict=True):
        values: list[float | None] = []
        for t in runs:
            if metric == "best_val_loss":
                v = t.summary.hyperparameters.get("best_val_loss")
                values.append(float(v) if isinstance(v, (int, float)) else None)
            else:
                rec = dev.get(t.summary.run_id)
                values.append(_metric(rec, metric) if rec else None)
        ax.barh(
            ys,
            [v if v is not None else float("nan") for v in values],
            color=PALETTE[2],
            height=0.6,
            edgecolor=SURFACE,
            linewidth=2,
        )
        for y, v in zip(ys, values, strict=True):
            ax.text(
                v or 0.0,
                y,
                "  n/a" if v is None else "  " + fmt.format(v),
                va="center",
                fontsize=8,
                color=TEXT,
            )
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        finite = [v for v in values if v is not None]
        if finite:
            ax.set_xlim(0, max(finite) * 1.35 or 1)
    axes[0][0].set_yticks(ys, labels, fontsize=7)
    axes[0][0].invert_yaxis()
    fig.tight_layout()
    _footer(
        fig,
        "Selection uses the development suite (ste-fidelity-dev) and validation loss, never "
        "training loss and never the final golden suite. " + STE_DISCLAIMER,
    )
    return fig


_ERROR_CHECKS = [
    "safety_labels_preserved",
    "negation_preserved",
    "identifiers_preserved",
    "numbers_preserved",
    "measurements_preserved",
    "no_unsupported_additions",
    "terminology_forbidden",
    "no_source_language_words",
    "sequence_preserved",
    "no_contradiction",
    "english_output",
    "length_plausible",
    "output_contract",
]


def fig_error_distribution(records: list[BenchmarkRecord], _: list[TrainingRecord]) -> Figure:
    """Critical guard failures by category and model (in-domain items)."""
    key, rows = _ste_group(records)
    rows = rows[: len(PALETTE)]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_title("Critical errors by category (output guard, in-domain items)")
    if not rows:
        _empty(ax, EMPTY_STE)
        return fig
    counts = [
        [
            sum(1 for i in r.items if not i.expected_unsupported and c in (i.guard_critical or []))
            for c in _ERROR_CHECKS
        ]
        for r in rows
    ]
    ys = list(range(len(_ERROR_CHECKS)))
    height = 0.8 / len(rows)
    for i, (r, row) in enumerate(zip(rows, counts, strict=True)):
        offs = [y + (i - (len(rows) - 1) / 2) * height for y in ys]
        ax.barh(
            offs,
            row,
            height=height,
            color=PALETTE[i],
            edgecolor=SURFACE,
            linewidth=1,
            label=_label(r),
        )
    ax.set_yticks(ys, _ERROR_CHECKS, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Items (a single item can fail several checks)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    _footer(fig, f"Suite/evaluator: {_key_short(key or '')}. {STE_DISCLAIMER}")
    return fig


BUILDERS: dict[str, Callable[[list[BenchmarkRecord], list[TrainingRecord]], Figure]] = {
    "ste_compliance_by_model": fig_ste_compliance,
    "violations_over_time": fig_violations_over_time,
    "training_loss": fig_training_loss,
    "quality_vs_parameters": fig_quality_vs_parameters,
    "quality_vs_disk_size": fig_quality_vs_disk,
    "pareto_frontier": fig_pareto,
    "inference_performance": fig_inference,
    "violation_categories": fig_violation_categories,
    "meaning_vs_compliance": fig_meaning_vs_compliance,
    "output_length_repetition": fig_length_repetition,
    "baseline_finetuned_quantized": fig_roles,
    "ste_only_output_rate": fig_ste_only_rate,
    "non_english_output_rate": fig_non_english,
    "translation_quality_by_language": fig_translation_by_language,
    "technical_meaning_preservation": fig_meaning,
    "out_of_domain_handling": fig_ood,
    "unnecessary_rewriting": fig_identity,
    "ste_violations_per_100_words": fig_ste_violations,
    "quality_vs_size": fig_quality_vs_size,
    "validation_loss": fig_validation_loss,
    "model_size": fig_model_size,
    "model_improvement_history": fig_improvement_history,
    "safety_label_preservation": fig_label_preservation,
    "negation_preservation": fig_negation_preservation,
    "identifier_preservation": fig_identifier_preservation,
    "version_comparison": fig_version_comparison,
    "compliance_vs_critical_errors": fig_compliance_vs_critical,
    "hyperparameter_trials": fig_hyperparameter_trials,
    "error_distribution": fig_error_distribution,
}


def render_all(
    benchmarks: list[BenchmarkRecord],
    trainings: list[TrainingRecord],
    out_dir: Path,
    formats: tuple[str, ...] = ("svg", "png"),
    names: tuple[str, ...] | None = None,
    primary_key: str | None = None,
) -> list[Path]:
    """Render figures (all by default) into ``out_dir``. Returns the written paths.

    ``primary_key``: comparability key of the primary suite; model-comparison figures
    use it instead of the newest group (development-suite runs are newer).
    """
    global _PRIMARY_KEY
    _PRIMARY_KEY = primary_key
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names or FIGURES:
        fig = BUILDERS[name](benchmarks, trainings)
        for fmt in formats:
            path = out_dir / f"{name}.{fmt}"
            metadata = {"Date": None} if fmt == "svg" else {"Software": None}
            fig.savefig(path, format=fmt, dpi=150, metadata=metadata, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
    return written
