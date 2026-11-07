"""``steward benchmark ...`` subcommands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_FAILURE = 1


def find_root(start: Path | None = None) -> Path:
    """Return the nearest directory with ``pixi.toml`` (the project root)."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pixi.toml").exists():
            return candidate
    return here


def add_benchmark_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``benchmark`` and its subcommands."""
    bench = subparsers.add_parser(
        "benchmark",
        help="Run, track, compare, and report benchmarks.",
        description="Benchmark tracking. Results are automated heuristic checks, "
        "not certification.",
    )
    bench.add_argument("--root", type=Path, default=None, help="Project root (default: auto).")
    bench.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    bench.add_argument("--log-format", default="text", choices=["text", "json"])
    sub = bench.add_subparsers(dest="benchmark_command", metavar="<action>", required=True)
    # The log options also work after the action (e.g. `pixi run benchmark --log-level DEBUG`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--log-level", default=argparse.SUPPRESS, choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    common.add_argument("--log-format", default=argparse.SUPPRESS, choices=["text", "json"])

    run = sub.add_parser(
        parents=[common], name="run", help="Benchmark registered candidates on the golden suite."
    )
    run.add_argument("--candidate", action="append", default=[], help="Candidate ID (repeatable).")
    run.add_argument("--force", action="store_true", help="Run even when a current result exists.")
    run.add_argument("--require-gpu", action="store_true", help="Fail when no GPU is visible.")
    run.add_argument("--base-url", default=None, help="Provider URL (default: $OLLAMA_HOST).")
    run.add_argument("--no-regenerate", action="store_true", help="Do not update figures/report.")
    run.set_defaults(handler=_cmd_run)

    hist = sub.add_parser(parents=[common], name="history", help="List the experiment history.")
    hist.add_argument("--json", action="store_true")
    hist.add_argument("--type", choices=["benchmark", "training"], default=None)
    hist.set_defaults(handler=_cmd_history)

    comp = sub.add_parser(
        parents=[common], name="compare", help="Compare results and detect regressions."
    )
    comp.add_argument("candidate", nargs="?", help="Candidate or experiment ID (default: all).")
    comp.add_argument("--against", default=None, help="Reference candidate or experiment ID.")
    comp.set_defaults(handler=_cmd_compare)

    plot = sub.add_parser(
        parents=[common], name="plot", help="Regenerate SVG/PNG figures from the history."
    )
    plot.set_defaults(handler=_cmd_plot)

    report = sub.add_parser(
        parents=[common], name="report", help="Regenerate figures, docs/benchmarks.md, and README."
    )
    report.add_argument("--no-figures", action="store_true")
    report.set_defaults(handler=_cmd_report)

    status = sub.add_parser(
        parents=[common], name="status", help="Show pending work, stale results, and hook state."
    )
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=_cmd_status)

    ver = sub.add_parser(
        parents=[common],
        name="verify",
        help="Check history, suite, data, links, and generated files.",
    )
    ver.add_argument(
        "--check-generated",
        action="store_true",
        help="Also check that figures and reports are up to date.",
    )
    ver.add_argument("--no-datasets", action="store_true", help="Skip local data/ checks.")
    ver.set_defaults(handler=_cmd_verify)


def add_evaluate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``evaluate``: benchmark one trained model (or registered candidate)."""
    ev = subparsers.add_parser(
        "evaluate",
        help="Benchmark a trained model and append the result to the history.",
        description=(
            "Benchmark a training run (runs/<id> or runs/<id>/adapter) or a registered "
            "candidate ID on its STE suite. Appends to benchmarks/results/history.jsonl and "
            "updates figures and reports. Results are automated heuristic checks, not "
            "certification. Run in the CUDA environment: pixi run -e cuda steward evaluate ..."
        ),
    )
    ev.add_argument(
        "--model", required=True, help="runs/<id>, runs/<id>/adapter, or a candidate ID."
    )
    ev.add_argument("--suite", default=None, help="Suite directory name (default: the run's).")
    ev.add_argument(
        "--baseline",
        action="store_true",
        help="Also benchmark the untuned base-model candidate on the same suite "
        "(skipped when it already has a current result).",
    )
    ev.add_argument("--force", action="store_true", help="Run even when a current result exists.")
    ev.add_argument("--require-gpu", action="store_true", help="Fail when no GPU is visible.")
    ev.add_argument("--no-regenerate", action="store_true", help="Do not update figures/report.")
    ev.add_argument("--root", type=Path, default=None, help="Project root (default: auto).")
    ev.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ev.add_argument("--log-format", default="text", choices=["text", "json"])
    ev.set_defaults(dispatch=run_benchmark_command, handler=_cmd_evaluate)


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from steward.cli import default_ollama_url
    from steward.evaluation.compare import compare
    from steward.evaluation.records import CandidateRole, RecordStatus
    from steward.evaluation.service import (
        Outcome,
        default_provider_factory,
        ingest_training_runs,
        regenerate,
        run_benchmarks,
    )

    project = _project(args)
    registry = project.registry
    ref = Path(args.model)
    dataset_version = None
    if ref.is_dir():
        run_dir = ref.resolve()
        if not (run_dir / "training_summary.json").is_file():
            run_dir = run_dir.parent
        ingests = {i.path.parent.resolve(): i for i in ingest_training_runs(project)}
        ingest = ingests.get(run_dir)
        if ingest is None:
            print(f"error: no training_summary.json under runs/ for {ref}", file=sys.stderr)
            return EXIT_FAILURE
        if ingest.candidate is None or ingest.record is None:
            print(f"error: {run_dir.name} cannot be evaluated: {ingest.message}", file=sys.stderr)
            return EXIT_FAILURE
        candidate = ingest.candidate
        dataset_version = ingest.record.summary.dataset_version
    else:
        candidate = registry.get(args.model)
    if args.suite:
        project.suite_named(args.suite)  # validate the name
        candidate = candidate.model_copy(update={"suite": args.suite})
    plan = [(candidate, dataset_version)]
    baseline = None
    if args.baseline:
        bases = [
            c
            for c in registry.candidates
            if c.role is CandidateRole.BASE
            and c.provider == candidate.provider
            and c.model == candidate.model
            and c.id != candidate.id
        ]
        if not bases:
            print(f"error: no base-model candidate for {candidate.model}", file=sys.stderr)
            return EXIT_FAILURE
        baseline = bases[0].model_copy(update={"suite": candidate.suite})
        plan.insert(0, (baseline, None))
    factory = default_provider_factory(default_ollama_url())
    outcomes = []
    for cand, version in plan:
        outcomes += run_benchmarks(
            project,
            [cand],
            factory,
            force=args.force and cand is candidate,
            require_gpu=args.require_gpu,
            dataset_version=version,
        )
    for o in outcomes:
        print(f"{o.outcome.value:<12} {o.candidate_id}: {o.message}")
    if any(o.outcome is Outcome.APPENDED for o in outcomes) and not args.no_regenerate:
        regenerate(project)
        print("Updated figures, docs/benchmarks.md, benchmarks/results/summary.json, README.md.")
    if baseline is not None:
        key = project.key_for(candidate.suite)
        latest = {}
        for r in project.benchmarks():
            if r.status is RecordStatus.COMPLETED and r.comparability_key == key:
                latest[r.candidate.candidate_id] = r
        cand_rec, base_rec = latest.get(candidate.id), latest.get(baseline.id)
        if cand_rec is not None and base_rec is not None:
            comp = compare(cand_rec, base_rec, project.thresholds, "baseline")
            for f in comp.findings:
                if f.kind.value != "unchanged":
                    print(f"vs baseline: {f.kind.value.upper()} {f.metric}: {f.message}")
    bad = [o for o in outcomes if o.outcome in {Outcome.FAILED, Outcome.UNAVAILABLE}]
    return EXIT_FAILURE if bad else EXIT_OK


def add_sweep_select_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register ``sweep-select``: evaluate sweep trials on the development suite and rank."""
    sel = subparsers.add_parser(
        "sweep-select",
        help="Evaluate completed sweep trials on the development suite and select one.",
        description="Benchmarks each completed trial on the sweep's selection suite (a "
        "development suite, never the final golden suite), ranks the trials (critical "
        "failures, technical errors, meaning preservation, best validation loss), and "
        "writes the result to runs/sweeps/<sweep_id>/sweep.json.",
    )
    sel.add_argument("--sweep", type=Path, required=True)
    sel.add_argument("--require-gpu", action="store_true")
    sel.add_argument("--root", type=Path, default=None)
    sel.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sel.add_argument("--log-format", default="text", choices=["text", "json"])
    sel.set_defaults(dispatch=run_benchmark_command, handler=_cmd_sweep_select)


def _cmd_sweep_select(args: argparse.Namespace) -> int:
    from steward.cli import default_ollama_url
    from steward.evaluation.records import RecordStatus
    from steward.evaluation.service import (
        default_provider_factory,
        ingest_training_runs,
        regenerate,
        run_benchmarks,
    )
    from steward.training.config import TrainingConfigError
    from steward.training.sweep import RANK_METRICS, load_state, load_sweep, save_state, select

    project = _project(args)
    root = project.paths.root
    try:
        sweep = load_sweep(args.sweep)
    except TrainingConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    state = load_state(root, sweep, args.sweep)
    ingests = {i.path.parent.resolve(): i for i in ingest_training_runs(project)}
    factory = default_provider_factory(default_ollama_url())
    key = project.key_for(sweep.selection_suite)
    for trial in state.trials:
        if trial.status != "completed" or not trial.run_dir:
            continue
        ingest = ingests.get((root / trial.run_dir).resolve())
        if ingest is None or ingest.candidate is None:
            trial.error = f"cannot be evaluated: {ingest.message if ingest else 'no summary'}"
            continue
        candidate = ingest.candidate.model_copy(update={"suite": sweep.selection_suite})
        assert ingest.record is not None
        run_benchmarks(
            project,
            [candidate],
            factory,
            require_gpu=args.require_gpu,
            dataset_version=ingest.record.summary.dataset_version,
        )
        records = [
            r
            for r in project.benchmarks()
            if r.candidate.candidate_id == candidate.id
            and r.comparability_key == key
            and r.status is RecordStatus.COMPLETED
        ]
        if records:
            rec = records[-1]
            trial.dev_experiment_id = rec.experiment_id
            names = (
                *RANK_METRICS,
                "ste_rule_pass_rate",
                "ste_only_output_rate",
                "negation_preservation",
                "safety_label_preservation",
                "terminology_accuracy",
                "unnecessary_rewrite_rate",
            )
            trial.dev_metrics = {
                n: (rec.metrics[n].value if n in rec.metrics else None) for n in names
            }
    selected = select(state)
    path = save_state(root, state)
    regenerate(project)
    for t in sorted(state.trials, key=lambda t: t.rank or 99):
        m = t.dev_metrics
        print(
            f"rank {t.rank or '-'} {t.name:<4} critical {m.get('critical_failures')} "
            f"tech-error {m.get('technical_error_rate')} meaning {m.get('meaning_preservation')} "
            f"best-val {t.best_val_loss} ({t.dev_experiment_id or t.error or t.status})"
        )
    print(f"Selected: {selected or 'none'}  State: {path}")
    return EXIT_OK if selected else EXIT_FAILURE


def run_benchmark_command(args: argparse.Namespace) -> int:
    """Entry point from :mod:`steward.cli`."""
    from steward.evaluation.history import HistoryError
    from steward.evaluation.suite import BenchmarkConfigError
    from steward.generation.prompts import PromptTemplateError
    from steward.logs import configure_logging

    configure_logging(args.log_level, args.log_format)
    try:
        result: int = args.handler(args)
        return result
    except (BenchmarkConfigError, HistoryError, PromptTemplateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE


def _project(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from steward.evaluation.project import BenchmarkProject

    return BenchmarkProject.at(args.root.resolve() if args.root else find_root())


def _cmd_run(args: argparse.Namespace) -> int:
    from steward.cli import default_ollama_url
    from steward.evaluation.service import (
        Outcome,
        default_provider_factory,
        ingest_training_runs,
        regenerate,
        run_benchmarks,
    )

    project = _project(args)
    registry = project.registry
    # Finished training runs (runs/*/training_summary.json) become history records
    # and benchmark candidates. Nothing here starts training.
    trained = {}
    for ingest in ingest_training_runs(project):
        if ingest.appended:
            print(f"recorded     training run {ingest.path.parent.name}: {ingest.message}")
        if ingest.candidate is not None and ingest.record is not None:
            trained[ingest.candidate.id] = (ingest.candidate, ingest.record.summary.dataset_version)
    if args.candidate:
        candidates = [trained[c][0] if c in trained else registry.get(c) for c in args.candidate]
    else:
        candidates = [*registry.candidates, *(c for c, _ in trained.values())]
    if not candidates:
        print("error: no candidates registered in benchmarks/models.toml", file=sys.stderr)
        return EXIT_FAILURE
    factory = default_provider_factory(args.base_url or default_ollama_url())
    outcomes = []
    for candidate in candidates:
        outcomes += run_benchmarks(
            project,
            [candidate],
            factory,
            force=args.force,
            require_gpu=args.require_gpu,
            dataset_version=trained[candidate.id][1] if candidate.id in trained else None,
        )
    for o in outcomes:
        print(f"{o.outcome.value:<12} {o.candidate_id}: {o.message}")
    if any(o.outcome is Outcome.APPENDED for o in outcomes) and not args.no_regenerate:
        regenerate(project)
        print("Updated figures, docs/benchmarks.md, benchmarks/results/summary.json, README.md.")
    bad = [o for o in outcomes if o.outcome in {Outcome.FAILED, Outcome.UNAVAILABLE}]
    if bad:
        print(f"{len(bad)} candidate(s) did not produce a completed result.", file=sys.stderr)
        return EXIT_FAILURE
    return EXIT_OK


def _cmd_history(args: argparse.Namespace) -> int:
    from steward.evaluation.records import BenchmarkRecord

    project = _project(args)
    records = [r for r in project.history() if args.type in (None, r.record_type)]
    if args.json:
        print(
            json.dumps(
                [
                    r.model_dump(mode="json", exclude={"items"})
                    if isinstance(r, BenchmarkRecord)
                    else r.model_dump(mode="json")
                    for r in records
                ],
                indent=2,
            )
        )
        return EXIT_OK
    if not records:
        print("The experiment history is empty.")
        return EXIT_OK
    for r in records:
        if isinstance(r, BenchmarkRecord):
            m = r.metrics.get("ste_rule_pass_rate")
            score = f"pass={m.value:.2f}" if m and m.value is not None else ""
            name = r.candidate.candidate_id
            key = r.comparability_key
        else:
            score, name, key = f"steps={len(r.summary.curve)}", r.summary.run_id, ""
        print(
            f"{r.created_at:%Y-%m-%d %H:%M} {r.experiment_id} {r.record_type:<9} "
            f"{r.status.value:<9} {name:<24} {score:<10} {key}"
        )
    return EXIT_OK


def _find(records, ident: str):  # type: ignore[no-untyped-def]
    from steward.evaluation.compare import latest_by_candidate

    for r in records:
        if r.experiment_id == ident:
            return r
    return latest_by_candidate(records).get(ident)


def _cmd_compare(args: argparse.Namespace) -> int:
    from steward.evaluation.compare import compare, select_model
    from steward.evaluation.report import comparisons

    project = _project(args)
    records = project.benchmarks()
    if args.candidate and args.against:
        cand, ref = _find(records, args.candidate), _find(records, args.against)
        if cand is None or ref is None:
            print("error: candidate or reference has no completed result.", file=sys.stderr)
            return EXIT_FAILURE
        comps = [compare(cand, ref, project.thresholds, "explicit")]
    else:
        comps = comparisons(project)
        if args.candidate:
            comps = [
                c
                for c in comps
                if args.candidate in {c.candidate.candidate.candidate_id, c.candidate.experiment_id}
            ]
    if not comps:
        print("Nothing to compare yet: a comparison needs two completed, comparable results.")
    exit_code = EXIT_OK
    for comp in comps:
        head = (
            f"{comp.candidate.candidate.candidate_id} vs {comp.reference_kind} "
            f"{comp.reference.candidate.candidate_id}"
        )
        if not comp.comparable:
            print(
                f"{head}: NOT COMPARABLE ({comp.candidate.comparability_key} vs "
                f"{comp.reference.comparability_key})."
            )
        for f in comp.findings:
            if f.kind.value != "unchanged":
                sig = " [CIs separated]" if f.significant else ""
                print(f"{head}: {f.kind.value.upper()} {f.metric}: {f.message}{sig}")
        if comp.regressions:
            exit_code = EXIT_FAILURE
    from steward.evaluation.compare import latest_by_candidate

    latest = latest_by_candidate(records, project.key)
    if latest:
        selected, _ = select_model(latest, project.thresholds.selection)
        print(
            "Selected model: "
            + (
                selected.candidate.candidate_id
                if selected
                else "none (no candidate meets the requirements)"
            )
        )
    return exit_code


def _cmd_plot(args: argparse.Namespace) -> int:
    from steward.evaluation.plots import render_all

    project = _project(args)
    paths = render_all(
        project.benchmarks(), project.trainings(), project.paths.figures, primary_key=project.key
    )
    print(f"Wrote {len(paths)} files to {project.paths.figures}.")
    return EXIT_OK


def _cmd_report(args: argparse.Namespace) -> int:
    from steward.evaluation.service import regenerate

    project = _project(args)
    written = regenerate(project, figures=not args.no_figures)
    print(f"Updated {len(written)} file(s).")
    return EXIT_OK


def _cmd_status(args: argparse.Namespace) -> int:
    from steward.automation.state import HookState
    from steward.evaluation.fingerprints import dataset_fingerprint
    from steward.evaluation.status import (
        evaluator_version_warnings,
        pending_candidates,
        result_states,
    )

    project = _project(args)
    records = project.benchmarks()
    dataset = dataset_fingerprint(project.paths.root)
    states = result_states(records, project.keys, dataset)
    pending = pending_candidates(project.registry, records, project.keys)
    hook = HookState.load(project.paths.root)
    data = {
        "comparability_key": project.key,
        "suite_review_state": project.suite.manifest.review_state,
        "dataset_fingerprint": dataset,
        "results": [
            {
                "experiment_id": s.record.experiment_id,
                "candidate": s.record.candidate.candidate_id,
                "status": s.status.value,
                "reasons": s.reasons,
            }
            for s in states
        ],
        "pending_candidates": pending,
        "evaluator_warnings": evaluator_version_warnings(project, records),
        "hook_pending_changes": hook.pending_summary(),
        "outstanding": hook.outstanding,
        "last_stop": hook.last_stop,
        "hooks_mode": hook.mode(),
    }
    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return EXIT_OK
    print(f"Comparability key: {data['comparability_key']} (suite: {data['suite_review_state']})")
    print(f"Local dataset fingerprint: {dataset or 'no data/ files'}")
    print(f"Hooks mode: {data['hooks_mode']}")
    print(f"Results: {len(states)}")
    for s in states:
        extra = f" ({'; '.join(s.reasons)})" if s.reasons else ""
        print(
            f"  {s.status.value:<9} {s.record.experiment_id} "
            f"{s.record.candidate.candidate_id}{extra}"
        )
    print("Pending candidates: " + (", ".join(pending) or "none"))
    for w in data["evaluator_warnings"]:
        print(f"WARNING: {w}")
    changes = data["hook_pending_changes"]
    print("Changes waiting for validation: " + (json.dumps(changes) if changes else "none"))
    for item in hook.outstanding:
        print(f"OUTSTANDING: {item}")
    if hook.last_stop:
        print(
            f"Last Stop-hook run: {hook.last_stop.get('finished_at')} "
            f"result={hook.last_stop.get('result')}"
        )
    return EXIT_OK


def _cmd_verify(args: argparse.Namespace) -> int:
    from steward.evaluation.verify import IssueLevel, verify

    project = _project(args)
    issues = verify(project, generated=args.check_generated, datasets=not args.no_datasets)
    for issue in issues:
        print(f"{issue.level.value.upper():<7} [{issue.check}] {issue.message}")
    errors = [i for i in issues if i.level is IssueLevel.ERROR]
    print(f"verify: {len(errors)} error(s), {len(issues) - len(errors)} warning(s).")
    return EXIT_FAILURE if errors else EXIT_OK
