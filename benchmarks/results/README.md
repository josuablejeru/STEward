# Benchmark results

* `history.jsonl`: append-only experiment history (benchmark and training
  records). Each line stores its SHA-256 and the hash of the previous line.
  Never edit or delete lines; `pixi run benchmark-check` detects changes.
* `summary.json`: generated summary of the latest comparable results.

Regenerate summaries, figures, and the report with `pixi run benchmark-report`.
