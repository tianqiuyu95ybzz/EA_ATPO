# Legacy local asset snapshot (2026-11-22)

Source: old clone `EA_ATPO`, branch `main`, HEAD `ad9d0e31660b973a3e5ddfcadce6701f767e0f62`. This directory preserves reviewed source/config/report updates that were present locally but not represented by matching blobs on the current V3 migration base. The source clone was not modified.

Included: relative-value Router code, minimal-fix diagnostics, their test/validation commands, the local relative-value config, an aggregate precheck `offline_split_metrics.csv`, a sanitized aggregate `training_summary.json`, and the method audit. The copied training summary omits its validation row IDs; only the case count and aggregate metrics are retained. The version of `run_stepwise_router_eval.py` reflects the one tracked modification in the old clone. The model-service shell launchers are historical references and were not executed.

Excluded: case split, case/state IDs, state-action targets, all Router checkpoint NPZ/JSON files, per-case or free-text output, logs, model caches/weights, and download artifacts. Those assets remain only in restricted old-server storage. The copied code may require those excluded private inputs and is not the current V3 experiment or a claim of generalization.
