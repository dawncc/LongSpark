# Historical execution provenance

These files describe the final pre-cleanup native ModelRunner V2 benchmark.
`audit-composition.py` is the original audit tool whose digest is recorded in
`audit-full.json`. Its source-integrity phase requires the execution-time source
layout. V2-only cleanup moved helper functions and removed the old manual
modules, so that historical integrity check is not a current-tree check.

Use `scripts/audit_native_v2_gsm8k.py` with the current `validation/gsm8k100`
fixture to recompute the preserved output, accuracy, acceptance and graph
results. See the cleanup report for verification of the current source tree.
