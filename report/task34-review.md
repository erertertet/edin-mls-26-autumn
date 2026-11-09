# Task 3 and Task 4 independent review

Three independent AI agents reviewed the completed implementation and cluster
evidence before branch publication. None participated in implementing or
writing the files they reviewed. Reviews were read-only, scoped to actual
correctness, coursework requirements, regressions and reproducibility. They
did not request speculative input hardening or unrelated refactoring.

## Reviewed state

Base: `ae01d2939b2bae74c9ee3adcf0d82d05963e7d4d` (Task 2 merged).
Runtime evidence: Slurm job `3707553`, 2026-11-09, H200 MIG 1g.18gb.
Kernel SHA-256 hashes are recorded in the [results](task34-kernels-results.md)
and match the tested files. No kernel changes were made after the successful
cluster run or in response to review.

| Reviewer | Scope | Outcome |
| --- | --- | --- |
| `review_task3_independent` | GEMM indices, coalesced loads, edge tiles, both barriers, bias/GELU, official and extra cases | No actionable findings |
| `review_task4_independent` | GQA, shifted causal mask, inactive warps, online rescaling, warp participation, shared-memory ordering | No actionable findings |
| `review_integration_independent` | Batch failure propagation, tests, hashes, experiment medians, report claims, evidence packaging | No substantive findings; packaging reminder resolved |

The integration reviewer noted that `report/.gitignore` ignores `*.log`.
All 59 intended raw logs and `source.sha256` were explicitly staged; the
reviewer independently verified their inclusion. Downloaded course materials,
build outputs and model weights are excluded.

## Evidence and limits

The reviewers checked source and raw logs rather than relying only on a summary.
They confirmed 33 official cases, five additional cases, nine clean sanitizer
runs, four correct model modes and 264 variant case/round results. The
integration reviewer recomputed reported medians and checked source hashes.
No additional local compilation, tests, inference or GPU execution was performed.

Review does not extend runtime coverage: validation used one GPU/MIG profile
and one supplied recording; large attention cases use sampled CPU references;
large cases and experimental variants were not sanitizer-tested. These limits
remain disclosed in the results. No unresolved actionable findings remained
before publication. The PR remains subject to the user's review and approval.
