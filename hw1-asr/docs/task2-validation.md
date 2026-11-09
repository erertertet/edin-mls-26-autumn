# Task 2: cluster validation

Run all compilation, tests, inference and experiments in an Edinburgh Slurm GPU
allocation. Local machines are only for editing and lightweight inspection.

## Submit

Reuse the `mls-asr` environment and cached model prepared with `setup_env.sh`.
Check existing jobs before requesting another GPU. From the cluster checkout:

```bash
squeue -u "$USER"
cd /path/to/checkout/hw1-asr
sbatch run_task2.sbatch
```

The job requests one GPU, two CPU cores, 16 GB host memory and one hour. It uses
the existing Makefile and runs model inference offline. Conda defaults to
`/opt/conda` or `~/miniconda3`; `CONDA_BASE` and `ASR_ENV` can override these.
Inspect `task2-JOB_ID.out` and `build/task2-validation/JOB_ID/`.

## Acceptance checks

- Compile the library, supplied tests and `test_task2_helpers`.
- Run the four Task 1 regression cases and all 11 supplied Task 2 cases.
- Test helper contracts: every lane receives the reduction, negative maxima,
  repeated calls, empty-summary identities, merge grouping and large value gaps.
- Run memcheck, racecheck and synccheck on the small supplied cases and helper
  tests. These runs must succeed; sanitizer findings return a nonzero exit code.
- Run `./run.sh --cuda task1 --profile`, then
  `./run.sh --cuda task1,task2 --profile` on the same GPU.

The helper binary is built through the Makefile's existing pattern rule:
`make build/sm_XX/test_task2_helpers`, using the GPU's actual `XX`. The supplied
test files and Makefile are unchanged.

## Experiments

The script creates source copies under the job's build directory. It compares
32, 64, 128, 256, 512 and 1024 threads, plus an RMSNorm version using shared
`atomicAdd`. Each variant runs three times, reversing order in the second round.
The atomic version also undergoes the three sanitizer checks.

A separate, deliberately unsafe copy removes the final scratch-reuse barrier
from `block_reduce_sum`. Its ordinary test and racecheck results are recorded,
including expected failures. It is never used for model inference or acceptance.
Passing that ordinary test does not establish race freedom.

The working sources remain unchanged during experiments, verified by checksums.
The standard flags and correct synchronization are retained. The long-vocabulary
multi-block alternative is a design sketch in the results notes, not an additional
production kernel.

## Interpreting results

The supplied tests report effective bandwidth using `2 * rows * cols * 4` bytes
for every operation. This convention omits repeated input reads, weight/bias
accesses, caches and physical transactions. Treat it as a comparison metric,
not measured DRAM bandwidth. Record the actual GPU or MIG profile and preserve
the raw logs with the report. Complete the change through a feature-branch pull
request for review; do not merge it or update `main` directly.
