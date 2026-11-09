# Task 3 and Task 4: cluster validation

Run compilation, tests, inference and experiments on the Edinburgh cluster in a
Slurm GPU allocation. Local machines are for editing and lightweight inspection.

## Submit and reproduce

Reuse the `mls-asr` environment and cached model. Check existing allocations first,
then submit from this checkout's `hw1-asr/` directory:

```bash
squeue -u "$USER"
cd /path/to/checkout/hw1-asr
sbatch run_task34.sbatch
```

The job requests one GPU, two CPU cores, 16 GB host memory and one hour. It uses
CUDA 12.8 and standard course compilation flags. Conda defaults to `/opt/conda`
or `~/miniconda3`; `CONDA_BASE` and `ASR_ENV` provide overrides. The model runs
offline. Logs are in `task34-JOB_ID.out` and `build/task34-validation/JOB_ID/`.

## Acceptance checks

- Build the shared library, all supplied tests, existing Task 2 helpers and the
  extra Task 3/4 edge-case test.
- Run `make test`: four Task 1, eleven Task 2, ten GEMM and eight attention cases.
- Run Task 2 helper checks, which cover reductions reused by attention.
- Run extra GEMM cases for GELU without bias and tile boundaries, plus attention
  cases for 96-dimensional heads and single-key GQA, using the supplied CPU references.
- Run memcheck, racecheck and synccheck on both supplied Task 3/4 `small` suites.
  Run the same checks on the extra edge cases. Sanitizer findings produce a
  nonzero exit code and stop the job.
- Run model profiles for `--cuda none`, `--cuda task1,task2`,
  `--cuda task1,task2,task3` and `--cuda all` on the same allocated device.

The official GEMM small filter selects three cases. Attention's filter selects
seven cases, excluding the audio-encoder case. All cases still
run in the ordinary suite. The attention reference checks all rows where its
sampling step is one and approximately 300 rows otherwise, including the first
and last; passing a large case is not an exhaustive elementwise CPU comparison.
Neither supplied tests nor their tolerances are changed.

## Controlled experiments

Source copies under the job's build directory compare GEMM tiles 8/16/32,
attention blocks of 1/2/4/8/16 warps, and each kernel with shared-memory padding
removed. Each configuration runs three complete correctness/timing rounds;
the second round reverses configuration order. Original sources remain at
`TILE=16` and `ATTN_WARPS=4`, with padding, and checksums verify preservation.
Experimental variants are not used for model inference.

The official tests report CUDA-event timings after warm-ups. GEMM's naive and
cuBLAS timings cover multiplication alone, while the coursework kernel includes
bias/GELU when requested: these are useful reference costs, not equivalent fused
workloads. cuBLAS uses the supplied pedantic FP32 setting. Attention GFLOP/s is
an approximate operation-count metric; it does not measure DRAM bandwidth.

Model measurements use one warm-up and three timed transcriptions, with a
separate instrumented profiling pass. Record mean time, variation, accuracy,
memory and per-operation times, and distinguish this sample from general ASR
quality or broad performance claims.

## Evidence and review

Keep environment/import-path records, GPU and driver details, build logs,
sanitizer output, all timing rounds, model profiles, source hashes and Slurm
completion status. Source hashes connect uncommitted working files to the run.
Independent agents review kernel correctness and integration/evidence after
validation. Resolve actionable findings and rerun affected checks before pushing
the feature branch and creating a PR for user review; do not merge it.
