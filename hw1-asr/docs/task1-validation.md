# Task 1: cluster validation

Run the coursework on the Edinburgh cluster. Local machines are for editing and
lightweight inspection only; do not run CUDA builds or tests locally.

## Prepare and submit

On the cluster login node, finish the environment and model download described in
the [homework README](../README.md). The validation job expects the `mls-asr`
environment and a cached `zai-org/GLM-ASR-Nano-2512` model. It runs offline.

Check for an existing allocation before requesting another GPU:

```bash
squeue -u "$USER"
cd ~/edin-mls-26-autumn/hw1-asr
sbatch run_task1.sbatch
```

The job requests one GPU, two CPU cores, 16 GB of host memory and one hour.
It discovers conda at `/opt/conda` or `~/miniconda3`; set `CONDA_BASE` or `ASR_ENV`
before submission if your installation differs. To inspect the job:

```bash
squeue -u "$USER"
tail -f task1-JOB_ID.out
```

## Checks and experiments

The script uses the supplied Makefile and unchanged reference tests:

1. Compile the kernels and tests with standard mathematics and 256 threads.
2. Run all four RoPE cases; run Compute Sanitizer on the small cases if available.
3. Check the PyTorch transcription baseline, then Task 1 transcription and profiling.
4. Compare 32, 64, 128, 256, 512 and 1024 threads, three runs each on the same GPU.
5. Run a separate fast-math accuracy experiment.

Experimental source copies and binaries are isolated under
`build/task1-validation/JOB_ID/`. The original kernel and its default build are
preserved. A fast-math comparison failure is recorded as an experiment; any failure
in the standard-math tests or model validation stops the job.

## Evidence

The same directory contains environment versions, import paths, model revision,
source checksums and individual logs. Preserve these with any reported results.
The test's GB/s metric counts Q/K reads and writes; it is effective bandwidth,
not a measurement of all DRAM traffic. Three short runs do not establish a
universal best block size. A completed Slurm job alone is not the acceptance
criterion: inspect the correctness and transcription logs.
