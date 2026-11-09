# Task 2 reductions: cluster validation and report notes

Task 2 passes all 11 supplied cases, the extra helper checks, the Task 1
regression tests and the combined Task 1+2 transcription check. Production
uses 256 threads, standard mathematics and both shared-scratch barriers.
All compilation and execution took place on the Edinburgh cluster; no local
GPU was used. This change is intended for a feature-branch pull request and
must be reviewed before merging into `main`.

## Implementation and scope

The seven blanks in `task2_reduce.cuh` and `task2_reduce.cu` now implement:

- Warp sum and maximum using a five-step XOR shuffle butterfly.
- Block maximum using per-warp shared partials and two barriers. Unused entries
  contribute negative infinity, so entirely negative rows remain correct.
- Stable `MaxSum` merging, with explicit empty-summary identities to avoid
  evaluating `-infinity - -infinity`.
- RMSNorm from a block sum of squares and a reciprocal square root.
- Two-pass softmax: build a local summary per thread, find the block maximum,
  rescale and sum the summaries, then reread the input to write probabilities.

All threads participate in the collective operations, including threads without
input columns. The supplied LayerNorm, launch interfaces, official tests and
Makefile are unchanged. Task 3 and Task 4 remain unfinished.

## Environment and provenance

| Item | Recorded value |
| --- | --- |
| Date / Slurm job | 2026-11-09 UTC / `3707496` |
| Node / completion | `saxa.inf.ed.ac.uk` / `COMPLETED`, exit `0:0`, 00:04:24 |
| Allocation | 1 GPU, 2 CPU cores, 16 GB host memory |
| Device | NVIDIA H200 **MIG 1g.18gb**, 16 SMs, `sm_90` |
| Driver / nvcc | `595.91.07` / `12.8.93` |
| Compute Sanitizer | `2025.1.0.0` |
| Python / PyTorch | `3.11.18` / `2.10.0+cu128` |
| Transformers / NumPy | `5.0.0` / `2.4.6` |
| Safetensors / Hugging Face Hub | `0.8.0` / `1.33.0` |
| Repository base | `17e638c540aa521d31e5de1d482b1c94b6c422ba` |
| Model snapshot | `61ba4e0b3309b6656edea3e93e419f7bd5c61957` |

The source SHA-256 values are:

```text
task2_reduce.cu  e6abcaeac38a642114626227965ce34519c2d16e25b017f4ccfc16717349b796
task2_reduce.cuh 29dc47f65038539c76c2ba54dbb1831f5e29f1f5008ab6168759e6c14534bfdb
```

See [raw evidence](task2-evidence/2026-11-09-job-3707496/) for the full environment,
build commands, per-run logs, Slurm accounting and source checksums. The final
integrity check confirms the tested sources were unchanged by the experiments.
The allocation was released on completion. The cluster checkout is isolated at
`~/mls-worktrees/task2-reductions`, preserving the original checkout's edits.

## Correctness

The normalizations use `atol=1e-4`, softmax uses `atol=1e-7`, and both use
`rtol=1e-4`, as specified by the unmodified CPU-reference tests.

| Case | Result | Maximum absolute error |
| --- | --- | ---: |
| RMSNorm, 1 x 40 | PASS | 8.27e-8 |
| RMSNorm, 5 x 1001 | PASS | 3.68e-7 |
| RMSNorm, 64 x 2048 | PASS | 4.35e-7 |
| LayerNorm, 5 x 1001 | PASS | 3.91e-7 |
| LayerNorm, 1500 x 1280 | PASS | 5.42e-7 |
| Softmax, 1 x 7 | PASS | 1.11e-8 |
| Softmax, 3 x 1001 | PASS | 1.04e-9 |
| Softmax, 4 x 513, inputs +/-100 | PASS | 2.67e-8 |
| Softmax, 2 x 300, inputs -200 to -150 | PASS | 6.11e-9 |
| Softmax, 1 x 59264 | PASS | 8.13e-11 |
| Softmax, 64 x 4096 | PASS | 7.95e-10 |

Helper tests check all lanes at six block sizes, repeated scratch reuse, negative
maxima, empty summaries, merge grouping and a 200-unit input gap. All passed;
integer reductions were exact, and finite summary error was at most 1.12e-7.
All four Task 1 cases also passed.

Memcheck, racecheck and synccheck each passed on both the official small-case
suite and helper tests. Racecheck reported zero errors and zero warnings.
The large audio LayerNorm and 64 x 4096 softmax cases passed the full ordinary
test but are excluded by the supplied sanitizer `small` filter.

## Model integration

Both modes returned 100% supplied word accuracy and `Status: PASS` on the same
3.50-second recording, producing:

> Concord returned to its place amidst the tents.

| Mode | Mean time, three runs | Peak GPU memory |
| --- | ---: | ---: |
| `--cuda task1 --profile` | 1305.0 +/- 0.5 ms | 8.90 GiB |
| `--cuda task1,task2 --profile` | 1294.7 +/- 0.2 ms | 8.90 GiB |

These measurements exclude model loading and follow one warm-up. The observed
end-to-end reduction is approximately 0.8%, from a short experiment on one device
and one sample; it is not evidence of a general speedup or general ASR accuracy.

The separate profiling pass shows the trade-offs clearly:

| Operation | Calls | Task 1 only, Torch ms | Task 1+2, CUDA ms |
| --- | ---: | ---: | ---: |
| RMSNorm | 741 | 17.9 | 6.2 |
| LayerNorm | 65 | 3.5 | 4.0 |
| Softmax | 13 | 0.3 | 1.8 |

RMSNorm improved, while LayerNorm and vocabulary softmax were slower than the
framework path. Profiling introduces instrumentation and is a separate pass;
its operation times should not be equated with the unprofiled wall-clock means.

## Block-size and atomic experiments

Every variant passed all 11 cases in three runs. The second round reversed the
order. Each test timing uses three warm-ups and 20 CUDA-event-timed calls.
The table reports median **effective GB/s**, using the supplied test's convention.

| Threads | RMSNorm 64x2048 | LayerNorm 1500x1280 | Softmax 1x59264 | Softmax 64x4096 |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 44.9 | 166.2 | 0.6 | 52.6 |
| 64 | 81.7 | 267.9 | 1.2 | 98.8 |
| 128 | 141.0 | 272.4 | 2.5 | 169.8 |
| **256 (retained)** | **214.2** | **258.2** | **4.9** | **237.9** |
| 512 | 278.9 | 205.6 | 9.5 | 252.7 |
| 1024 | 226.5 | 134.3 | 16.4 | 222.9 |

The 256-thread vocabulary run took about 0.097 ms. Larger blocks help distribute
that long row among more threads, but hurt the many-row LayerNorm case. There is
no single winner across the supplied shapes; the validated default is retained.
This does not establish that 256 is optimal for end-to-end inference.

Replacing only RMSNorm's shuffle reduction with a shared `atomicAdd` accumulator
also passed ordinary and sanitizer checks, but its decoder-case median fell
from 214.2 to 63.2 GB/s (approximately 3.4 times longer for the same workload).
The single shared accumulator creates contention, consistent with the slowdown.
The atomic version was not selected.

The bandwidth numerator is `2 * rows * cols * sizeof(float)` for every supplied
operation. It excludes additional input passes, weights, biases and cache or
transaction effects. It is not physical DRAM traffic or bandwidth utilization.

## Synchronization experiment and report question 4

Removing only the final barrier in `block_reduce_sum`, in an isolated copy,
left every ordinary test passing. Racecheck nevertheless reported **one error**
between the shared read and write sites, aggregating **160 hazards**. The
deliberately faulty run was rejected by the sanitizer. Production retains the
barrier and has a clean racecheck result.

The possible sequence is:

1. A fast warp finishes reading the first call's shared partial sums.
2. It enters LayerNorm's second reduction and writes a new warp sum into the
   same shared slot.
3. A slower warp is still loading that slot for the first reduction, so it may
   read a value from the wrong invocation.

The final barrier prevents the second call's writes until all warps have finished
their reads. Ordinary numerical agreement in one execution cannot prove absence
of this timing-dependent hazard. This matches the purpose of NVIDIA's
[racecheck tool](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#racecheck-tool).

## Report question 3: input passes

This softmax reads each input element twice: once while constructing a thread's
`MaxSum`, then once to write the normalized output. Reducing and rescaling the
summaries requires no further input reads. The textbook max/sum/write approach
reads each input three times. Including the output write gives logical float32
traffic of `12 * rows * cols` versus `16 * rows * cols` bytes, excluding small
summary storage and implementation/cache effects. Fewer passes do not guarantee
higher throughput: online merging adds dependent arithmetic and exponentials.

## Long-row alternative: design sketch

The vocabulary case launches only one block, so only one of this MIG device's
16 SMs can execute that block. A possible multi-block implementation would:

1. Split a row into tiles; each block writes one `MaxSum` summary.
2. Launch a reduction over those summaries to obtain each row's maximum and sum.
3. Launch many blocks to normalize and write the row tiles.

Kernel boundaries provide global ordering; block barriers cannot synchronize
different blocks. This exposes more parallelism but adds scratch storage and
launch overhead, so it needs separate measurement. It is a design sketch only,
not an implemented or benchmarked optimization in this change.

## Reproduction, reuse and assistance

Follow [the cluster instructions](../hw1-asr/docs/task2-validation.md) and submit
`sbatch run_task2.sbatch` from the checkout's `hw1-asr/` directory. Experimental
source copies are generated from the exact validated sources. No new dependency
was added; the course Makefile, CPU references, model runner and installed
environment were reused.

The implementation was checked against the
[CUDA 12.8 shuffle and synchronization APIs](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#warp-shuffle-functions),
[math APIs](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html)
and the course's online-normalizer formula, also described by
[Milakov and Gimelshein](https://arxiv.org/abs/1805.02867).
[CUB BlockReduce](https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/block_reduce.cuh)
is an existing general solution, but delegating the required reductions to it
would replace the coursework's own helper implementation. The small custom
kernels therefore follow the supplied skeleton and NVIDIA intrinsics.

Codex assisted with implementation, tests, orchestration, cluster execution,
analysis and these notes. Include this assistance in the final report's required
AI-use declaration after reviewing the work. This Task 2 record does not replace
the final all-task tests and full-model profile required once all tasks are done.
