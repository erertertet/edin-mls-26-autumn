# Task 3 and Task 4: implementation and cluster results

The four coursework tasks pass the combined official tests, and the model
transcribes the supplied recording correctly with every kernel group on CUDA.
The default coursework implementation is slower than the framework baseline on
this device. It uses less peak allocated GPU memory in this sample; performance
and accuracy conclusions are limited to these workloads.

## Implementation

Task 3 fills four blanks: bounded input and transposed-weight tile loads,
shared-memory multiplication with publication/reuse barriers, and a fused
bias/GELU epilogue. Adjacent x threads read adjacent weight elements. Threads
outside the output still participate in loads and both barriers.

Task 4 fills six blanks: GQA mapping, cooperative K/V loading, masked scores,
online maximum/sum/output rescaling, weighted-value accumulation and output.
All warps follow the same tile loop, including spare warps. A publication
barrier also makes shared queries ready on the first iteration. Empty summaries
avoid subtracting negative infinities. The fixed accumulator array and unrolled
loops expose constant indices to the compiler.

Defaults remain `TILE=16` and `ATTN_WARPS=4`, with shared-memory padding.
The Task 1/2 code, supplied tests and tolerances, Makefile, model and Python
wrappers are unchanged. No new dependency was added.

## Environment and provenance

All compilation, tests, inference and experiments ran inside Slurm job
`3707553` on `saxa.inf.ed.ac.uk`, starting 2026-11-09 03:24:22 UTC. No local GPU
was used. The isolated worktree is `~/mls-worktrees/task3-task4`; the original
cluster checkout and its edits were preserved.

| Item | Value |
| --- | --- |
| Device | NVIDIA H200 MIG 1g.18gb, 16 SMs, sm_90 |
| Allocation | One GPU, two CPU cores, 16 GB host memory |
| Driver / nvcc | 595.91.07 / 12.8.93 |
| Compute Sanitizer | 2025.1.0.0 |
| Python / Torch | 3.11.18 / 2.10.0+cu128 |
| Transformers / NumPy | 5.0.0 / 2.4.6 |
| Safetensors / Hugging Face Hub | 0.8.0 / 1.33.0 |
| Base commit | ae01d2939b2bae74c9ee3adcf0d82d05963e7d4d |
| Model snapshot | 61ba4e0b3309b6656edea3e93e419f7bd5c61957 |

Environment checks recorded installed packages and actual import paths;
`pip check` passed. Standard flags are `-O3 -std=c++17 -lineinfo -diag-suppress 177`.
The GPU-name query also reports the physical H200's memory; the allocated device
is the **1g.18gb MIG slice**, not the whole GPU.

```text
task3_gemm.cu       e235a0882bcd4fd134e7bc5061b6e3f95f46ab9cd9bf606cfa10fed3636e204a
task4_attention.cu  4800be3f1584e800bef2d73bee349873700b26ba97bad3ceeb75897240ca8745
```

The job completed successfully (`COMPLETED`, exit `0:0`) in **8m20s** and
released its GPU allocation. The final source-integrity check passed.

## Correctness and sanitizer coverage

All 33 official cases passed: four RoPE, eleven reductions, ten GEMM and eight
attention cases. Task 2 helper checks also passed. GEMM uses the original
`atol=rtol=1e-3`; attention uses `atol=1e-4, rtol=1e-3`.
The largest absolute errors in the default GEMM and attention suites were
`3.72e-5` and `5.21e-7`, respectively.

Five additional cases reuse the supplied CPU references: GELU without bias,
an exact output tile with partial K, 96-dimensional GQA with a spare warp,
96-dimensional shifted causal attention, and single-key batched GQA. All passed;
maximum error was `3.84e-7`, and the single-key case was exact.

Memcheck, racecheck and synccheck passed on GEMM's three small cases,
attention's seven small cases, and all five added cases. Large GEMM shapes and
the audio attention case received ordinary correctness checks, not sanitizer
runs. The supplied attention reference samples approximately 300 query rows in
large cases (including the first and last); this is not exhaustive verification
of every output. Official tests were not changed to expand these checks.

## Model integration

All four modes produced the expected transcription and supplied word accuracy
of 100%: "Concord returned to its place amidst the tents."

| CUDA groups | Mean transcription time | Time/token | Peak allocated GPU memory |
| --- | ---: | ---: | ---: |
| none (framework reference) | 1331.9 +/- 0.3 ms | 102.5 ms | 8.90 GiB |
| task1,task2 | 1294.7 +/- 0.5 ms | 99.6 ms | 8.90 GiB |
| task1,task2,task3 | 5144.9 +/- 0.1 ms | 395.8 ms | 8.90 GiB |
| all | 5672.1 +/- 0.4 ms | 436.3 ms | 8.56 GiB |

Each mode uses one warm-up and three timed runs, excluding model loading,
followed by a separate instrumented profile. All-CUDA time is approximately
4.26 times the framework baseline. Peak allocated memory is 0.34 GiB lower.
These are single-sample/device measurements, not general ASR quality or speed
claims. Modes ran in the listed order, so order effects are not excluded.

In the separate all-CUDA profile, 2,757 linear calls take 4960.0 ms and 396
attention calls take 688.6 ms. For task1/2/3 with framework attention, they take
4959.1 and 160.6 ms. Thus attention integration adds approximately 527.2 ms to
the observed mean; the linear kernel is still the largest cost. Profile totals
are instrumented and must not be substituted for unprofiled timing means.

## Controlled parameter experiments

All ten configurations passed every applicable official case in all three
rounds (264 case/round results). Median times below are in milliseconds.

| GEMM configuration | Decoder q_proj | Encoder q_proj | Encoder fc1 + GELU | lm_head |
| --- | ---: | ---: | ---: | ---: |
| TILE=8 | 0.915 | 8.109 | 32.457 | 2.910 |
| TILE=16, retained | 0.611 | 5.255 | 21.101 | 4.060 |
| TILE=32 | 0.518 | 4.460 | 18.372 | 7.138 |
| TILE=16, no padding | 1.397 | 12.473 | 49.998 | 9.803 |

The larger tile helps dense shapes but hurts the single-row vocabulary case:
31 of its 32 thread rows have no output. Removing padding increases encoder
fc1 time by about 2.37 times, consistent with shared-memory bank conflicts.
The default fc1 throughput is 932 GFLOP/s, versus approximately 110 for naive
GEMM and 6182 for cuBLAS in the supplied comparisons. Default lm_head achieves
about 119.6 GB/s of weights-only effective bandwidth.

| Attention configuration | Audio encoder | Text decoder |
| --- | ---: | ---: |
| 1 warp | 126.120 | 0.547 |
| 2 warps | 40.930 | 0.167 |
| 4 warps, retained | 20.712 | 0.062 |
| 8 warps | 17.044 | 0.044 |
| 16 warps | 17.081 | 0.041 |
| 4 warps, no K padding | 106.063 | 0.220 |

Eight/sixteen warps improve these large shapes; their audio times are nearly
equal. Removing K padding makes the default audio case about 5.12 times slower.
Without padding at D=64, lanes reading a fixed dimension of different key rows
access the same bank. Padding spreads those requests across banks.

We retain the validated default settings as the coursework baseline. The
GEMM sweep demonstrates shape trade-offs, and the promising attention variants
have not received model-level/sanitizer validation. These results identify
tuning opportunities rather than establish default settings as optimal.

Each configuration runs three complete correctness/timing rounds; the second
round reverses order. Tables use medians of the printed per-run values. Kernel
timing normally averages 20 calls after three warm-ups; the supplied naive GEMM
reference uses five calls after one warm-up. Source copies are isolated from
the production files, and variants are not used for model inference.

The supplied naive and cuBLAS times cover matrix multiplication alone, whereas
our kernel includes bias/GELU when requested. The GFLOP/s numerator `2*M*N*K`
also excludes epilogue arithmetic. cuBLAS uses pedantic FP32, without TF32.
`lm_head` has no epilogue, so its compared operations match. Its weight matrix
contains 485,490,688 bytes; weights-only effective bandwidth is
`485.490688 / time_ms` GB/s, not measured physical DRAM traffic.

## Report discussion: Task 3

For M=N=K=1024, the naive kernel requests `2*M*N*K = 2,147,483,648` floats
(8 GiB). Default tiled GEMM requests `2*M*N*K/16 = 134,217,728` floats
(512 MiB), sixteen times fewer, ignoring caches and output writes.
Reuse alone does not reproduce cuBLAS's register tiling, instruction scheduling
and device/shape-specific tuning.

Encoder fc1 produces `1500*5120*4 = 30,720,000` bytes. Fusing GELU saves the
separate activation's read and write, **61,440,000 bytes**, plus its launch.
The GEMM output write still occurs. Both tile barriers remain necessary: the
first publishes every tile load; the second prevents overwrite while another
thread still reads. A thread outside y can still provide an x or W element
needed by a valid output, so an early return would be incorrect.

## Report discussion: Task 4

One encoder layer's hypothetical full score tensor contains
`20*1500*1500*4 = 180,000,000` bytes (171.66 MiB). Our per-block shared storage
is `4*[32*(D+1)+32*D+4*D] = 272*D+128` bytes: **17,536 bytes** for D=64 and
**34,944 bytes** for D=128. This compares one score tensor with one block's
scratch space, not total memory across all resident blocks or the full model.

All warps must use the same block tile-loop bound. Stopping at individual
`last_key` values can mismatch barriers and omit required cooperative loads.
If a finite maximum does not change, `corr=1`; the first finite tile has
`corr=0`. An unused warp can have `m_new=-INFINITY`, so the empty summary keeps
`corr=1`, `p=0` and zero accumulators without evaluating `-inf - -inf`.

With four queries per block, the noncausal encoder needs `1500/4=375` blocks
per head. Each key and value vector is logically loaded 375 times per head per
layer, before cache effects. Increasing warps reduces repeated loads but raises
block resource use and reduces block-level scheduling opportunities.
The fixed four-element accumulator permits constant indexing and compiler
scalarization; register placement is not guaranteed under resource pressure.

## Reproduction, reuse and assistance

Follow [cluster reproduction instructions](../hw1-asr/docs/task34-validation.md).
The [raw evidence directory](task34-evidence/2026-11-09-job-3707553/) contains
builds, correctness checks, sanitizer reports, profiles, timing rounds,
environment/import paths, source hashes and Slurm accounting.
See the [independent review record](task34-review.md) for review scope,
findings and the resolved evidence-packaging check.

We reused course code, references, GELU and Task 2 helpers. The
[CUDA 12.8 documentation](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html),
[Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html),
[FlashAttention paper](https://arxiv.org/abs/2205.14135),
[CUTLASS](https://github.com/NVIDIA/cutlass) and
[FlashAttention implementation](https://github.com/Dao-AILab/flash-attention)
were consulted. The external kernels are mature alternatives, but using them
for the task computation would replace the custom CUDA work required by this
course. No external kernel or new dependency was incorporated.

AI assistance was used for implementation, validation orchestration, analysis,
documentation and independent code review. Include this in the final report's
required AI-use declaration. These notes provide Task 3/4 report material and
full-ASR validation; they are not the final six-page coursework report or RAG
systems investigation.
