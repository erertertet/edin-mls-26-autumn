# Task 1 RoPE: cluster validation and report notes

Task 1 passes the supplied reference tests and model transcription check. The
retained implementation uses 256 threads per block and standard `cosf`/`sinf`.
All compilation, CUDA tests and model execution took place on the Edinburgh
cluster. No local GPU was used. This document records Task 1 only.

## Implementation

[`task1_rope.cu`](../hw1-asr/csrc/task1_rope.cu) fills the four supplied blanks:

- Assign one thread to each pair and return when its index exceeds the work size.
- Process Q and K in one launch; switch the pointer, head count and relative index
  for threads assigned to K.
- Recover `row = idx / half`, `i = idx % half` and
  `pos = (row / num_heads) % seq_len`. The final modulo resets positions per batch.
- Read both original values before rotating `(v[i], v[i + half])` in place.

Each pair has one owner, so no shared memory or synchronization is needed.
Dimensions outside `rotary_dim` are untouched. Supplied interfaces and tests are
unchanged; Tasks 2–4 remain unimplemented.

## Environment and evidence

| Item | Recorded value |
| --- | --- |
| Date | 2026-11-09 UTC |
| Slurm job / node | `3707421` / `saxa.inf.ed.ac.uk` |
| Completion | `COMPLETED`, exit `0:0`, elapsed `00:05:19` |
| Allocation | 1 GPU, 2 CPU cores, 16 GB host memory |
| CUDA device | NVIDIA H200 **MIG 1g.18gb**, 16 SMs, `sm_90` |
| Driver / nvcc | `595.91.07` / `12.8.93` |
| Python / PyTorch | `3.11.18` / `2.10.0+cu128` |
| Transformers / NumPy | `5.0.0` / `2.4.6` |
| Safetensors / Hugging Face Hub | `0.8.0` / `1.33.0` |
| Repository base | `499be3760a5871e10ec967b97d44f362257283de` |
| Model snapshot | `61ba4e0b3309b6656edea3e93e419f7bd5c61957` |

Kernel SHA-256:
`f5c63b00b606ac5d17583033ae27085bcbc2f29f30c6bfcde5be8892a9b6d0a9`.

Raw logs and source checksums are in
[`task1-evidence/2026-11-09-job-3707421/`](task1-evidence/2026-11-09-job-3707421/).
The source-integrity check passed after every experiment had finished. The GPU
allocation was released when the job completed.

## Correctness and integration

The unmodified tests compare complete Q/K arrays with the supplied CPU reference,
using `abs(error) <= 1e-5 + 1e-4 * abs(reference)`.

| Official case | Result | Maximum absolute error |
| --- | --- | ---: |
| Tiny, multiple batches and unequal Q/K heads | PASS | 1.03e-7 |
| Odd sizes, partial rotation | PASS | 1.02e-7 |
| Audio encoder, 1500 positions, partial rotation | PASS | 1.91e-7 |
| Text decoder, GQA 16/4, full rotation | PASS | 1.73e-7 |

Compute Sanitizer memcheck reported **0 errors** on the three small cases.
The full audio case passed the ordinary test; it is intentionally excluded by
the supplied `small` sanitizer mode.

Both model modes transcribed the 3.50-second sample as:

> Concord returned to its place amidst the tents.

| Mode | Supplied word accuracy | Mean transcription time | Peak GPU memory |
| --- | ---: | ---: | ---: |
| `./run.sh --cuda none` | 100%, PASS | 1331.0 ms | 8.90 GiB |
| `./run.sh --cuda task1 --profile` | 100%, PASS | 1304.1 ms | 8.90 GiB |

Each mean uses three runs after one warm-up. Model loading is excluded.
Task 1 mode also enables the supplied `provided.cu` kernels, so the roughly 2%
end-to-end difference cannot be attributed solely to RoPE. This is one sample,
not an evaluation of general transcription quality. The separate profiling pass
recorded 396 CUDA RoPE calls, 5.1 ms total and 0.4% of profiled operation time;
profiling adds instrumentation and is distinct from the timing runs above.

## Block-size experiment

Every variant passed all four correctness cases. Each binary was run three times
on the same allocation; the second round reversed the block-size order. Each
test measurement uses three warm-up launches and 20 CUDA-event-timed launches.

| Threads per block | Audio median GB/s | Observed range GB/s |
| ---: | ---: | ---: |
| 32 | 208.7 | 208.6–209.1 |
| 64 | 289.9 | 289.9–290.2 |
| 128 | 290.0 | 289.5–290.6 |
| **256 (retained)** | **289.0** | **288.9–289.8** |
| 512 | 282.7 | 282.7–282.8 |
| 1024 | 277.0 | 276.6–277.4 |

The 64–256 results are within about 0.4%; these short measurements do not justify
replacing the default or claiming a universally optimal block size. One-warp
blocks can limit resident warps through the per-SM block limit, a plausible
explanation for the 32-thread slowdown; occupancy was not separately measured.
Tiny cases print approximately 0.002 ms across sizes, consistent with launch and
dispatch overhead dominating their small workload. Host/device copies are outside
the test's timed region, and the printed time has only microsecond resolution.

## Report question 1: bytes and bandwidth

For the audio case, `B=1`, `S=1500`, `Hq=Hk=20`, `rotary_dim=32` and float32
elements occupy four bytes. Each rotated value is read and written once:

```text
Q/K bytes = 2 × B × S × (Hq + Hk) × rotary_dim × sizeof(float)
          = 2 × 1 × 1500 × 40 × 32 × 4
          = 15,360,000 bytes = 15.36 MB per call
```

The retained 256-thread variant achieved median effective bandwidth 289.0 GB/s.
Relative to the supplied test's approximately 563 GB/s peak estimate for the
assigned MIG device, this is `289.0 / 563 ≈ 51.3%`. The first acceptance run
reported 289.2 GB/s and 0.053 ms, consistent with the repeated measurements.

This follows the test's useful-Q/K-byte accounting, which excludes `inv_freq`
loads and does not measure physical DRAM transactions or cache traffic. Use the
MIG device's estimate rather than the full H200 specification. NVIDIA documents
the [1g.18gb profile](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html)
as a partition of the GPU's memory and compute resources.

## Fast-math experiment

A separate build used `-O3 -std=c++17 --use_fast_math`. The audio case failed:
maximum error **1.94e-4**, with **52,077 / 3,840,000** checked values outside
tolerance. The other three cases passed. The audio case's maximum angle is
1499 radians, compared with 63 for the decoder and 2/6 for the small cases.

CUDA documents that this flag substitutes less accurate intrinsic implementations
for functions including `sinf` and `cosf`; the larger-angle audio case is consistent
with that accuracy trade-off. See the
[CUDA 12.8 mathematical-functions documentation](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#intrinsic-functions).
The experimental failure is preserved in the logs. Fast math was not selected,
and its separate binaries cannot overwrite the validated standard-math build.

## Reproduce and acknowledge assistance

Follow [`task1-validation.md`](../hw1-asr/docs/task1-validation.md) and submit
`sbatch run_task1.sbatch` from `hw1-asr/` on the cluster. The script reuses the
course Makefile, reference tests and model runner; it adds no package dependencies.
`setup_env.sh` was reused for the course environment. Kernel arithmetic is custom
CUDA as required by the coursework, using NVIDIA's verified math APIs rather
than an external RoPE implementation.

Codex assisted with the kernel implementation, validation script, cluster setup,
execution, analysis and these notes. Carry this assistance into the final
report's required AI-use declaration after reviewing the material.
