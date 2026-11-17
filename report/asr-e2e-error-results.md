# End-to-end ASR kernel numerical discrepancy

The complete all-CUDA pipeline differs from the all-PyTorch reference by at most
**5.245209e-5 in logits** and **1.788139e-6 in probabilities** on the supplied
**one 3.505-second clip**. All **13 generated token IDs (including EOS)** and the final decoded
text are identical. Thus zero output disagreement coexists with nonzero
floating-point discrepancy; it does not establish WER=0 or a general ASR error
bound.

GPU Slurm job **3727116** completed in **9m28s** on 2026-11-17. Independent CPU
audit **3727120** passed in **9s**. These are allocation/job durations, not
transcription latency measurements. The historical timing and peak-memory
ablation in [the kernel result note](task34-kernels-results.md) is preserved
separately.

## Comparison and coverage

The experiment uses the existing `zai-org/GLM-ASR-Nano-2512` snapshot
`61ba4e0b3309b6656edea3e93e419f7bd5c61957`, the course float32 loader and original
greedy `GlmAsr.generate` implementation. One loaded model and one processor
preparation provide the same inputs to every mode. No model, kernel,
transcript-normalization rule or retrieval result is changed.

| Pipeline mode | Custom tasks | Aligned prediction positions | Vocabulary per position | Token mismatches versus PyTorch | Exact decoded-text match |
| --- | --- | ---: | ---: | ---: | --- |
| All PyTorch | none; reference | 13 | 59,264 | Reference | Reference |
| RoPE + reductions | task1,task2 | 13/13 | 59,264 | 0/13 | Yes |
| RoPE + reductions + GEMM | task1,task2,task3 | 13/13 | 59,264 | 0/13 | Yes |
| All CUDA | all four tasks | 13/13 | 59,264 | 0/13 | Yes |

As in the supplied adapter, partial CUDA modes also enable the supplied CUDA
kernels. We retain every pre-softmax final-vocabulary logit and post-softmax
probability at every generated position, plus projected-audio embeddings.
This compares **770,432 logits and 770,432 probabilities per candidate mode**,
not only the chosen tokens, top scores or sampled vocabulary entries. Each
audio projection contains **44 × 2,048 = 90,112 values**.

Numerical output comparisons require matching decoder-input token histories.
All token IDs match here, so all 13 positions are comparable. Under the frozen
protocol, the first differing prediction would still have a common history;
later divergent branches would not be described as same-history numerical
error. No such branch occurred in this run.

## Metrics

For flattened candidate values `x` and reference values `y`, the calculations
convert the saved arrays to float64 and use:

- Maximum absolute discrepancy: `max(abs(x-y))`.
- Mean absolute discrepancy: `mean(abs(x-y))`.
- RMS discrepancy: `sqrt(mean((x-y)**2))`.
- Relative L2 discrepancy: `norm(x-y) / norm(y)`, without an epsilon.

The relative L2 value is zero if both the reference norm and difference norm
are zero, and undefined otherwise when the reference norm is zero. Every
reference norm in these observed comparisons is nonzero. Reported values below
are rounded from the audited saved metrics. They measure discrepancy against
the framework computation, not mathematical exactness or transcription error.
No end-to-end numerical acceptance threshold was selected.

### Full-vocabulary logits

| Custom CUDA groups | Maximum absolute | Mean absolute | RMS | Relative L2 |
| --- | ---: | ---: | ---: | ---: |
| task1,task2 | 2.574921e-5 | 3.195843e-6 | 4.196805e-6 | 8.505339e-7 |
| task1,task2,task3 | 6.103516e-5 | 6.117645e-6 | 7.813694e-6 | 1.583541e-6 |
| all | 5.245209e-5 | 6.409950e-6 | 8.183610e-6 | 1.658509e-6 |

### Full-vocabulary probabilities

| Custom CUDA groups | Maximum absolute | Mean absolute | RMS | Relative L2 |
| --- | ---: | ---: | ---: | ---: |
| task1,task2 | 4.172325e-7 | 3.778965e-12 | 9.183031e-10 | 2.271151e-7 |
| task1,task2,task3 | 3.993511e-6 | 1.857601e-11 | 6.091016e-9 | 1.506432e-6 |
| all | 1.788139e-6 | 9.931626e-12 | 2.881914e-9 | 7.127561e-7 |

### Projected-audio embeddings

| Custom CUDA groups | Maximum absolute | Mean absolute | RMS | Relative L2 |
| --- | ---: | ---: | ---: | ---: |
| task1,task2 | 1.049042e-5 | 6.419642e-7 | 8.337358e-7 | 1.278535e-6 |
| task1,task2,task3 | 1.049042e-5 | 6.417831e-7 | 8.373432e-7 | 1.284067e-6 |
| all | 9.536743e-6 | 8.319603e-7 | 1.080925e-6 | 1.657600e-6 |

These integrated differences include downstream propagation through the model;
they are distinct from the older isolated GEMM/attention maximum errors.
Adding a kernel group need not monotonically increase or decrease any one
metric. These three operating configurations do not identify an independent
error contribution from each task.

## Repeat controls and probability diagnostics

There are six generation runs in this order: all PyTorch, task1+2,
task1+2+3, all CUDA, all PyTorch repeat, all CUDA repeat. Both endpoint modes
therefore receive one additional run with the same loaded model and inputs.

| Within-mode comparison | Maximum logit discrepancy | Maximum probability discrepancy | Maximum audio-projection discrepancy | Token mismatches |
| --- | ---: | ---: | ---: | ---: |
| PyTorch first versus repeat | 0 | 0 | 0 | 0/13 |
| All CUDA first versus repeat | 0 | 0 | 0 | 0/13 |

Mean absolute, RMS and relative L2 discrepancies are also zero in both repeat
comparisons, and decoded text is identical. This is an observed within-run
repeat control on one device, not a promise of bitwise reproducibility across
executions, architectures or software versions.

| First run of mode | Maximum absolute probability-mass deviation from 1 | Logit/probability argmax disagreements |
| --- | ---: | ---: |
| All PyTorch | 2.452480e-7 | 0/13 |
| task1,task2 | 2.117376e-7 | 0/13 |
| task1,task2,task3 | 2.843886e-7 | 0/13 |
| All CUDA | 2.299955e-7 | 0/13 |

The saved traces are finite. The auditor verifies that probability argmaxes
equal generated token IDs and that recorded chosen probabilities equal the
corresponding full-vocabulary entries.

## Numerical context, interpretation and limits

The actual PyTorch settings after configuration are:

```text
torch.backends.fp32_precision = "ieee"
torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.fp32_precision = "ieee"
```

Only the new precision controls are used; deprecated `allow_tf32` controls are
not mixed into this run. This requests internal FP32 computation for the
relevant framework backends. It does not imply that distinct floating-point
implementations must be bitwise identical. The target-version
[PyTorch 2.10 precision documentation](https://docs.pytorch.org/docs/2.10/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)
and [numerical-accuracy discussion](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)
describe these distinctions.

The new run is instrumented and uses an explicit precision context separate
from the historical uninstrumented timing study. It supplies no new latency,
peak-memory or speed comparison. The older `3.50 s` clip label is its rounded
duration; the audited audio has **56,080 samples at 16 kHz**, giving **3.505 s**.

The historical benchmark's printed `Accuracy=100%` is coverage of the set of
distinct expected words after uppercasing and punctuation removal. It ignores
order, repetition and extra words and is not WER. Its old logs preserve the
final timed run's decoded text, not complete token IDs for every timed run.
The new trace establishes zero token and exact-text disagreement against
PyTorch in this particular run; it does not establish recognition accuracy on
other audio, WER=0, semantic correctness, a corpus-wide error bound or a
downstream retrieval/answer benefit.

There is one model load, one processor preparation, **six local-cluster
`model.generate` calls** and **six audio-encoder calls**. External ASR calls,
external API calls and RAG-encoder calls are all **zero**. This is a local
kernel-integration experiment on the cluster, separate from the saved-data
four-model edit-distance study.

## Environment, audit and provenance

| Item | Recorded value |
| --- | --- |
| GPU / host | NVIDIA H200 MIG 1g.18gb, 16 SMs, `saxa.inf.ed.ac.uk` |
| Device-visible total memory | 17,179,869,184 bytes |
| CUDA capability / Torch CUDA | 9.0 / 12.8 |
| PyTorch / Transformers | 2.10.0+cu128 / 5.0.0 |
| NumPy / Safetensors / Hugging Face Hub | 2.4.6 / 0.8.0 / 1.33.0 |
| GPU analysis job / CPU audit job | 3727116 / 3727120 |
| Analysis Git commit | `99d15d9515cd8cca201f8cb551eecd013978675f` |
| Model snapshot | `61ba4e0b3309b6656edea3e93e419f7bd5c61957` |

The CPU audit independently recomputes **145 global/per-position metric
blocks**, including **65 aligned per-position comparisons**, across the three
candidate comparisons and two repeat controls. It checks **78 captured
prediction rows**, **4,622,592 logits** and the same number of probabilities,
plus audio projections, source/library/input integrity and token-history
alignment. It verifies **13 source-file hashes** and **10 model snapshot-file
hashes**. The audit does not rerun inference or independently decode tokens.
Its `1e-10` relative tolerance checks agreement of metric recomputation, not
acceptance of a kernel's discrepancy. Audit status is **pass**, with no errors
and no kernel numerical acceptance threshold.

| Artifact | SHA256 |
| --- | --- |
| Supplied audio | `103c3f15eb3715ebc6243d142244128a3ac39b6bfae315baa6b8dc4a8be14aa8` |
| Measurement script | `7cd81a77deb9ed869f500eb1353a23346be5d3e831abaa90fe53b32a233af9b3` |
| Frozen protocol | `000a0225dfcaa3422304c7f012ef2ebc7ac93fd9eca0e523384fcf4a4e3fb407` |
| Task 1 RoPE | `f5c63b00b606ac5d17583033ae27085bcbc2f29f30c6bfcde5be8892a9b6d0a9` |
| Task 2 reductions | `e6abcaeac38a642114626227965ce34519c2d16e25b017f4ccfc16717349b796` |
| Task 3 GEMM | `e235a0882bcd4fd134e7bc5061b6e3f95f46ab9cd9bf606cfa10fed3636e204a` |
| Task 4 attention | `4800be3f1584e800bef2d73bee349873700b26ba97bad3ceeb75897240ca8745` |
| Loaded/saved CUDA library | `76817bd7fef1878069515d12eb4156f76368c21e12076a00ec450c1a7c78da8a` |
| Configuration | `6f3f146d994805d8b4b82bb33198d74aa80fb84bd0407affc7bd0eb8317ae152` |
| Runs metadata | `07f1fb0b759bb1e1026e19fdb5bb9ae9f2be59ad7c2e2af6c1eafcf4a641c96d` |
| Numerical summary | `d4bd44908a87262c8d3da03dd68b935b2a70a4edf93ba3252deb695f8be260e7` |
| Completion manifest | `91b60fe89098000d6ee1ca3beb8dcafbacd441136c908125b4e7841ca4729a9d` |
| Independent auditor source | `e8d01a15a8ee7da4c588dbb1e484d814501ca484c009ed5c518ade4971415ea6` |
| Independent audit result | `67deb6112f7c464abda750eb3d3574308a0a56b05b1f33f720a9e20b7dd8c468` |

Reproduction uses [the frozen protocol](asr-e2e-error-protocol.md),
[measurement runner](../hw1-asr/measure_asr_e2e_error.py),
[GPU Slurm wrapper](../hw1-asr/run_asr_e2e_error.sbatch),
[saved-array auditor](../hw1-asr/audit_asr_e2e_error.py) and
[CPU Slurm wrapper](../hw1-asr/run_audit_asr_e2e_error.sbatch).
All coursework execution remains inside Edinburgh Slurm allocations. The
public result note contains aggregate discrepancies and hashed provenance;
the full numerical arrays, token IDs, decoded text and model weights remain
private and are excluded from the report source archive.
