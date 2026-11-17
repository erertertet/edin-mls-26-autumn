# Full 475-query ASR kernel numerical and output comparison

This completed experiment compares **all custom CUDA tasks** with the same GLM-ASR-Nano model's **all-PyTorch reference** on all **475** supplied query recordings. Every planned query is retained; the earlier single-clip fixture is separate and is not a 476th query. No external ASR/API or RAG-encoder calls are made.

## Observed results

- Full independent-greedy token-ID sequences agree for **475/475 (100.000%)** queries, including EOS and length differences.
- Decoded transcripts agree for **475/475 (100.000%)** queries under the frozen decoder-output definition.
- Complete PyTorch-reference-history numerical coverage is **7660/7660 positions (100.000%)**; independent greedy shared-history coverage is 100.000%.
- The empirical maximum absolute logit discrepancy is **3.749847e-03** over 453962240 full-vocabulary values.
- The empirical maximum absolute probability discrepancy is **2.660751e-04** over 453962240 full-vocabulary values.

These are exact summaries of the observed finite set, not confidence bounds or a population error guarantee. No bootstrap intervals, post-hoc numerical acceptance threshold, WER/CER estimate or downstream retrieval claim is introduced.

## Numerical discrepancy on the full reference history

For every query, the PyTorch greedy sequence defines the text-token history. CUDA retains its own audio projection. Scores are reused while independent histories match; after a divergence, CUDA is replayed on the remaining reference text-token history. A candidate EOS does not terminate this fixed-reference-history scoring early. This covers the full reference trajectory without treating later scores from two different greedy histories as same-history numerical error.

All **59,264** vocabulary entries at every reference prediction position are included for logits and probabilities. Statistics use float64 calculations on saved float32 values. For candidate `x` and reference `y`, absolute discrepancy is `abs(x-y)`, RMS is `sqrt(mean((x-y)**2))`, and relative L2 is `norm(x-y)/norm(y)`. A zero reference norm yields zero relative L2 only when the difference norm is also zero; otherwise relative L2 is undefined.

The empirical maximum is the maximum of the query maxima. Other query-max summaries and macro MAE/RMS/relative-L2 give each included query equal weight. Macro relative L2 averages only queries for which it is defined; their count is reported below. The p95 uses NumPy's `method='linear'`.

| Output | Empirical max | Mean query max | Median query max | p95 query max | Macro MAE | Macro RMS | Macro relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logits | 3.749847e-03 | 1.514454e-04 | 7.820129e-05 | 4.722834e-04 | 1.298071e-05 | 1.727460e-05 | 3.323301e-06 |
| probs | 2.660751e-04 | 5.184132e-06 | 3.576279e-06 | 1.134798e-05 | 2.008838e-11 | 7.866162e-09 | 1.981386e-06 |
| audio_embeddings | 1.604426e-02 | 2.807103e-04 | 1.049638e-04 | 8.784294e-04 | 2.501649e-06 | 8.278658e-06 | 1.415900e-05 |

Pooled MAE is `sum(abs(diff))/sum(elements)`; pooled RMS is `sqrt(sum(diff**2)/sum(elements))`; pooled relative L2 is `sqrt(sum(diff**2)/sum(reference**2))`. These ratios are not averages of per-query ratios.

| Output | Values | Pooled MAE | Pooled RMS | Pooled relative L2 | Defined query relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| logits | 453962240 | 1.296695e-05 | 3.607343e-05 | 6.846018e-06 | 475/475 |
| probs | 453962240 | 2.001955e-11 | 1.898890e-08 | 4.753525e-06 | 475/475 |
| audio_embeddings | 63639552 | 2.381772e-06 | 2.209664e-05 | 3.774078e-05 | 475/475 |

## Output agreement and coverage by cohort

Exact token sequence agreement uses complete token IDs, including EOS. Positional token identity counts equal tokens at the same index and divides by the sum of the larger sequence lengths; unmatched length positions count as disagreements. It is not edit distance: an insertion can shift many subsequent positions. Reference-history argmax identity instead compares CUDA's argmax with the reference token at each fixed-reference-history step.

| Cohort | Queries | Exact tokens | Exact decoded text | Positional token identity | Reference-history argmax identity | Greedy shared-history coverage | Full-reference coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 475 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| practice | 350 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| dev | 125 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| pilot | 8 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| remaining_dev | 117 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| remaining_all | 467 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |

Decoded text is `processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()`. The separate unstripped decoded-string exact-agreement count is 475/475. No acronym, case, punctuation or other transcript-format normalization is applied in the edit-distance comparison.

## Token and transcript edit distances

Unit-cost Levenshtein insertion, deletion and substitution distances use complete token-ID sequences (including EOS), Unicode code points of the decoded output, or `text.split()` word sequences. Word splitting retains case and punctuation but ignores pure whitespace differences. The mature [RapidFuzz Levenshtein API](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html) is pinned to [version 3.14.6](https://pypi.org/project/RapidFuzz/3.14.6/), with unit weights and no processor.

For each query, length-normalized edit distance is `distance/max(lengths)`; both empty gives zero. Macro NED averages these per-query ratios. Pooled NED is `sum(edits)/sum(max(lengths))`, a distinct length-weighted aggregate; its all-empty convention is zero. Here 'normalized' refers only to length scaling, not text normalization. Empty transcripts are retained.

| Cohort | Unit | Mean edits | Macro NED (%) | Pooled NED (%) | Zero-distance queries |
| --- | --- | ---: | ---: | ---: | ---: |
| all | tokens | 0.000000 | 0.000000 | 0.000000 | 475/475 |
| all | characters | 0.000000 | 0.000000 | 0.000000 | 475/475 |
| all | words | 0.000000 | 0.000000 | 0.000000 | 475/475 |
| practice | tokens | 0.000000 | 0.000000 | 0.000000 | 350/350 |
| practice | characters | 0.000000 | 0.000000 | 0.000000 | 350/350 |
| practice | words | 0.000000 | 0.000000 | 0.000000 | 350/350 |
| dev | tokens | 0.000000 | 0.000000 | 0.000000 | 125/125 |
| dev | characters | 0.000000 | 0.000000 | 0.000000 | 125/125 |
| dev | words | 0.000000 | 0.000000 | 0.000000 | 125/125 |
| pilot | tokens | 0.000000 | 0.000000 | 0.000000 | 8/8 |
| pilot | characters | 0.000000 | 0.000000 | 0.000000 | 8/8 |
| pilot | words | 0.000000 | 0.000000 | 0.000000 | 8/8 |
| remaining_dev | tokens | 0.000000 | 0.000000 | 0.000000 | 117/117 |
| remaining_dev | characters | 0.000000 | 0.000000 | 0.000000 | 117/117 |
| remaining_dev | words | 0.000000 | 0.000000 | 0.000000 | 117/117 |
| remaining_all | tokens | 0.000000 | 0.000000 | 0.000000 | 467/467 |
| remaining_all | characters | 0.000000 | 0.000000 | 0.000000 | 467/467 |
| remaining_all | words | 0.000000 | 0.000000 | 0.000000 | 467/467 |

These compare two implementations of one model without independent reference transcripts. They measure implementation/output disagreement, not WER/CER, recognition accuracy or semantic correctness. Exact agreement can preserve a common recognition error.

## Termination, replay and finite-set scope

The frozen generation cap is 100 new tokens. EOS termination and reaching the cap are recorded separately; reaching the cap on an EOS token can occur in both categories.

| Cohort | Reference EOS | CUDA EOS | Reference at cap | CUDA at cap | Reference cap without EOS | CUDA cap without EOS | Reference empty text | CUDA empty text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 475 | 475 | 0 | 0 | 0 | 0 | 0 | 0 |
| practice | 350 | 350 | 0 | 0 | 0 | 0 | 0 | 0 |
| dev | 125 | 125 | 0 | 0 | 0 | 0 | 0 | 0 |
| pilot | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| remaining_dev | 117 | 117 | 0 | 0 | 0 | 0 | 0 | 0 |
| remaining_all | 467 | 467 | 0 | 0 | 0 | 0 | 0 | 0 |

Reference-history replay is used for 4 queries, with 51 stored replay rows and 51 replay next-token calls. These counts include the four predeclared first-query shard controls; they are not simply the number of divergent queries. The controls compare full replay with the natural CUDA common-prefix scores: maximum measured logit/probability discrepancies are 0.000000e+00/0.000000e+00.

All 475 supplied queries have been examined in earlier work. These are descriptive finite-set results, with no sampling or skipped-query extrapolation. The six cohorts overlap: all = practice 350 + dev 125; dev = pilot 8 + remaining dev 117; remaining all 467 excludes the pilot. They are not independent replications or a new confirmatory holdout.

## Precision, audit and provenance

Global, CUDA-matmul and cuDNN FP32 controls are `ieee`, under the pinned model/kernel/source configuration. These controls do not guarantee that different implementations yield identical floating-point values. The [PyTorch 2.10 numerical-accuracy notes](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html) describe operation-order and platform effects. This instrumented experiment supplies no new latency, peak-memory or speed comparison and does not replace the historical timing ablation.

- Summary Slurm job: `3728059`.
- GPU evidence jobs: `3728009`, `3728058`, `3728060`, `3728064`.
- Independent saved-array audit job: `3728059`; status **pass**.
- The audit binds all 475 query records and trace files and all six aggregate cohorts. It recomputes metrics and distances without rerunning the model, encoder or external APIs.
- Summarizer versions: NumPy `2.4.6`, RapidFuzz `3.14.6`.
- Model snapshot: `61ba4e0b3309b6656edea3e93e419f7bd5c61957`.

The completed query evidence contains 950 local model-generation calls and 950 audio-encoder calls, plus the separately counted reference-history replay calls. Recorded runtime job counters include 950 model-generation and 950 audio-encoder calls. Interrupted jobs can have unpersisted attempts, so runtime counters are not a claim to reconstruct every attempted call. Summarization and auditing make no model, encoder or external API calls.

No numerical acceptance tolerance was fitted after observing errors. The model is the reference implementation, not an independent ground-truth recognizer. The original single-clip result and the four-ASR normalization/retrieval experiments remain separate evidence.

| Bound input or source | SHA256 |
| --- | --- |
| `analysis-config.json` | `9c339ad3fe0343663b1a326372386f0b5dfeedd557e7a70472d419628b01c568` |
| `config.json` | `82f6e7d84487e05814b6519c2b4db10c8b5997bcd7c39e34c9a260073884d724` |
| `jobs/3728009.json` | `b048faf16c8518e9858a5350559c3d055658f7443241fefe392ff6f17416bbc1` |
| `jobs/3728058.json` | `8051d71d5db05222468e497bc46312c84ada1dcefc06ff00ddd98145f20923e6` |
| `jobs/3728060.json` | `45186d0b6a2c1aa1bea35b757f64606f4bf0109c449ff65af74c1f2f06513a18` |
| `jobs/3728064.json` | `00a75f4eb6029aeaea5fdd55738b7c5ab2af27730a72b72be04a69b788d223a0` |
| `library-build.json` | `d87e1052348ce00d3d47d35b9da7f58dc77e64d8f6cee9e50c516c6366f86d7c` |
| `shards/0.json` | `bc7426df5a04ecde7f0afbdebc0f45fc11c5aea688915eaf6510dbe657e7b7b5` |
| `shards/1.json` | `6700a05614f72af4b121bc1c0dd43d852b58d4667a01f54d695f64d647706dbd` |
| `shards/2.json` | `924ad9922e512f46222d815c8d91c0df70033a9cc41c3aa0ef8badda8bdde30a` |
| `shards/3.json` | `bfadf230d89e61b013d30f8b34511cfcafb5d591948fef7f6c0c9dc27c3aabe1` |
| `staging-proof.json` | `83bed36fd16f489eca898e52626087a012e28ce4e22c456a63a8955a12517a79` |

| Frozen inference source | SHA256 |
| --- | --- |
| `hw1-asr/Makefile` | `42941e1e0fb40a46d8cfd6e33f9fff60d26a44ba70e3291bc07f4d43cd4ffaff` |
| `hw1-asr/asr-full-e2e-plan.json` | `7a733d1c594c85aea91b9c3785993c8ecd8ee8958e5d0a4885066ce47f01a0c6` |
| `hw1-asr/benchmark.py` | `be6516f1c626eb50761c260788ee0561a73043d3775fb983bea5c803089cbecb` |
| `hw1-asr/build_asr_full_e2e_library.py` | `1fccb3776ec9d19f64b7c5150ffde24d72ee94bf9d92bcd9bd34030d896b1f87` |
| `hw1-asr/csrc/asr_kernels.h` | `41ab732721576e45e9463dfc608ced9ecdda4796702db462ae1d2dfeb55f127f` |
| `hw1-asr/csrc/common.cuh` | `adddccbeba9058d2b2904fc5bd172e5d02595e3d1cd2c9600153049139ffe4ce` |
| `hw1-asr/csrc/provided.cu` | `d9084f2898b953dedaede136b77d4b32f45b609b7e84eb07cb78946b9cd060d2` |
| `hw1-asr/csrc/task1_rope.cu` | `f5c63b00b606ac5d17583033ae27085bcbc2f29f30c6bfcde5be8892a9b6d0a9` |
| `hw1-asr/csrc/task2_reduce.cu` | `e6abcaeac38a642114626227965ce34519c2d16e25b017f4ccfc16717349b796` |
| `hw1-asr/csrc/task2_reduce.cuh` | `29dc47f65038539c76c2ba54dbb1831f5e29f1f5008ab6168759e6c14534bfdb` |
| `hw1-asr/csrc/task3_gemm.cu` | `e235a0882bcd4fd134e7bc5061b6e3f95f46ab9cd9bf606cfa10fed3636e204a` |
| `hw1-asr/csrc/task4_attention.cu` | `4800be3f1584e800bef2d73bee349873700b26ba97bad3ceeb75897240ca8745` |
| `hw1-asr/glm_asr_cuda/__init__.py` | `608b62885afb583e58bcca58901fdff22179f3d9d5caf928f9658cd2d8fd5eeb` |
| `hw1-asr/glm_asr_cuda/model.py` | `c690ed8dbfcb6e9b775d1b2f6da72b0b055fbeb965fd838afaaa0f49bf008379` |
| `hw1-asr/glm_asr_cuda/ops.py` | `de224fee6464b5d3288297a7d0ea2eb66e9206d95bea59586349d2979290e906` |
| `hw1-asr/measure_asr_e2e_error.py` | `7cd81a77deb9ed869f500eb1353a23346be5d3e831abaa90fe53b32a233af9b3` |
| `hw1-asr/measure_asr_full_e2e.py` | `ef2960a6c58ae81cbf2926cd4dcd4734612ba526614e5eee9774a90333fb201e` |
| `hw1-asr/prepare_asr_full_e2e.py` | `43a36179ab563f6fb34faa8f39a31370523c9b8bc807949e5830e3ed5ce014b4` |
| `hw1-asr/run_asr_full_e2e.sbatch` | `eb904ae0e36349f13bf8b2114fe864391fe5b1b25747c750adccce528150ab76` |
| `hw1-asr/run_build_asr_full_e2e.sbatch` | `925c4603fb473014a4fde820cc45b972b12b6c179459365faad0b8212d68b73d` |
| `hw1-asr/run_prepare_asr_full_e2e.sbatch` | `bc017b12f7d61eae30b8830bb83743da1a63695458f51f3a1888a3e801c4f04c` |
| `rag/src/asr_worker.py` | `e2bec126f4e05d88a3f8913504b25733627e3a7b536bebe01aa05f46f7679f59` |
| `report/asr-full-e2e-protocol.md` | `ae2dd6d76792af8db72d54a7499ff481ea234353bdc7fbb9e14d64203027e7ad` |
| Loaded CUDA library | `f4914777a519c5a6029da698e6440c0e4067b3e5dcf0a1dc2b450319db26a008` |
| Aggregator source | `654decf992d288b3ea1d18327a5a45d2bb2ff7b41797ad12f56e8131c4cda02e` |

[All cohort aggregates as CSV](asr-full-e2e.csv) contain unrounded counts and metrics. This report, the aggregate summary and CSV contain no query IDs, transcripts or token sequences. Per-query records, traces, audio and model weights remain private. The [full-set protocol](asr-full-e2e-protocol.md) documents reproduction; the [single-clip result](asr-e2e-error-results.md) and [historical kernel note](task34-kernels-results.md) retain their original scope.
