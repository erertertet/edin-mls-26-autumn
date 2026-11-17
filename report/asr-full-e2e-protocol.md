# Full 475-audio end-to-end ASR kernel validation

Extend the separate single-fixture check to **all 475 supplied recordings**, using the existing frozen audio selection (350 practice, 125 dev). The fixture `test_audio.wav` is historical evidence and is not added as a 476th query. Run fresh **all-PyTorch (`none`) and all-CUDA (`all`) GLM-ASR-Nano** inference on every query, with the same model snapshot, prepared input and IEEE float32 settings. This is a kernel-implementation comparison; it does not rerun the external four-model ASR experiment, the RAG encoder or retrieval.

## Two estimands

1. **Complete reference-history numerical discrepancy.** At every PyTorch greedy prediction position, compare all 59,264 vocabulary logits and probabilities against CUDA conditioned on the same reference text-token history. Retain CUDA's own audio projection, so encoder discrepancy remains part of the end-to-end comparison. Reuse CUDA's independent greedy trace while histories match, including the first differing prediction. Only for divergent queries, feed reference token IDs and recompute the missing CUDA tail with the existing `next_token_probs` and embedding operations, through reference EOS or the frozen 100-token limit. No model layer is reimplemented. Report full coverage explicitly.
2. **Independent greedy output disagreement.** Compare both complete greedy token-ID sequences, including EOS, and both decoded texts. Do not compare post-divergence independent-greedy logits as same-history arithmetic error. Report common-prefix coverage separately from full reference-history coverage.

The first query of each fixed shard additionally replays the entire reference history. Where its greedy and reference histories coincide, this checks replay equivalence against the saved natural CUDA trace. Preserve control results and replay-call counts. Do not introduce a post-hoc kernel acceptance threshold.

## Fixed execution and inputs

`asr-full-e2e-plan.json` fixes the model, 100-token ceiling, four shards, precision and metric conventions before full inference. Reuse the original 475-query selection and split membership, with deterministic shard membership `original_index % 4`. Keep all six earlier cohorts; they overlap and are not independent replications. Do not replace failed queries, exclude long clips, drop empty decoded strings or discard sequences that hit the token cap.

Reuse snapshot `61ba4e0b3309b6656edea3e93e419f7bd5c61957` and the **exact previously audited sm_90 library** with SHA256 `76817bd7fef1878069515d12eb4156f76368c21e12076a00ec450c1a7c78da8a`. Its kernel sources are unchanged. This avoids unnecessary rebuilding of identical kernels. Each worker loads the model once; each query has one processor preparation shared by both modes. Reuse `asr_worker.read_audio` for validated 16 kHz mono PCM16 inputs. Use only PyTorch 2.10's new `fp32_precision='ieee'` controls globally and for CUDA matmul/cuDNN; do not mix old TF32 flags.

Stage one shared read-only checkpoint/audio cache on the allocated saxa node, verify all hashes once, and bind its proof to the global configuration. Record actual device, versions, import paths, source hashes and library bytes for each worker. Run four GPU array shards on H200 MIG 1g.18gb slices (sm_90, 16 SMs), within Slurm's resource limits. Work is resumable only when a completed query's configuration, input and trace hashes match. Persist each query atomically and release its traces from memory before the next. Preserve failures; aggregate only after all 475 pairs are complete.

## Numerical metrics

Preserve complete float32 traces privately, without vocabulary sampling or lossy conversion. Reconstruct each CUDA reference-history array from its valid greedy prefix and stored replay tail. Compute discrepancies in float64, keeping per-query `elements`, sum absolute differences, sum squared differences and sum squared reference values.

For each logits/probabilities field report:

- empirical maximum absolute discrepancy across the observed corpus;
- mean, median and 95th percentile of per-query maximum discrepancies;
- macro means of query MAE, RMS and relative L2;
- pooled MAE `sum(abs_diff)/sum(elements)`, pooled RMS `sqrt(sum(squared_diff)/sum(elements))`, and pooled relative L2 `sqrt(sum(squared_diff)/sum(squared_reference))`.

Label macro and pooled quantities distinctly. A zero reference norm gives relative error zero only when the difference is also zero; otherwise it is undefined. Preserve non-finite/shape failures rather than assigning zero error. Report projected-audio discrepancies as an additional diagnostic when retained. Observed maxima are finite-dataset measurements, not bounds on future inputs.

## Token and transcript agreement

Report complete token-sequence exact-match queries, complete decoded-text matches, token lengths, first divergences, EOS/no-EOS and cap counts. Token positional identity counts equal IDs at the same index and uses the larger sequence length as denominator, including unmatched suffix positions. Separately report CUDA argmax agreement along the full PyTorch-reference history. These are different quantities.

Use RapidFuzz 3.14.6 unit-cost Levenshtein over token-ID sequences, Unicode code points and whitespace-word sequences. For decoded text use `tokenizer.decode(..., skip_special_tokens=True).strip()`, matching the existing adapter, with no further case, punctuation or acronym normalization. Report absolute edit counts, macro mean `distance/max(lengths)` and pooled `sum(distance)/sum(max(lengths))`, with two empty sequences giving zero distance. Keep zero-distance queries and empty outputs. These reference-free implementation differences are **not WER/CER against a human transcript**.

## Evidence, reuse and reporting

The actual requirements, installed versions and import paths were checked. Reuse Python 3.11.18, PyTorch 2.10.0+cu128, Transformers 5.0.0, Safetensors 0.8.0 and NumPy 2.4.6, plus existing RapidFuzz 3.14.6 in the separate CPU-analysis venv. Official target-version PyTorch, NumPy compressed-array and RapidFuzz sequence-distance documentation were consulted. No new inference dependency, distance algorithm or replacement model implementation is introduced.

An independent CPU audit reads every query's lossless traces, reconstructs the reference-history comparison, checks capture/argmax/termination and hash accounting, and recomputes numerical sufficient statistics and edit distances. Publish aggregate results only. Full tokens, texts, arrays and query mappings remain under ignored results. Preserve the old single-fixture and four-ASR results separately. Rebuild the main report, keep its six-page main-body limit, and visually verify the updated PDF before pushing final results to PR5.

References: [PyTorch 2.10 precision controls](https://docs.pytorch.org/docs/2.10/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices), [NumPy 2.4 compressed arrays](https://numpy.org/doc/2.4/reference/generated/numpy.savez_compressed.html), [RapidFuzz Levenshtein sequences](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html), [single-fixture results](asr-e2e-error-results.md).
