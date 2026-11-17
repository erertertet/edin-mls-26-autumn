# Formatting-normalized ASR pilot — 17 November 2026

**The four models' retrieval differences disappear on these eight queries after the same formatting normalization.** Each query has one identical normalized string across GLM, Whisper, MAI and Voxtral. Independent encoding of all 32 normalized transcripts gives identical selected IDs, scores, recall and document-token counts across models for each policy.

This is a post-hoc diagnostic using the saved transcripts from [the raw pilot](asr-rag-four-model-pilot-results.md). It does not establish equal general ASR quality or that ASR choice is irrelevant.

## Fixed method and execution

- Rule `acronym-punctuation-v1` was committed before the rerun: [frozen protocol](asr-rag-normalization-protocol.md) and [normalizer](../rag/extensions/transcript_normalization.py).
- NFKC, canonical apostrophes, bounded initialism merging (`I S A/I.S.A./ISA`, `U.S./US`), casefold, punctuation/hyphen-to-space and whitespace folding. Word-internal apostrophes, numeric separators/signs/percent suffixes and bracketed content are retained. No word, number, spelling, synonym or domain-specific expansion is applied.
- Same 32 successful saved transcripts, eight audio identities, original qrels, `enc_a` CPU behavior, 24,000 corpus vectors, full float32 exact cosine ranking and stable corpus-row tie break. No query replacement or parameter retuning.
- Fixed top-5 and all scores >=0.8925 with top-1 fallback and no cap. The corpus and record texts are unchanged; normalization affects query transcripts only.
- Same pinned Mistral tokenizer as the raw pilot, revision `a11f36bebf709121056b1dbcc943d1c6afbe494d`. Context tokens sum individual retrieved document texts without special tokens, questions or prompt framing.
- Source commit: `af608f6f4feb688f325da0ffe6496f48d819d4e9`, clean checkout. CPU-only Slurm job **3724575**, node **saxa**, four CPUs/12 GB, completed in **8m32s** including shared-filesystem reads and staging. No GPU allocation, ASR inference, API calls, answer generation or judging was used. New API cost: **USD 0**.
- Python 3.11.18, Unicode data 14.0.0, NumPy 2.4.6, sentence-transformers 6.3.0, transformers 5.0.0, torch 2.10.0+cu128 and tokenizers 0.22.2. Existing encoder dtype behavior is retained, including the float16-on-CPU warning.
- The recomputed **64 raw-control cells exactly match** the original selected IDs, counts, hits, recalls, tokens and fallback flags. Maximum selected-score drift is **0.0**, below the predeclared 1e-6 gate.

## Raw versus normalized results

All means use the same eight paired queries. Recall changes are **normalized minus raw**, in percentage points; token changes are mean document-text tokens.

| ASR | Retrieval | Raw recall (%) | Normalized recall (%) | Change (pp) | Raw tokens | Normalized tokens | Token change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | 48.542 | 58.958 | +10.417 | 1091.750 | 1115.500 | +23.750 |
| Whisper large-v3 | top-5 | 54.792 | 58.958 | +4.167 | 1115.250 | 1115.500 | +0.250 |
| MAI-Transcribe 2 | top-5 | 54.792 | 58.958 | +4.167 | 1144.375 | 1115.500 | -28.875 |
| Voxtral Small Transcribe 3 | top-5 | 54.792 | 58.958 | +4.167 | 1144.375 | 1115.500 | -28.875 |
| GLM-ASR-Nano | cutoff 0.8925 | 44.375 | 44.375 | +0.000 | 757.750 | 555.750 | -202.000 |
| Whisper large-v3 | cutoff 0.8925 | 47.500 | 44.375 | -3.125 | 792.750 | 555.750 | -237.000 |
| MAI-Transcribe 2 | cutoff 0.8925 | 44.375 | 44.375 | +0.000 | 805.250 | 555.750 | -249.500 |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | 44.375 | 44.375 | +0.000 | 781.000 | 555.750 | -225.250 |

For every model, normalized top-5 returns five documents per query. The normalized cutoff returns **2.5 documents on average**, with top-1 fallback on **3/8 queries**. Candidate-minus-GLM recall and token differences are zero for both policies, including the cutoff-minus-top-5 interaction contrast.

The cutoff uses fewer tokens while returning lower mean recall than top-5 (**44.375% versus 58.958%**). This comparison is not a matched-recall comparison between retrieval policies.

## Per-query equality and changes

| Query | Distinct raw strings | Distinct normalized strings | Same normalized top-5 IDs across models | Same normalized cutoff IDs across models |
| --- | ---: | ---: | --- | --- |
| `q_bbf5affa094195bb` | 2 | 1 | yes | yes |
| `q_e3e73f9ccdca5810` | 3 | 1 | yes | yes |
| `q_3df64fbcb882bbee` | 1 | 1 | yes | yes |
| `q_a8ea5a4d5297476b` | 2 | 1 | yes | yes |
| `q_4649a82e62e04522` | 3 | 1 | yes | yes |
| `q_754d65c473d57184` | 3 | 1 | yes | yes |
| `q_9bbd995c2b2ddb55` | 1 | 1 | yes | yes |
| `q_900c4b5340b303c0` | 1 | 1 | yes | yes |

Equal normalized strings leading to equal deterministic retrieval outputs is expected. Equality was measured after independently encoding every arm; outputs were not copied from GLM.

Two queries explain all top-5 recall changes:

- `q_e3e73f9ccdca5810`, the `I S A/ISA` example: GLM rises from **0.5 to 1.0**; the other three models remain at 1.0. This contributes **+6.25 pp** to GLM's mean.
- `q_3df64fbcb882bbee`, the checkbook example: all four raw transcripts were already identical, yet the combined case/punctuation transformation changes recall from **1/3 to 2/3** for every model. This contributes **+4.167 pp** to every mean. The other six queries' top-5 recalls are unchanged.

GLM's cutoff recall is unchanged on **each of the eight queries**, while its mean context cost falls from **757.75 to 555.75 tokens**. Four GLM cutoff selections change IDs. This is a descriptive input-normalization benefit at the same fixed policy on this sample, not a calibrated efficiency frontier or a general deployment guarantee.

Whisper loses its raw cutoff advantage: mean recall falls by **3.125 pp**, from 47.500% to the common 44.375%. MAI/Voxtral retain their raw mean cutoff recall, with lower context tokens. Combined normalization does not isolate which component caused each effect; acronym, case, punctuation and spacing were not separately ablated.

## Independent validation and preserved evidence

- Four formatting equivalence, punctuation, content-preservation and idempotence tests passed inside Slurm before retrieval.
- Separate CPU Slurm audit **3724592** completed in six seconds and passed: **128 retrieval cells**, **16 summary rows**, **32 transcript mappings**, **64 text hashes**, **66 input hashes** and **5 rerun source hashes** were checked. It retokenized **55 unique retrieved documents** from the original corpus and recounted recall from original qrels.
- The audit checks provenance, saved selected-score/rule structure, metrics, equality and accounting. It does not rerun the encoder or independently reconstruct full rankings; full uncapped ranking is performed by the rerun script.
- Private run directory: `/home/s2487016/edin-mls-asr-pilot/rag/results/asr-four-model-normalized-20261116/`. It retains raw-to-normalized mappings, both conditions' per-query outputs and summaries, source/input hashes, environment, raw-control check, comparison, logs and audit. The date suffix identifies the original pilot batch.
- Downloaded archive SHA256: `8e1613469cdc59435def0596555b93e4b61af33709cfce32bb8377a8931bc5c6`.
- Key evidence SHA256: normalized retrieval `0fa58465aa56b97fdbe1240152782db1578f148aede15063ce48e477e6b95b56`; comparison `2169ccb20a61c322ecb6639e519c078b55e311d95a2a6ea648cbb2421afce129`.

Reproduce from the PR checkout with `sbatch rag/extensions/run_asr_normalization_rerun.sbatch` after preparing the saved raw pilot and course assets. [Rerun code](../rag/extensions/asr_normalization_rerun.py), [Slurm submission](../rag/extensions/run_asr_normalization_rerun.sbatch), [tests](../rag/tests/test_transcript_normalization.py) and [independent auditor](../rag/extensions/audit_asr_normalization_rerun.py) are included in this PR. Keep private evidence and credentials out of Git.

## Interpretation

In these eight reused dev examples, the apparent cross-model retrieval advantage is removable by one common formatting transformation; no residual between-model difference remains under either fixed policy. Formatting also changes absolute retrieval performance even when raw ASR texts already agree. This supports treating transcript normalization as a controlled factor in the ASR–retrieval study.

The rule was chosen after seeing the raw pilot, and the sample/cutoff are not held out. No reference-based WER, repeated ASR samples, confidence intervals, downstream answer evaluation or cross-model latency benchmark was run. The combined rule cannot identify a unique causal contribution for acronym formatting versus punctuation/case, and it is not a universal initialism parser.
