# Four-model ASR × retrieval pilot — 16 November 2026

The eight-query pilot completed on Edinburgh Slurm. It compares **GLM-ASR-Nano, Whisper large-v3, MAI-Transcribe 2 and Voxtral Small Transcribe 3** under fixed top-5 and absolute cutoff 0.8925.

The main observation is an interaction between transcript representation and retrieval selection: all three alternatives gain 6.25 recall percentage points under top-5, while the cutoff gains 3.125 points for Whisper and zero for MAI/Voxtral. The top-5 gain is entirely concentrated in one acronym-formatting example. This is exploratory evidence, not a general ASR ranking.

## Design and measurement boundaries

- Source commit for the completed run: `75d1a1e0a84a3bbbac294c839888dc90a15ffa93`.
- Successful Slurm job: **3724508**, completed in **12m08s**, including staging, validation, compilation, model loading and the experiment. This is not an ASR latency benchmark.
- GPU: NVIDIA H200 MIG 1g.18gb, SM90; CUDA 12.8.93. GLM used the course adapter with `ASR_CUDA=all`, greedy decoding and a 100-new-token ceiling. GLM snapshot: `61ba4e0b3309b6656edea3e93e419f7bd5c61957`.
- Same eight dev recordings, **44.175 seconds** total, sampled without replacement from 125 eligible queries using sorted IDs and `random.Random(26)`. Selection and audio SHA256 were frozen before transcription. There were no query substitutions.
- Same supplied `enc_a` encoder, 384-dimensional vectors, full exact cosine ranking over 24,000 records, corpus-row tie breaking and original positive qrels.
- The supplied encoder ran on CPU. The installed libraries loaded its weights in float16, as recorded by the runtime warning; vectors were exposed as float32 by the existing adapter. All four ASR arms shared these settings.
- Top-5 selects five documents. Cutoff selects all scores >= 0.8925, with top-1 fallback and no candidate-count cap.
- Token cost is the **sum of individually tokenized retrieved document texts**, with `add_special_tokens=False`, using `mistralai/Mistral-Small-4-119B-2603`, revision `a11f36bebf709121056b1dbcc943d1c6afbe494d`.
- Tokenizer SHA256: `2ba5b3330fd84d5376fcca797cfb3b42eee6241ce23e3271e6fb2a115a8751bd`. Counts exclude the question, system text, labels, separators and chat framing. They are not billed LLM prompt tokens.
- The external models used OpenRouter's dedicated transcription endpoint, English language setting, temperature zero, JSON output and no vocabulary hints. MAI requested Azure verbatim style with diarization disabled. Raw requested options and returned metadata are retained. No automatic paid POST retries, answer generation or judging were used.

## Complete paired results

Every row uses the **same eight complete four-model queries**. Changes are candidate ASR minus GLM for the same retrieval rule.

| ASR | Retrieval | Macro recall (%) | Change vs GLM (pp) | Mean context tokens | Token change vs GLM | Mean docs | Fallback queries |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | 48.542 | +0.000 | 1091.750 | +0.000 | 5.000 | 0 |
| Whisper large-v3 | top-5 | 54.792 | +6.250 | 1115.250 | +23.500 | 5.000 | 0 |
| MAI-Transcribe 2 | top-5 | 54.792 | +6.250 | 1144.375 | +52.625 | 5.000 | 0 |
| Voxtral Small Transcribe 3 | top-5 | 54.792 | +6.250 | 1144.375 | +52.625 | 5.000 | 0 |
| GLM-ASR-Nano | cutoff 0.8925 | 44.375 | +0.000 | 757.750 | +0.000 | 3.000 | 3 |
| Whisper large-v3 | cutoff 0.8925 | 47.500 | +3.125 | 792.750 | +35.000 | 3.375 | 3 |
| MAI-Transcribe 2 | cutoff 0.8925 | 44.375 | +0.000 | 805.250 | +47.500 | 3.125 | 3 |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | 44.375 | +0.000 | 781.000 | +23.250 | 2.875 | 3 |

For each replacement, the cutoff-minus-top-5 recall-change contrast is **−3.125 pp for Whisper** and **−6.250 pp for MAI/Voxtral**. These are descriptive contrasts on eight queries, without a significance claim.

All three alternatives consume more mean document-text tokens than GLM at both frozen operating points. MAI/Voxtral's higher context-token counts under the cutoff do not accompany higher mean recall in this sample. Comparing top-5 and cutoff directly also changes recall; this table is not a matched-recall token-efficiency experiment.

## Where the recall changes come from

### Acronym representation: `q_e3e73f9ccdca5810`

GLM writes the acronym as `I S A`; all three alternatives write `ISA`. The wording otherwise asks the same investment question, with some punctuation/spacing differences. The top-5 recall changes from **0.5 to 1.0** for all three alternatives. This single query accounts for the entire **+6.25 pp** aggregate top-5 change.

At the cutoff:

| Transcript source | Recall | Documents | Context tokens |
| --- | ---: | ---: | ---: |
| GLM | 0.5 | 3 | 1,265 |
| Whisper | 1.0 | 4 | 860 |
| MAI | 0.5 | 4 | 1,645 |
| Voxtral | 0.5 | 4 | 1,645 |

Whisper's ranking places an additional qrel-relevant record inside the cutoff prefix. MAI/Voxtral put that record fifth, outside their four-document cutoff selection, although it appears in their top-5. Equal document counts therefore need not give equal evidence coverage or token cost.

This does not establish that one spelling is a recognition error: an independently verified reference transcript was not used. It demonstrates sensitivity to how the speech is rendered as text.

### Punctuation and top-1 fallback: `q_bbf5affa094195bb`

GLM, MAI and Voxtral return identical wording. Whisper inserts a question break between the words corresponding to transaction and records. Top-5 still contains the same set and recall remains **0.25**.

Every model's highest score is below 0.8925, so the cutoff falls back to top-1. Whisper swaps the first two records, reducing fallback recall from **0.25 to 0**. Its top score is slightly higher (0.881719 versus 0.880386), which does not make the selected first record qrel-relevant.

Whisper's cutoff gains 0.5 recall on the acronym query and loses 0.25 on this query. Across eight queries, the net change is **+3.125 pp**. No other query changes recall.

## Validation and API accounting

- All **32 transcripts** and **64 retrieval cells** completed; no failed or missing queries were used in the summary.
- A separate Slurm CPU audit, job **3724535**, recomputed all 64 recalls against original qrels, retokenized **53 unique selected documents**, checked the paired summary and validated audio/model provenance. The audit passed.
- All **24 external responses** carry distinct saved request identifiers and cost metadata. Response costs sum to **USD 0.004331313833**.
- A later read-only key-usage check reports **USD 0.004331305000** of cumulative usage from a zero-usage start. The absolute difference from the response sum is below USD 0.00000001; the amounts are reported separately rather than declared bit-for-bit identical. Both round to **USD 0.00433**.
- Targeted Slurm checks cover native SDK JSON request/response handling and use of the same complete query intersection for every summary row.
- Earlier attempts stopped before any ASR API request: one failed because the node's shared `/tmp` was full; one was stopped during slow shared-filesystem preflight. The successful run used node-local scratch for data/model reads and temporary files.
- Existing library/data-loader/encoder/retrieval algorithms and supplied CUDA test interfaces were not replaced. The pilot composes the existing implementations.

The private evidence directory is `rag/results/asr-four-model-pilot-20261116/` in the pilot worktree. It includes frozen selection/configuration, transcripts and raw responses, request metadata, retrieval rows, tokenizer metadata, environment/source hashes, completion, summary and the independent audit. A private downloaded archive has SHA256 `902cd7e2316bff50fd5ab3e26af0fbdd02364238cd4916c4a11024b794f71eb9`. Raw responses, audio and credentials are not committed with this note.

Key artifact SHA256 values:

| Artifact | SHA256 |
| --- | --- |
| `selection.json` | `3dc3e0325a45af4584dbbf99202faa65c90900801788112e5865cb0ee170a13f` |
| `config.json` | `1bb04b19d6c545d4baa6585842912932efdf0df11de72ecc1c990be04bcedaf6` |
| `retrieval.jsonl` | `5faca76bbe412533d9384869f6f92daacb2d8c4f3d603f8a1655352900e3e9a6` |
| `summary.json` | `70166ecbeb8f67bf2e7a416ef20621c8df7c8f2c926d8213416a2133c19b44cc` |

## Interpretation and limits

This pilot supports studying **ASR output representation × retrieval policy**. It does not show that a higher public ASR leaderboard position guarantees higher retrieval recall. The aggregate improvement is dominated by one acronym-formatting example, and punctuation changes a fallback decision in another.

The eight queries come from dev data previously used to choose the cutoff; they are not an independent holdout. All methods share the same original qrels, whose incompleteness remains a limitation. There is one transcript per audio/model, no independent reference-based WER measurement and no answer-quality evaluation. The external service slugs are not immutable checkpoint identifiers, and provider decoding implementations differ.

First-GLM call wall time includes worker/model setup; saved API wall times include network/provider service. This run makes no cross-model inference-speed or full audio-to-answer latency claim. Earlier 324-query answer batches are not pooled into this pilot.

A useful next controlled comparison would apply one predeclared transcript-normalization policy to all models, alongside the raw-transcript condition, to separate acronym/punctuation effects from other ASR differences. That follow-up has not been run.

## Reproduction

Use [the frozen protocol](asr-rag-whisper-pilot.md), [runner](../rag/extensions/asr_retrieval_pilot.py), [Slurm submission](../rag/extensions/run_asr_retrieval_pilot.sbatch) and [dependency constraints](../rag/extensions/asr-pilot-constraints.txt). The accounting recount is [audit_asr_retrieval_pilot.py](../rag/extensions/audit_asr_retrieval_pilot.py). All coursework execution must remain inside an Edinburgh Slurm allocation.
