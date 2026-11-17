# Full-audio ASR × formatting normalization × retrieval results — 2026-11-17 UTC

Subsequent analysis: [recall–token curves, separated calibration and paired intervals](asr-matched-curves-results.md). It reuses this experiment's saved vectors and preserves every fixed result below; it makes no new ASR or encoder calls.

Transcript comparison: [character and word edit-distance results](asr-edit-distance-results.md) quantify all six model pairs before and after the same frozen normalization. They measure inter-model disagreement without a reference transcript.

The frozen experiment completed **475/475 supplied audio queries**: **350 practice and 125 dev**, with **1,900 four-model transcripts and all 7,600 retrieval cells**. The exact 32 original pilot transcripts were reused; the remaining 467 recordings were completed with the same four ASR arms. This note presents saved audited aggregates; it does not regenerate or revise transcripts.

## Main findings

- **The large pilot recall gain does not generalize to the expanded sample.** On all 475 queries, normalization changes top-5 macro recall by only -0.130 to +0.256 percentage points across the four models. On the 117 remaining dev queries, GLM changes by **-0.783 pp**, compared with +10.417 pp in the original eight-query pilot.
- **Formatting agreement increases, but model differences remain.** Four-model text identity increases from **195/475 (41.1%)** to **390/475 (82.1%)**. The remaining **85 queries have text differences**; this is not a count of queries with different retrieval results. Retrieval gaps do not uniformly shrink: Voxtral-minus-GLM top-5 recall is +1.551 pp raw and +1.609 pp normalized.
- **Mean document-token costs decrease for every model at both policies.** Top-5 always returns five documents, so its reduction reflects the lengths of selected document texts, not fewer returned documents. The mean top-5 token reductions are approximately 29–39 tokens per query, with small mixed recall changes.
- **Cutoff token savings accompany lower recall.** With the fixed 0.8925 cutoff, all four models lose mean recall (-0.380 to -1.211 pp) while using fewer tokens (approximately 66–123 fewer per query). This is not a lossless or matched-recall efficiency improvement. No threshold recalibration was performed.
- These are descriptive measurements for this provided dataset and frozen configuration. The rule was chosen after the pilot; practice/dev and remaining cohorts are reported separately below. The results do not establish equal general ASR quality, a universal normalization benefit or statistical significance.

## Fixed settings and provenance

- Normalization policy: `acronym-punctuation-v1`; unchanged file SHA256: `fff6993b2f1ae85db3a65a6c645a6e2ac3ad0ec415eb82af3f73a763de8e9dd2`.
- The same rule is applied to every model: initialism merging, Unicode/case/punctuation/hyphen and whitespace normalization. Raw text is retained privately; no reference-based correction, domain dictionary or corpus-text normalization is applied.
- Same supplied `enc_a` on CPU with its existing dtype behavior, 24,000 corpus records and their existing vectors; independent encoding of each raw and normalized transcript; full float32 cosine ranking with stable corpus-row tie breaking.
- Fixed top-5, or all scores >=0.8925 with top-1 fallback if none qualify and no document-count cap. Original positive qrels are selected by each recording's own practice/dev split. No threshold retuning is included.
- Context cost is the sum of individually tokenized selected document texts, without special tokens, question/system text or prompt framing. It is **not a billed LLM prompt-token count**.
- Tokenizer: `mistralai/Mistral-Small-4-119B-2603`, revision `a11f36bebf709121056b1dbcc943d1c6afbe494d`, SHA256 `2ba5b3330fd84d5376fcca797cfb3b42eee6241ce23e3271e6fb2a115a8751bd`.
- GLM snapshot: `61ba4e0b3309b6656edea3e93e419f7bd5c61957`; `ASR_CUDA=all`, greedy decoding and 100 new-token ceiling. OpenRouter requests retain the pilot's English/temperature-zero/JSON options and MAI verbatim setting. External service aliases are not immutable checkpoint identifiers.
- All experiment execution, tests and audits take place in Edinburgh Slurm. No answer generation, judging or reference-based WER evaluation is included.

| Environment Slurm job | Stage | Source commit | Host | Recorded worktree |
| --- | --- | --- | --- | --- |
| `3724675` | all | `7fe292655f987de7a41b1898e377b32db02bc7a7` | `saxa.inf.ed.ac.uk` | clean |

Completed-experiment job: **3724675**. Independent audit job: **3724958**, status **pass**. Rendering allocation Slurm job: **3724958**.

## Coverage and cohort results

Each cohort uses one common complete query set across all 16 settings (four ASR models × raw/normalized × two retrieval rules). Recall is the macro mean of original-query labelled-evidence recall. Changes below are **normalized minus raw**; recall changes are percentage points (pp). Token changes use the document-text measure above. These are descriptive measurements, without significance claims.

### All supplied audio (475 queries)

Complete paired coverage: **475/475 queries**, **7600/7600 retrieval cells**; missing cells: **0**.

Four-model text identity, counted per original query: **195/475 raw**, **390/475 normalized**. Identical normalized inputs can lead to identical deterministic retrieval outputs; this does not establish equal general ASR quality.

| ASR | Retrieval | Raw recall (%) | Normalized recall (%) | Δ recall (pp) | Raw tokens | Normalized tokens | Δ tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | 41.033 | 41.085 | +0.053 | 1098.240 | 1069.057 | -29.183 |
| Whisper large-v3 | top-5 | 41.421 | 41.677 | +0.256 | 1113.872 | 1075.322 | -38.549 |
| MAI-Transcribe 2 | top-5 | 42.403 | 42.273 | -0.130 | 1115.579 | 1077.263 | -38.316 |
| Voxtral Small Transcribe 3 | top-5 | 42.584 | 42.694 | +0.111 | 1112.775 | 1075.238 | -37.537 |
| GLM-ASR-Nano | cutoff 0.8925 | 30.989 | 30.609 | -0.380 | 553.048 | 487.505 | -65.543 |
| Whisper large-v3 | cutoff 0.8925 | 32.576 | 31.365 | -1.211 | 618.798 | 496.078 | -122.720 |
| MAI-Transcribe 2 | cutoff 0.8925 | 32.817 | 31.871 | -0.945 | 579.269 | 502.796 | -76.474 |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | 33.098 | 32.398 | -0.700 | 577.112 | 502.857 | -74.255 |

Query-level changes after normalization; every count uses the same **475** paired queries. Token decrease/increase counts describe context size, not an accuracy-adjusted benefit.

| ASR | Retrieval | Recall gain / loss / tie | Token decrease / increase / tie | Changed selected-ID lists |
| --- | --- | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | 14 / 11 / 450 | 127 / 80 / 268 | 342 |
| Whisper large-v3 | top-5 | 17 / 10 / 448 | 133 / 91 / 251 | 356 |
| MAI-Transcribe 2 | top-5 | 14 / 11 / 450 | 132 / 77 / 266 | 342 |
| Voxtral Small Transcribe 3 | top-5 | 17 / 10 / 448 | 130 / 80 / 265 | 346 |
| GLM-ASR-Nano | cutoff 0.8925 | 7 / 10 / 458 | 78 / 34 / 363 | 142 |
| Whisper large-v3 | cutoff 0.8925 | 8 / 18 / 449 | 110 / 35 / 330 | 167 |
| MAI-Transcribe 2 | cutoff 0.8925 | 6 / 13 / 456 | 81 / 33 / 361 | 143 |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | 6 / 11 / 458 | 79 / 35 / 361 | 145 |

### Dev (125 queries)

Complete paired coverage: **125/125 queries**, **2000/2000 retrieval cells**; missing cells: **0**.

Four-model text identity, counted per original query: **46/125 raw**, **106/125 normalized**. Identical normalized inputs can lead to identical deterministic retrieval outputs; this does not establish equal general ASR quality.

| ASR | Retrieval | Raw recall (%) | Normalized recall (%) | Δ recall (pp) | Raw tokens | Normalized tokens | Δ tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | 37.950 | 37.883 | -0.067 | 1039.256 | 1028.216 | -11.040 |
| Whisper large-v3 | top-5 | 37.683 | 38.083 | +0.400 | 1071.792 | 1037.536 | -34.256 |
| MAI-Transcribe 2 | top-5 | 38.350 | 38.283 | -0.067 | 1057.480 | 1033.384 | -24.096 |
| Voxtral Small Transcribe 3 | top-5 | 38.350 | 38.283 | -0.067 | 1054.552 | 1026.048 | -28.504 |
| GLM-ASR-Nano | cutoff 0.8925 | 27.743 | 27.143 | -0.600 | 571.760 | 474.808 | -96.952 |
| Whisper large-v3 | cutoff 0.8925 | 29.210 | 27.810 | -1.400 | 635.904 | 477.664 | -158.240 |
| MAI-Transcribe 2 | cutoff 0.8925 | 29.210 | 28.610 | -0.600 | 585.328 | 479.784 | -105.544 |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | 29.210 | 28.610 | -0.600 | 582.480 | 479.904 | -102.576 |

Query-level changes after normalization; every count uses the same **125** paired queries. Token decrease/increase counts describe context size, not an accuracy-adjusted benefit.

| ASR | Retrieval | Recall gain / loss / tie | Token decrease / increase / tie | Changed selected-ID lists |
| --- | --- | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | 5 / 4 / 116 | 32 / 27 / 66 | 89 |
| Whisper large-v3 | top-5 | 6 / 3 / 116 | 38 / 27 / 60 | 101 |
| MAI-Transcribe 2 | top-5 | 4 / 3 / 118 | 34 / 22 / 69 | 90 |
| Voxtral Small Transcribe 3 | top-5 | 4 / 3 / 118 | 36 / 24 / 65 | 94 |
| GLM-ASR-Nano | cutoff 0.8925 | 2 / 2 / 121 | 23 / 10 / 92 | 40 |
| Whisper large-v3 | cutoff 0.8925 | 2 / 4 / 119 | 28 / 10 / 87 | 44 |
| MAI-Transcribe 2 | cutoff 0.8925 | 1 / 1 / 123 | 22 / 10 / 93 | 40 |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | 1 / 1 / 123 | 20 / 11 / 94 | 40 |

### Remaining dev, excluding the pilot (117 queries)

Complete paired coverage: **117/117 queries**, **1872/1872 retrieval cells**; missing cells: **0**.

Four-model text identity, counted per original query: **43/117 raw**, **98/117 normalized**. Identical normalized inputs can lead to identical deterministic retrieval outputs; this does not establish equal general ASR quality.

| ASR | Retrieval | Raw recall (%) | Normalized recall (%) | Δ recall (pp) | Raw tokens | Normalized tokens | Δ tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | 37.226 | 36.442 | -0.783 | 1035.667 | 1022.248 | -13.419 |
| Whisper large-v3 | top-5 | 36.513 | 36.656 | +0.142 | 1068.821 | 1032.205 | -36.615 |
| MAI-Transcribe 2 | top-5 | 37.226 | 36.869 | -0.356 | 1051.538 | 1027.769 | -23.769 |
| Voxtral Small Transcribe 3 | top-5 | 37.226 | 36.869 | -0.356 | 1048.410 | 1019.932 | -28.479 |
| GLM-ASR-Nano | cutoff 0.8925 | 26.606 | 25.965 | -0.641 | 559.043 | 469.274 | -89.769 |
| Whisper large-v3 | cutoff 0.8925 | 27.960 | 26.677 | -1.282 | 625.179 | 472.325 | -152.855 |
| MAI-Transcribe 2 | cutoff 0.8925 | 28.173 | 27.532 | -0.641 | 570.291 | 474.590 | -95.701 |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | 28.173 | 27.532 | -0.641 | 568.906 | 474.718 | -94.188 |

Query-level changes after normalization; every count uses the same **117** paired queries. Token decrease/increase counts describe context size, not an accuracy-adjusted benefit.

| ASR | Retrieval | Recall gain / loss / tie | Token decrease / increase / tie | Changed selected-ID lists |
| --- | --- | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | 3 / 4 / 110 | 30 / 25 / 62 | 84 |
| Whisper large-v3 | top-5 | 5 / 3 / 109 | 35 / 25 / 57 | 95 |
| MAI-Transcribe 2 | top-5 | 3 / 3 / 111 | 31 / 22 / 64 | 86 |
| Voxtral Small Transcribe 3 | top-5 | 3 / 3 / 111 | 33 / 24 / 60 | 90 |
| GLM-ASR-Nano | cutoff 0.8925 | 2 / 2 / 113 | 19 / 10 / 88 | 36 |
| Whisper large-v3 | cutoff 0.8925 | 1 / 3 / 113 | 24 / 9 / 84 | 39 |
| MAI-Transcribe 2 | cutoff 0.8925 | 1 / 1 / 115 | 18 / 10 / 89 | 36 |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | 1 / 1 / 115 | 17 / 10 / 90 | 36 |

### Practice (350 queries)

Complete paired coverage: **350/350 queries**, **5600/5600 retrieval cells**; missing cells: **0**.

Four-model text identity, counted per original query: **149/350 raw**, **284/350 normalized**. Identical normalized inputs can lead to identical deterministic retrieval outputs; this does not establish equal general ASR quality.

| ASR | Retrieval | Raw recall (%) | Normalized recall (%) | Δ recall (pp) | Raw tokens | Normalized tokens | Δ tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | 42.134 | 42.229 | +0.095 | 1119.306 | 1083.643 | -35.663 |
| Whisper large-v3 | top-5 | 42.755 | 42.960 | +0.205 | 1128.900 | 1088.817 | -40.083 |
| MAI-Transcribe 2 | top-5 | 43.851 | 43.698 | -0.152 | 1136.329 | 1092.934 | -43.394 |
| Voxtral Small Transcribe 3 | top-5 | 44.096 | 44.270 | +0.174 | 1133.569 | 1092.806 | -40.763 |
| GLM-ASR-Nano | cutoff 0.8925 | 32.148 | 31.847 | -0.301 | 546.366 | 492.040 | -54.326 |
| Whisper large-v3 | cutoff 0.8925 | 33.778 | 32.635 | -1.143 | 612.689 | 502.654 | -110.034 |
| MAI-Transcribe 2 | cutoff 0.8925 | 34.105 | 33.036 | -1.069 | 577.106 | 511.014 | -66.091 |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | 34.486 | 33.750 | -0.735 | 575.194 | 511.054 | -64.140 |

Query-level changes after normalization; every count uses the same **350** paired queries. Token decrease/increase counts describe context size, not an accuracy-adjusted benefit.

| ASR | Retrieval | Recall gain / loss / tie | Token decrease / increase / tie | Changed selected-ID lists |
| --- | --- | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | 9 / 7 / 334 | 95 / 53 / 202 | 253 |
| Whisper large-v3 | top-5 | 11 / 7 / 332 | 95 / 64 / 191 | 255 |
| MAI-Transcribe 2 | top-5 | 10 / 8 / 332 | 98 / 55 / 197 | 252 |
| Voxtral Small Transcribe 3 | top-5 | 13 / 7 / 330 | 94 / 56 / 200 | 252 |
| GLM-ASR-Nano | cutoff 0.8925 | 5 / 8 / 337 | 55 / 24 / 271 | 102 |
| Whisper large-v3 | cutoff 0.8925 | 6 / 14 / 330 | 82 / 25 / 243 | 123 |
| MAI-Transcribe 2 | cutoff 0.8925 | 5 / 12 / 333 | 59 / 23 / 268 | 103 |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | 5 / 10 / 335 | 59 / 24 / 267 | 105 |

### Remaining audio, excluding the pilot (467 queries)

Complete paired coverage: **467/467 queries**, **7472/7472 retrieval cells**; missing cells: **0**.

Four-model text identity, counted per original query: **192/467 raw**, **382/467 normalized**. Identical normalized inputs can lead to identical deterministic retrieval outputs; this does not establish equal general ASR quality.

| ASR | Retrieval | Raw recall (%) | Normalized recall (%) | Δ recall (pp) | Raw tokens | Normalized tokens | Δ tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | 40.904 | 40.779 | -0.125 | 1098.351 | 1068.261 | -30.090 |
| Whisper large-v3 | top-5 | 41.192 | 41.381 | +0.189 | 1113.848 | 1074.634 | -39.214 |
| MAI-Transcribe 2 | top-5 | 42.191 | 41.987 | -0.203 | 1115.086 | 1076.608 | -38.478 |
| Voxtral Small Transcribe 3 | top-5 | 42.374 | 42.416 | +0.041 | 1112.233 | 1074.548 | -37.685 |
| GLM-ASR-Nano | cutoff 0.8925 | 30.760 | 30.373 | -0.386 | 549.542 | 486.336 | -63.206 |
| Whisper large-v3 | cutoff 0.8925 | 32.320 | 31.142 | -1.178 | 615.818 | 495.056 | -120.762 |
| MAI-Transcribe 2 | cutoff 0.8925 | 32.619 | 31.657 | -0.962 | 575.398 | 501.889 | -73.510 |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | 32.904 | 32.193 | -0.712 | 573.619 | 501.951 | -71.668 |

Query-level changes after normalization; every count uses the same **467** paired queries. Token decrease/increase counts describe context size, not an accuracy-adjusted benefit.

| ASR | Retrieval | Recall gain / loss / tie | Token decrease / increase / tie | Changed selected-ID lists |
| --- | --- | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | 12 / 11 / 444 | 125 / 78 / 264 | 337 |
| Whisper large-v3 | top-5 | 16 / 10 / 441 | 130 / 89 / 248 | 350 |
| MAI-Transcribe 2 | top-5 | 13 / 11 / 443 | 129 / 77 / 261 | 338 |
| Voxtral Small Transcribe 3 | top-5 | 16 / 10 / 441 | 127 / 80 / 260 | 342 |
| GLM-ASR-Nano | cutoff 0.8925 | 7 / 10 / 450 | 74 / 34 / 359 | 138 |
| Whisper large-v3 | cutoff 0.8925 | 7 / 17 / 443 | 106 / 34 / 327 | 162 |
| MAI-Transcribe 2 | cutoff 0.8925 | 6 / 13 / 448 | 77 / 33 / 357 | 139 |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | 6 / 11 / 450 | 76 / 34 / 357 | 141 |

### Original pilot (8 queries)

Complete paired coverage: **8/8 queries**, **128/128 retrieval cells**; missing cells: **0**.

Four-model text identity, counted per original query: **3/8 raw**, **8/8 normalized**. Identical normalized inputs can lead to identical deterministic retrieval outputs; this does not establish equal general ASR quality.

| ASR | Retrieval | Raw recall (%) | Normalized recall (%) | Δ recall (pp) | Raw tokens | Normalized tokens | Δ tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | 48.542 | 58.958 | +10.417 | 1091.750 | 1115.500 | +23.750 |
| Whisper large-v3 | top-5 | 54.792 | 58.958 | +4.167 | 1115.250 | 1115.500 | +0.250 |
| MAI-Transcribe 2 | top-5 | 54.792 | 58.958 | +4.167 | 1144.375 | 1115.500 | -28.875 |
| Voxtral Small Transcribe 3 | top-5 | 54.792 | 58.958 | +4.167 | 1144.375 | 1115.500 | -28.875 |
| GLM-ASR-Nano | cutoff 0.8925 | 44.375 | 44.375 | +0.000 | 757.750 | 555.750 | -202.000 |
| Whisper large-v3 | cutoff 0.8925 | 47.500 | 44.375 | -3.125 | 792.750 | 555.750 | -237.000 |
| MAI-Transcribe 2 | cutoff 0.8925 | 44.375 | 44.375 | +0.000 | 805.250 | 555.750 | -249.500 |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | 44.375 | 44.375 | +0.000 | 781.000 | 555.750 | -225.250 |

Query-level changes after normalization; every count uses the same **8** paired queries. Token decrease/increase counts describe context size, not an accuracy-adjusted benefit.

| ASR | Retrieval | Recall gain / loss / tie | Token decrease / increase / tie | Changed selected-ID lists |
| --- | --- | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | 2 / 0 / 6 | 2 / 2 / 4 | 5 |
| Whisper large-v3 | top-5 | 1 / 0 / 7 | 3 / 2 / 3 | 6 |
| MAI-Transcribe 2 | top-5 | 1 / 0 / 7 | 3 / 0 / 5 | 4 |
| Voxtral Small Transcribe 3 | top-5 | 1 / 0 / 7 | 3 / 0 / 5 | 4 |
| GLM-ASR-Nano | cutoff 0.8925 | 0 / 0 / 8 | 4 / 0 / 4 | 4 |
| Whisper large-v3 | cutoff 0.8925 | 1 / 1 / 6 | 4 / 1 / 3 | 5 |
| MAI-Transcribe 2 | cutoff 0.8925 | 0 / 0 / 8 | 4 / 0 / 4 | 4 |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | 0 / 0 / 8 | 3 / 1 / 4 | 4 |

## Independent audit and cost accounting

The independent audit passed for **1900 transcripts**, **3800 text hashes**, **3800 saved-vector rankings over 24000 records each**, and **7600 retrieval cells**. It checked uncapped cutoff completeness, original qrel recall, **2624 unique retokenized documents**, **96 cohort-summary rows** and **48 normalization contrasts**. It did not rerun ASR or the encoder.

The original pilot's **128 raw/normalized control cells** were reproduced; maximum selected-score drift was **0.000000000000**, within the fixed 1e-6 gate. Independent full-ranking score drift was **0.000000000000**.

- **New-attempt response costs:** USD **0.253048456111**, from **1401** newly logged attempts. This includes a failed attempt only when that attempt returned cost metadata.
- **Reported costs for all successful transcript responses, including the cached pilot:** USD **0.257379769944**. Cost metadata covers **1425/1425** successful external responses. This has a different scope from new-attempt costs and excludes failed-attempt charges.
- Failed or ambiguous attempts: **0**. Unresolved failed/ambiguous cost: **no**.
- Later read-only key-usage snapshot (2026-11-17T02:12:20.822693+00:00, Slurm **3724958**): usage change USD **0.253047849000**; separately recorded new-response costs USD **0.253048456111**. These separate measurements are not assumed to be bit-for-bit equal.

## Limits and interpretation boundaries

The formatting rule was chosen after inspecting the original eight dev queries. It was fixed before the remaining recordings, but the public dev data had already informed cutoff selection; the pilot and remaining-dev cohorts are not independent held-out tests. Practice and dev results are therefore also shown separately.

The combined rule changes initialisms, case, punctuation, hyphens and spacing together. These are not component-specific causal ablations or universal spelling/recognition corrections. One transcript per audio/model is used; cached pilot calls and new API calls span separate requests, and the external service aliases do not guarantee immutable model revisions. The GLM 100-new-token ceiling is retained.

No independent reference transcript, WER evaluation, answer-quality evaluation, repeated-ASR uncertainty estimate or cross-model inference-latency benchmark is included. Original qrels may be incomplete. A policy using fewer tokens while recovering less evidence is not a matched-recall efficiency win. No statistical significance or general model-superiority claim is made by these tables.

## Reproduction and preserved evidence

Use [the frozen full protocol](asr-rag-full-protocol.md), [experiment runner](../rag/extensions/asr_full_experiment.py), [Slurm submission](../rag/extensions/run_asr_full_experiment.sbatch), [cohort summary](../rag/extensions/asr_full_summary.py), [independent audit](../rag/extensions/audit_asr_full.py) and [presentation renderer](../rag/extensions/render_asr_full_results.py). Run coursework only inside Edinburgh Slurm allocations.

The private run directory retains the frozen selection/configuration, cache origins, transcript mappings, request attempts/responses, query vectors, per-query retrieval rows, summaries, billing, environments and audit. This public note embeds aggregates only; raw audio, transcript text, mappings, responses and credentials are not included.

The archived evidence was downloaded privately to `D:\MLS\pilot-results\asr-four-model-full-20261117`. Archive SHA256: `fea1f2c4aa6a2ce2a20222322c1aca67401874d19c4f7a2d90dbf57b05f303b4`.

The scripts use the prepared cluster checkout `~/edin-mls-asr-pilot`, installed course assets, pinned environments/tokenizer and saved pilot runs. Copy the support scripts to the private cache before submitting the provided wrappers:

```bash
mkdir -p "$HOME/.cache/mls-asr-full"
cp rag/extensions/asr_full_preflight.py rag/extensions/audit_asr_full.py \
   rag/extensions/refresh_asr_full_billing.py rag/extensions/render_asr_full_results.py \
   "$HOME/.cache/mls-asr-full/"
sbatch rag/extensions/run_asr_full_preflight.sbatch
# After preflight passes:
sbatch rag/extensions/run_asr_full_experiment.sbatch
# After the full experiment completes:
sbatch rag/extensions/run_asr_full_audit.sbatch
```

Source/input hashes and caches are checked before reuse. The audit reconstructs rankings from saved vectors; it does not make paid transcription calls. The public note contains a presentation-only Markdown fix and the interpretation above in addition to the Slurm-rendered tables; the privately archived generated report remains unchanged.

| Audited artifact | SHA256 |
| --- | --- |
| `config.json` | `08e26c97c6447d7147e971b5feb97e3ffe5ef41490ff7a1c11fca4af335ce43a` |
| `summary.json` | `618081a6298107aaea5322bfd09632fc00f993f1867f6fbb58e9d7b74c180654` |
| `retrieval.jsonl` | `c7217a6a01c4bf34f7cf717aeb986522b8f90efd75d2d058145b90917765d809` |
| `query-vectors.npz` | `e3600f52fcddd2f70b063c139ed30f1005bb2aecf5e4e0fbea4571e0cddfc10f` |
