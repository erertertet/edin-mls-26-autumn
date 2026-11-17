# Four-ASR transcript disagreement: edit distances

This saved-data follow-up measures the magnitude of differences among four ASR outputs for all **475** supplied audio queries. It compares the existing raw text with the saved frozen-normalized text; no transcript or normalization rule is changed. **New ASR calls: 0; encoder calls: 0; external API calls: 0.**

## Findings

- Across all queries, the six-pair mean characters NED changes from 1.996% to 0.881%; paired change -1.115 [-1.243, -0.992] percentage points.
- Across all queries, the six-pair mean words NED changes from 6.888% to 1.657%; paired change -5.230 [-5.867, -4.640] percentage points.

Four-model exact-character identity reproduces **195/475 raw → 390/475 normalized** queries. The remaining **85** normalized text disagreements are retained in every all-query mean. These distances quantify output convergence; they do not establish better recognition accuracy or retrieval.

## All-query overall comparison

Every estimate in the tables is a mean with a pointwise 95% percentile interval in brackets. Δ is normalized minus raw; a negative NED Δ means less inter-model disagreement. The overall value averages six model-pair distances within each query, then averages queries.

| Model pair | Unit | Mean edits: raw → normalized | Mean NED (%): raw → normalized | Paired NED Δ (pp), 95% CI | Zero queries: raw → normalized | NED decrease / tie / increase |
| --- | --- | --- | --- | --- | --- | --- |
| Overall: six-pair query mean | characters | 1.574 [1.324, 1.848] → 0.680 [0.468, 0.918] | 1.996 [1.696, 2.324] → 0.881 [0.628, 1.165] | -1.115 [-1.243, -0.992] | 195 → 390 | 262 / 196 / 17 |
| Overall: six-pair query mean | words | 0.914 [0.811, 1.024] → 0.226 [0.170, 0.287] | 6.888 [6.153, 7.661] → 1.657 [1.258, 2.086] | -5.230 [-5.867, -4.640] | 200 → 390 | 252 / 221 / 2 |

## All queries: six model pairs (n=475)

### Characters

| Model pair | Mean edits: raw → normalized | Mean NED (%): raw → normalized | Paired NED Δ (pp), 95% CI | Zero queries: raw → normalized | NED decrease / tie / increase |
| --- | --- | --- | --- | --- | --- |
| GLM-ASR-Nano / Whisper large-v3 | 2.629 [2.181, 3.137] → 1.187 [0.789, 1.642] | 3.253 [2.731, 3.831] → 1.514 [1.038, 2.056] | -1.739 [-1.944, -1.545] | 214 → 404 | 238 / 215 / 22 |
| GLM-ASR-Nano / MAI-Transcribe 2 | 1.684 [1.253, 2.171] → 1.183 [0.779, 1.629] | 2.131 [1.624, 2.692] → 1.507 [1.020, 2.031] | -0.624 [-0.764, -0.495] | 320 → 409 | 119 / 321 / 35 |
| GLM-ASR-Nano / Voxtral Small Transcribe 3 | 1.642 [1.213, 2.120] → 1.149 [0.752, 1.596] | 2.057 [1.568, 2.602] → 1.467 [0.987, 1.996] | -0.590 [-0.726, -0.466] | 323 → 413 | 117 / 324 / 34 |
| Whisper large-v3 / MAI-Transcribe 2 | 1.554 [1.371, 1.749] → 0.223 [0.149, 0.305] | 1.986 [1.749, 2.242] → 0.305 [0.202, 0.417] | -1.682 [-1.886, -1.484] | 249 → 434 | 219 / 249 / 7 |
| Whisper large-v3 / Voxtral Small Transcribe 3 | 1.486 [1.307, 1.676] → 0.219 [0.147, 0.303] | 1.910 [1.675, 2.154] → 0.313 [0.206, 0.433] | -1.597 [-1.797, -1.406] | 251 → 434 | 214 / 251 / 10 |
| MAI-Transcribe 2 / Voxtral Small Transcribe 3 | 0.451 [0.347, 0.562] → 0.118 [0.067, 0.175] | 0.637 [0.486, 0.805] → 0.179 [0.099, 0.270] | -0.457 [-0.590, -0.336] | 389 → 450 | 70 / 389 / 16 |
### Words

| Model pair | Mean edits: raw → normalized | Mean NED (%): raw → normalized | Paired NED Δ (pp), 95% CI | Zero queries: raw → normalized | NED decrease / tie / increase |
| --- | --- | --- | --- | --- | --- |
| GLM-ASR-Nano / Whisper large-v3 | 1.446 [1.278, 1.629] → 0.345 [0.255, 0.448] | 10.569 [9.394, 11.797] → 2.456 [1.833, 3.153] | -8.114 [-9.131, -7.147] | 220 → 404 | 227 / 247 / 1 |
| GLM-ASR-Nano / MAI-Transcribe 2 | 0.874 [0.728, 1.034] → 0.328 [0.244, 0.423] | 6.193 [5.251, 7.205] → 2.317 [1.738, 2.948] | -3.875 [-4.620, -3.190] | 320 → 409 | 117 / 356 / 2 |
| GLM-ASR-Nano / Voxtral Small Transcribe 3 | 0.869 [0.722, 1.032] → 0.307 [0.223, 0.402] | 6.124 [5.159, 7.144] → 2.202 [1.611, 2.867] | -3.922 [-4.675, -3.217] | 323 → 413 | 118 / 356 / 1 |
| Whisper large-v3 / MAI-Transcribe 2 | 1.017 [0.897, 1.145] → 0.143 [0.097, 0.198] | 8.128 [7.097, 9.216] → 1.126 [0.757, 1.539] | -7.002 [-7.993, -6.069] | 259 → 434 | 205 / 268 / 2 |
| Whisper large-v3 / Voxtral Small Transcribe 3 | 0.949 [0.840, 1.065] → 0.137 [0.093, 0.185] | 7.660 [6.693, 8.667] → 1.100 [0.737, 1.500] | -6.560 [-7.495, -5.646] | 262 → 434 | 202 / 270 / 3 |
| MAI-Transcribe 2 / Voxtral Small Transcribe 3 | 0.331 [0.259, 0.411] → 0.093 [0.055, 0.135] | 2.652 [2.072, 3.282] → 0.743 [0.424, 1.113] | -1.910 [-2.503, -1.352] | 389 → 450 | 68 / 400 / 7 |

## Remaining dev: six model pairs (n=117)

This is the same 117-query evaluation cohort used in the prior calibration-transfer analysis. It has already been inspected; this edit-distance follow-up is descriptive, not a new holdout.

### Characters

| Model pair | Mean edits: raw → normalized | Mean NED (%): raw → normalized | Paired NED Δ (pp), 95% CI | Zero queries: raw → normalized | NED decrease / tie / increase |
| --- | --- | --- | --- | --- | --- |
| GLM-ASR-Nano / Whisper large-v3 | 2.222 [1.709, 2.786] → 0.641 [0.265, 1.094] | 2.750 [2.071, 3.496] → 0.861 [0.349, 1.504] | -1.890 [-2.274, -1.512] | 47 → 100 | 65 / 48 / 4 |
| GLM-ASR-Nano / MAI-Transcribe 2 | 0.974 [0.564, 1.444] → 0.573 [0.214, 1.026] | 1.274 [0.716, 1.933] → 0.783 [0.284, 1.415] | -0.491 [-0.727, -0.289] | 83 → 103 | 26 / 84 / 7 |
| GLM-ASR-Nano / Voxtral Small Transcribe 3 | 1.077 [0.650, 1.573] → 0.581 [0.214, 1.051] | 1.384 [0.819, 2.075] → 0.762 [0.263, 1.394] | -0.621 [-0.884, -0.389] | 80 → 103 | 30 / 81 / 6 |
| Whisper large-v3 / MAI-Transcribe 2 | 1.556 [1.239, 1.872] → 0.137 [0.026, 0.291] | 1.912 [1.513, 2.329] → 0.198 [0.046, 0.405] | -1.713 [-2.087, -1.349] | 55 → 111 | 59 / 55 / 3 |
| Whisper large-v3 / Voxtral Small Transcribe 3 | 1.513 [1.179, 1.855] → 0.162 [0.026, 0.342] | 1.842 [1.434, 2.265] → 0.192 [0.036, 0.392] | -1.651 [-2.013, -1.305] | 60 → 111 | 56 / 60 / 1 |
| MAI-Transcribe 2 / Voxtral Small Transcribe 3 | 0.376 [0.214, 0.564] → 0.094 [0.009, 0.231] | 0.467 [0.256, 0.698] → 0.115 [0.007, 0.273] | -0.352 [-0.559, -0.172] | 97 → 113 | 16 / 97 / 4 |
### Words

| Model pair | Mean edits: raw → normalized | Mean NED (%): raw → normalized | Paired NED Δ (pp), 95% CI | Zero queries: raw → normalized | NED decrease / tie / increase |
| --- | --- | --- | --- | --- | --- |
| GLM-ASR-Nano / Whisper large-v3 | 1.436 [1.162, 1.718] → 0.239 [0.128, 0.359] | 10.579 [8.473, 12.760] → 1.843 [0.979, 2.822] | -8.736 [-10.674, -6.891] | 48 → 100 | 63 / 54 / 0 |
| GLM-ASR-Nano / MAI-Transcribe 2 | 0.684 [0.470, 0.915] → 0.214 [0.111, 0.333] | 4.730 [3.256, 6.305] → 1.560 [0.785, 2.451] | -3.170 [-4.441, -2.000] | 83 → 103 | 25 / 92 / 0 |
| GLM-ASR-Nano / Voxtral Small Transcribe 3 | 0.752 [0.530, 0.991] → 0.197 [0.103, 0.308] | 5.279 [3.725, 6.978] → 1.417 [0.677, 2.268] | -3.862 [-5.248, -2.534] | 80 → 103 | 30 / 87 / 0 |
| Whisper large-v3 / MAI-Transcribe 2 | 1.034 [0.812, 1.256] → 0.068 [0.017, 0.128] | 8.119 [6.248, 10.102] → 0.634 [0.142, 1.267] | -7.485 [-9.343, -5.770] | 58 → 111 | 56 / 60 / 1 |
| Whisper large-v3 / Voxtral Small Transcribe 3 | 0.974 [0.769, 1.179] → 0.068 [0.017, 0.128] | 7.752 [5.923, 9.711] → 0.548 [0.134, 1.130] | -7.203 [-9.005, -5.532] | 62 → 111 | 55 / 62 / 0 |
| MAI-Transcribe 2 / Voxtral Small Transcribe 3 | 0.299 [0.171, 0.436] → 0.051 [0.009, 0.111] | 2.118 [1.193, 3.136] → 0.330 [0.041, 0.704] | -1.789 [-2.806, -0.891] | 97 → 113 | 16 / 99 / 2 |

## Definitions and interpretation

Both metrics use unit-cost Levenshtein insertions, deletions and substitutions (each cost 1). Character distance operates on Python Unicode code points, preserving raw case, punctuation and whitespace. Word distance operates on `text.split()` sequences: case and punctuation remain significant, while pure whitespace differences disappear. No extra processor or raw-text cleanup is used. The implementation reuses [RapidFuzz's Levenshtein API](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html), pinned to [RapidFuzz 3.14.6](https://pypi.org/project/RapidFuzz/3.14.6/).

For each query and model pair, `NED = distance / max(left_length, right_length)` in the selected unit. This symmetric ratio is 0 for two empty sequences and 1 when only one is empty. Reported NED is the mean of per-query ratios, not total edits divided by total length. Normalization changes lengths and token boundaries, so absolute-edit changes and NED changes can differ. The NED change counts use a tie tolerance of `abs(delta) <= 1e-12`.

A pair's zero count measures equality in that unit. For the overall row it requires zero distance for all six pairs. Character zero therefore means all four saved strings are exactly identical; word zero only means identical whitespace-token sequences and can hide spacing differences.

These are **reference-free disagreement measures, not WER/CER**: there is no independent reference transcript in this analysis. Shared wrong words can yield zero distance, and a small edit can alter negation or an important entity. Reduced disagreement after the frozen formatting rule does not demonstrate lower transcription error, semantic equivalence, higher evidence recall, lower context cost or improved downstream answers. Those retrieval questions remain in the [full ASR/retrieval results](asr-rag-full-results.md) and [matched-recall analysis](asr-matched-curves-results.md).

## Paired uncertainty and cohort limits

Each cohort uses **5,000 query-level bootstrap replicates**, percentile 95% intervals, base seed **20261117** plus its saved cohort-order index. Within each replicate, the same sampled query indices apply to all six pairs, both units and both text conditions. Paired normalized-minus-raw intervals use those shared replicate means. The six pairs are not independent observations; the overall interval resamples queries, not 2,850 pair rows. Intervals are pointwise and unadjusted for multiple comparisons. They reflect query resampling conditional on the fixed saved transcripts, excluding ASR generation and provider variability.

The six cohorts overlap: all 475 = practice 350 + dev 125; dev 125 = pilot 8 + remaining dev 117; remaining all 467 excludes the pilot. They are not six independent replications. The normalization rule was developed after examining the pilot, and all-query results have already been inspected. The pilot intervals are especially limited by n=8. Bootstrap uncertainty does not account for shared ASR biases, changes to hosted model versions, or earlier analytical choices.

## Provenance and independent implementation check

- Analysis Slurm job: `3726982`.
- Independent audit Slurm job: `3726994`; status **pass**.
- Analysis Git commit: `ada32e3d87b12a86d50d3885bf0bf7d22ff3038b`.
- Frozen normalization policy: `acronym-punctuation-v1`.
- Frozen normalizer SHA256: `fff6993b2f1ae85db3a65a6c645a6e2ac3ad0ec415eb82af3f73a763de8e9dd2`.
- Audit checked **1,900** mappings, **11,400** character/word distance values and **84** aggregate rows, with **5,000** bootstrap replicates per cohort.

The audit reconstructs aggregates and intervals from saved inputs and cross-checks distances against RapidFuzz's separate Python implementation. This is an implementation check using the same library project, not an independent scientific replication. Original full-run source hashes remain bound in the audit. The [frozen edit-distance protocol](asr-edit-distance-protocol.md) describes the definitions and source checks.

| Audited analysis input | SHA256 |
| --- | --- |
| `analysis-config.json` | `f433e5939c85f8c315f4d8fe3e4f2435b2841b93ff6f4dd048a71af286a09ab9` |
| `bootstrap-values.npz` | `396b82f47c8718e397e82615e7a83f81a64b1d119beec29138e1b89713034e39` |
| `completion.json` | `ba33c514eb3b487c3ea99af3acb646cb3014ce6786aeca8395cfc3fb43bb24ce` |
| `distances.npz` | `9460b54dd5cb5343060e05cb7bbbd6deadb73efec0c508898b6a43ff265db06e` |
| `summary.json` | `8b873b8f32a905e75e18c2e339312af47efd5ce4b7fbeeba94c0e056adfa2dc8` |

| Original full-run evidence | SHA256 |
| --- | --- |
| `audit.json` | `f97c67d40f3f86ad7d79f83a61a0bb4ed2c431a98edf8a3cdbdf65c0b3f9e570` |
| `config.json` | `08e26c97c6447d7147e971b5feb97e3ffe5ef41490ff7a1c11fca4af335ce43a` |
| `summary.json` | `618081a6298107aaea5322bfd09632fc00f993f1867f6fbb58e9d7b74c180654` |
| `transcript-mappings.json` | `41cd9c58959709959872ad8aee09182e591601cc283d28fb3c569da72a983c3f` |

## Aggregate artifacts and remaining cohorts

[All 84 aggregate rows as CSV](asr-edit-distance.csv) include unrounded estimates and intervals for absolute edits and NED. The CSV contains no transcript text or query IDs. Per-query distances, transcripts and bootstrap query indices remain private.

<details>
<summary>Remaining cohort rows (including the remaining-dev overall values)</summary>

### remaining_dev (n=117)

| Model pair | Unit | Mean edits: raw → normalized | Mean NED (%): raw → normalized | Paired NED Δ (pp), 95% CI | Zero queries: raw → normalized | NED decrease / tie / increase |
| --- | --- | --- | --- | --- | --- | --- |
| Overall: six-pair query mean | characters | 1.286 [1.003, 1.584] → 0.365 [0.167, 0.605] | 1.605 [1.239, 2.015] → 0.485 [0.223, 0.814] | -1.120 [-1.342, -0.901] | 43 → 98 | 71 / 44 / 2 |
| Overall: six-pair query mean | words | 0.863 [0.704, 1.027] → 0.140 [0.077, 0.208] | 6.429 [5.202, 7.693] → 1.055 [0.590, 1.592] | -5.374 [-6.480, -4.298] | 43 → 98 | 70 / 47 / 0 |

### practice (n=350)

| Model pair | Unit | Mean edits: raw → normalized | Mean NED (%): raw → normalized | Paired NED Δ (pp), 95% CI | Zero queries: raw → normalized | NED decrease / tie / increase |
| --- | --- | --- | --- | --- | --- | --- |
| Overall: six-pair query mean | characters | 1.680 [1.356, 2.037] → 0.801 [0.537, 1.105] | 2.140 [1.756, 2.567] → 1.033 [0.705, 1.405] | -1.107 [-1.268, -0.955] | 149 → 284 | 186 / 149 / 15 |
| GLM-ASR-Nano / Whisper large-v3 | characters | 2.777 [2.194, 3.426] → 1.397 [0.891, 1.986] | 3.438 [2.756, 4.188] → 1.766 [1.147, 2.475] | -1.671 [-1.903, -1.438] | 164 → 296 | 168 / 164 / 18 |
| GLM-ASR-Nano / MAI-Transcribe 2 | characters | 1.949 [1.409, 2.563] → 1.414 [0.917, 1.989] | 2.451 [1.810, 3.179] → 1.784 [1.164, 2.487] | -0.667 [-0.836, -0.509] | 232 → 298 | 90 / 232 / 28 |
| GLM-ASR-Nano / Voxtral Small Transcribe 3 | characters | 1.849 [1.306, 2.469] → 1.366 [0.863, 1.946] | 2.304 [1.672, 3.025] → 1.736 [1.123, 2.441] | -0.568 [-0.726, -0.420] | 239 → 302 | 83 / 239 / 28 |
| Whisper large-v3 / MAI-Transcribe 2 | characters | 1.551 [1.320, 1.789] → 0.257 [0.166, 0.357] | 2.014 [1.706, 2.331] → 0.347 [0.220, 0.485] | -1.667 [-1.907, -1.424] | 190 → 315 | 156 / 190 / 4 |
| Whisper large-v3 / Voxtral Small Transcribe 3 | characters | 1.477 [1.251, 1.706] → 0.243 [0.157, 0.334] | 1.937 [1.637, 2.248] → 0.361 [0.226, 0.508] | -1.577 [-1.821, -1.344] | 187 → 315 | 154 / 187 / 9 |
| MAI-Transcribe 2 / Voxtral Small Transcribe 3 | characters | 0.477 [0.351, 0.614] → 0.129 [0.071, 0.197] | 0.697 [0.507, 0.910] → 0.205 [0.108, 0.322] | -0.492 [-0.658, -0.344] | 285 → 329 | 53 / 285 / 12 |
| Overall: six-pair query mean | words | 0.930 [0.798, 1.072] → 0.260 [0.190, 0.339] | 7.023 [6.078, 7.999] → 1.896 [1.403, 2.440] | -5.127 [-5.899, -4.382] | 154 → 284 | 177 / 171 / 2 |
| GLM-ASR-Nano / Whisper large-v3 | words | 1.443 [1.229, 1.671] → 0.389 [0.277, 0.520] | 10.500 [9.066, 11.969] → 2.717 [1.947, 3.585] | -7.783 [-8.994, -6.607] | 169 → 296 | 159 / 190 / 1 |
| GLM-ASR-Nano / MAI-Transcribe 2 | words | 0.937 [0.751, 1.137] → 0.374 [0.269, 0.494] | 6.671 [5.464, 7.916] → 2.624 [1.902, 3.425] | -4.047 [-4.955, -3.188] | 232 → 298 | 89 / 259 / 2 |
| GLM-ASR-Nano / Voxtral Small Transcribe 3 | words | 0.903 [0.711, 1.106] → 0.351 [0.246, 0.474] | 6.350 [5.135, 7.590] → 2.515 [1.752, 3.352] | -3.835 [-4.703, -2.984] | 239 → 302 | 84 / 265 / 1 |
| Whisper large-v3 / MAI-Transcribe 2 | words | 1.011 [0.863, 1.169] → 0.171 [0.111, 0.240] | 8.141 [6.886, 9.482] → 1.316 [0.859, 1.841] | -6.825 [-8.023, -5.701] | 197 → 315 | 145 / 204 / 1 |
| Whisper large-v3 / Voxtral Small Transcribe 3 | words | 0.940 [0.803, 1.083] → 0.163 [0.106, 0.223] | 7.629 [6.451, 8.903] → 1.310 [0.845, 1.823] | -6.320 [-7.464, -5.231] | 196 → 315 | 143 / 204 / 3 |
| MAI-Transcribe 2 / Voxtral Small Transcribe 3 | words | 0.343 [0.254, 0.440] → 0.109 [0.063, 0.163] | 2.847 [2.144, 3.618] → 0.898 [0.492, 1.389] | -1.950 [-2.694, -1.258] | 285 → 329 | 51 / 294 / 5 |

### dev (n=125)

| Model pair | Unit | Mean edits: raw → normalized | Mean NED (%): raw → normalized | Paired NED Δ (pp), 95% CI | Zero queries: raw → normalized | NED decrease / tie / increase |
| --- | --- | --- | --- | --- | --- | --- |
| Overall: six-pair query mean | characters | 1.279 [1.013, 1.563] → 0.341 [0.159, 0.565] | 1.591 [1.242, 1.968] → 0.454 [0.209, 0.755] | -1.137 [-1.351, -0.920] | 46 → 106 | 76 / 47 / 2 |
| GLM-ASR-Nano / Whisper large-v3 | characters | 2.216 [1.736, 2.752] → 0.600 [0.256, 1.032] | 2.735 [2.114, 3.449] → 0.805 [0.340, 1.412] | -1.930 [-2.306, -1.562] | 50 → 108 | 70 / 51 / 4 |
| GLM-ASR-Nano / MAI-Transcribe 2 | characters | 0.944 [0.568, 1.400] → 0.536 [0.200, 0.976] | 1.235 [0.715, 1.863] → 0.733 [0.271, 1.335] | -0.502 [-0.732, -0.307] | 88 → 111 | 29 / 89 / 7 |
| GLM-ASR-Nano / Voxtral Small Transcribe 3 | characters | 1.064 [0.672, 1.536] → 0.544 [0.208, 0.976] | 1.367 [0.827, 2.000] → 0.714 [0.261, 1.311] | -0.653 [-0.909, -0.419] | 84 → 111 | 34 / 85 / 6 |
| Whisper large-v3 / MAI-Transcribe 2 | characters | 1.560 [1.248, 1.864] → 0.128 [0.024, 0.264] | 1.909 [1.515, 2.291] → 0.186 [0.039, 0.367] | -1.723 [-2.074, -1.367] | 59 → 119 | 63 / 59 / 3 |
| Whisper large-v3 / Voxtral Small Transcribe 3 | characters | 1.512 [1.192, 1.832] → 0.152 [0.024, 0.312] | 1.834 [1.445, 2.226] → 0.179 [0.033, 0.360] | -1.655 [-2.008, -1.316] | 64 → 119 | 60 / 64 / 1 |
| MAI-Transcribe 2 / Voxtral Small Transcribe 3 | characters | 0.376 [0.216, 0.552] → 0.088 [0.008, 0.208] | 0.466 [0.263, 0.690] → 0.107 [0.007, 0.244] | -0.359 [-0.557, -0.183] | 104 → 121 | 17 / 104 / 4 |
| Overall: six-pair query mean | words | 0.872 [0.717, 1.029] → 0.131 [0.075, 0.193] | 6.509 [5.321, 7.695] → 0.988 [0.553, 1.475] | -5.521 [-6.577, -4.464] | 46 → 106 | 75 / 50 / 0 |
| GLM-ASR-Nano / Whisper large-v3 | words | 1.456 [1.184, 1.736] → 0.224 [0.120, 0.336] | 10.765 [8.780, 12.809] → 1.725 [0.927, 2.650] | -9.039 [-10.910, -7.251] | 51 → 108 | 68 / 57 / 0 |
| GLM-ASR-Nano / MAI-Transcribe 2 | words | 0.696 [0.488, 0.920] → 0.200 [0.096, 0.312] | 4.854 [3.441, 6.431] → 1.460 [0.721, 2.291] | -3.394 [-4.677, -2.241] | 88 → 111 | 28 / 97 / 0 |
| GLM-ASR-Nano / Voxtral Small Transcribe 3 | words | 0.776 [0.560, 1.008] → 0.184 [0.096, 0.288] | 5.491 [3.960, 7.162] → 1.326 [0.641, 2.129] | -4.165 [-5.575, -2.832] | 84 → 111 | 34 / 91 / 0 |
| Whisper large-v3 / MAI-Transcribe 2 | words | 1.032 [0.816, 1.240] → 0.064 [0.016, 0.120] | 8.091 [6.270, 9.909] → 0.593 [0.153, 1.165] | -7.498 [-9.216, -5.795] | 62 → 119 | 60 / 64 / 1 |
| Whisper large-v3 / Voxtral Small Transcribe 3 | words | 0.976 [0.768, 1.176] → 0.064 [0.016, 0.120] | 7.747 [5.947, 9.577] → 0.513 [0.114, 1.023] | -7.234 [-8.929, -5.575] | 66 → 119 | 59 / 66 / 0 |
| MAI-Transcribe 2 / Voxtral Small Transcribe 3 | words | 0.296 [0.168, 0.432] → 0.048 [0.008, 0.096] | 2.106 [1.209, 3.101] → 0.309 [0.038, 0.651] | -1.797 [-2.797, -0.913] | 104 → 121 | 17 / 106 / 2 |

### pilot (n=8)

| Model pair | Unit | Mean edits: raw → normalized | Mean NED (%): raw → normalized | Paired NED Δ (pp), 95% CI | Zero queries: raw → normalized | NED decrease / tie / increase |
| --- | --- | --- | --- | --- | --- | --- |
| Overall: six-pair query mean | characters | 1.167 [0.375, 1.958] → 0.000 [0.000, 0.000] | 1.387 [0.476, 2.324] → 0.000 [0.000, 0.000] | -1.387 [-2.324, -0.476] | 3 → 8 | 5 / 3 / 0 |
| GLM-ASR-Nano / Whisper large-v3 | characters | 2.125 [0.750, 3.500] → 0.000 [0.000, 0.000] | 2.516 [0.938, 4.069] → 0.000 [0.000, 0.000] | -2.516 [-4.069, -0.938] | 3 → 8 | 5 / 3 / 0 |
| GLM-ASR-Nano / MAI-Transcribe 2 | characters | 0.500 [0.125, 1.000] → 0.000 [0.000, 0.000] | 0.657 [0.149, 1.313] → 0.000 [0.000, 0.000] | -0.657 [-1.313, -0.149] | 5 → 8 | 3 / 5 / 0 |
| GLM-ASR-Nano / Voxtral Small Transcribe 3 | characters | 0.875 [0.250, 1.625] → 0.000 [0.000, 0.000] | 1.114 [0.298, 2.037] → 0.000 [0.000, 0.000] | -1.114 [-2.037, -0.298] | 4 → 8 | 4 / 4 / 0 |
| Whisper large-v3 / MAI-Transcribe 2 | characters | 1.625 [0.500, 2.875] → 0.000 [0.000, 0.000] | 1.865 [0.510, 3.353] → 0.000 [0.000, 0.000] | -1.865 [-3.353, -0.510] | 4 → 8 | 4 / 4 / 0 |
| Whisper large-v3 / Voxtral Small Transcribe 3 | characters | 1.500 [0.500, 2.750] → 0.000 [0.000, 0.000] | 1.716 [0.510, 3.008] → 0.000 [0.000, 0.000] | -1.716 [-3.008, -0.510] | 4 → 8 | 4 / 4 / 0 |
| MAI-Transcribe 2 / Voxtral Small Transcribe 3 | characters | 0.375 [0.000, 1.125] → 0.000 [0.000, 0.000] | 0.457 [0.000, 1.372] → 0.000 [0.000, 0.000] | -0.457 [-1.372, +0.000] | 7 → 8 | 1 / 7 / 0 |
| Overall: six-pair query mean | words | 1.000 [0.396, 1.542] → 0.000 [0.000, 0.000] | 7.672 [3.036, 11.750] → 0.000 [0.000, 0.000] | -7.672 [-11.750, -3.036] | 3 → 8 | 5 / 3 / 0 |
| GLM-ASR-Nano / Whisper large-v3 | words | 1.750 [0.750, 2.625] → 0.000 [0.000, 0.000] | 13.478 [5.625, 20.353] → 0.000 [0.000, 0.000] | -13.478 [-20.353, -5.625] | 3 → 8 | 5 / 3 / 0 |
| GLM-ASR-Nano / MAI-Transcribe 2 | words | 0.875 [0.250, 1.750] → 0.000 [0.000, 0.000] | 6.667 [2.083, 13.333] → 0.000 [0.000, 0.000] | -6.667 [-13.333, -2.083] | 5 → 8 | 3 / 5 / 0 |
| GLM-ASR-Nano / Voxtral Small Transcribe 3 | words | 1.125 [0.250, 1.875] → 0.000 [0.000, 0.000] | 8.590 [2.083, 14.583] → 0.000 [0.000, 0.000] | -8.590 [-14.583, -2.083] | 4 → 8 | 4 / 4 / 0 |
| Whisper large-v3 / MAI-Transcribe 2 | words | 1.000 [0.250, 1.875] → 0.000 [0.000, 0.000] | 7.687 [1.562, 14.423] → 0.000 [0.000, 0.000] | -7.687 [-14.423, -1.562] | 4 → 8 | 4 / 4 / 0 |
| Whisper large-v3 / Voxtral Small Transcribe 3 | words | 1.000 [0.250, 1.875] → 0.000 [0.000, 0.000] | 7.687 [1.562, 14.423] → 0.000 [0.000, 0.000] | -7.687 [-14.423, -1.562] | 4 → 8 | 4 / 4 / 0 |
| MAI-Transcribe 2 / Voxtral Small Transcribe 3 | words | 0.250 [0.000, 0.750] → 0.000 [0.000, 0.000] | 1.923 [0.000, 5.769] → 0.000 [0.000, 0.000] | -1.923 [-5.769, +0.000] | 7 → 8 | 1 / 7 / 0 |

### remaining_all (n=467)

| Model pair | Unit | Mean edits: raw → normalized | Mean NED (%): raw → normalized | Paired NED Δ (pp), 95% CI | Zero queries: raw → normalized | NED decrease / tie / increase |
| --- | --- | --- | --- | --- | --- | --- |
| Overall: six-pair query mean | characters | 1.581 [1.335, 1.859] → 0.692 [0.481, 0.924] | 2.006 [1.695, 2.333] → 0.896 [0.632, 1.177] | -1.110 [-1.237, -0.985] | 192 → 382 | 257 / 193 / 17 |
| GLM-ASR-Nano / Whisper large-v3 | characters | 2.638 [2.191, 3.135] → 1.208 [0.803, 1.657] | 3.266 [2.725, 3.850] → 1.539 [1.036, 2.078] | -1.726 [-1.925, -1.528] | 211 → 396 | 233 / 212 / 22 |
| GLM-ASR-Nano / MAI-Transcribe 2 | characters | 1.704 [1.274, 2.176] → 1.203 [0.803, 1.642] | 2.156 [1.625, 2.705] → 1.533 [1.038, 2.067] | -0.623 [-0.763, -0.493] | 315 → 401 | 116 / 316 / 35 |
| GLM-ASR-Nano / Voxtral Small Transcribe 3 | characters | 1.655 [1.244, 2.124] → 1.169 [0.771, 1.608] | 2.074 [1.568, 2.624] → 1.492 [1.002, 2.029] | -0.581 [-0.718, -0.452] | 319 → 405 | 113 / 320 / 34 |
| Whisper large-v3 / MAI-Transcribe 2 | characters | 1.552 [1.364, 1.743] → 0.227 [0.152, 0.313] | 1.988 [1.740, 2.240] → 0.310 [0.205, 0.426] | -1.678 [-1.879, -1.482] | 245 → 426 | 215 / 245 / 7 |
| Whisper large-v3 / Voxtral Small Transcribe 3 | characters | 1.486 [1.298, 1.675] → 0.223 [0.148, 0.306] | 1.914 [1.659, 2.163] → 0.318 [0.208, 0.441] | -1.595 [-1.795, -1.397] | 247 → 426 | 210 / 247 / 10 |
| MAI-Transcribe 2 / Voxtral Small Transcribe 3 | characters | 0.452 [0.349, 0.565] → 0.120 [0.071, 0.180] | 0.640 [0.489, 0.808] → 0.182 [0.104, 0.278] | -0.457 [-0.589, -0.334] | 382 → 442 | 69 / 382 / 16 |
| Overall: six-pair query mean | words | 0.913 [0.805, 1.027] → 0.229 [0.175, 0.291] | 6.874 [6.090, 7.651] → 1.686 [1.292, 2.113] | -5.189 [-5.814, -4.559] | 197 → 382 | 247 / 218 / 2 |
| GLM-ASR-Nano / Whisper large-v3 | words | 1.441 [1.268, 1.625] → 0.351 [0.259, 0.450] | 10.520 [9.323, 11.718] → 2.498 [1.845, 3.162] | -8.022 [-9.015, -7.018] | 217 → 396 | 222 / 244 / 1 |
| GLM-ASR-Nano / MAI-Transcribe 2 | words | 0.874 [0.724, 1.032] → 0.334 [0.246, 0.426] | 6.185 [5.209, 7.197] → 2.357 [1.758, 2.980] | -3.828 [-4.568, -3.108] | 315 → 401 | 114 / 351 / 2 |
| GLM-ASR-Nano / Voxtral Small Transcribe 3 | words | 0.865 [0.713, 1.028] → 0.313 [0.227, 0.405] | 6.081 [5.114, 7.117] → 2.240 [1.638, 2.891] | -3.842 [-4.601, -3.089] | 319 → 405 | 114 / 352 / 1 |
| Whisper large-v3 / MAI-Transcribe 2 | words | 1.017 [0.891, 1.148] → 0.146 [0.096, 0.199] | 8.135 [7.057, 9.223] → 1.145 [0.767, 1.571] | -6.990 [-7.967, -6.033] | 255 → 426 | 201 / 264 / 2 |
| Whisper large-v3 / Voxtral Small Transcribe 3 | words | 0.949 [0.831, 1.064] → 0.139 [0.094, 0.188] | 7.660 [6.640, 8.690] → 1.119 [0.748, 1.525] | -6.541 [-7.478, -5.623] | 258 → 426 | 198 / 266 / 3 |
| MAI-Transcribe 2 / Voxtral Small Transcribe 3 | words | 0.332 [0.257, 0.411] → 0.094 [0.058, 0.137] | 2.665 [2.074, 3.305] → 0.755 [0.443, 1.145] | -1.909 [-2.491, -1.339] | 382 → 442 | 67 / 393 / 7 |

</details>
