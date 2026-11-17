# Four-ASR transcript disagreement by edit distance

This saved-data follow-up quantifies the magnitude of transcript differences, extending the earlier exact-string identity counts. It uses all **475** supplied audio queries and the existing **1,900** four-model raw/normalized mappings. It makes **zero new ASR, encoder, generation, judging or external API calls**. Neither transcripts nor the frozen normalization rule are changed.

## Frozen metrics

Compare all six unordered pairs among GLM-ASR-Nano, Whisper large-v3, MAI-Transcribe 2 and Voxtral Small Transcribe 3, on both saved text conditions. Do not choose one ASR output as ground truth.

- **Character distance:** unit-cost Levenshtein insertions, deletions and substitutions over Python Unicode code points. Raw case, punctuation and whitespace remain significant. No extra preprocessing is applied.
- **Word distance:** the same unit-cost sequence distance over `text.split()` tokens. Case and punctuation remain significant. Whitespace-only differences disappear in this metric; they remain visible to character distance.
- **Normalized edit distance (NED):** `distance / max(len(left), len(right))` in the relevant unit, in [0,1]. Both sequences empty gives 0; one empty gives 1. Report the mean of per-query ratios, not a pooled edit/length ratio. A decrease can differ from the absolute-distance change because normalization changes lengths and token boundaries.
- Compare each pair's original raw strings with its saved `acronym-punctuation-v1` normalized strings. Do not lowercase, trim, normalize or tokenize the raw text beyond the explicitly defined whitespace split for the word metric.
- The overall value first averages all six pair distances within each query, then averages queries. Preserve zero-distance queries. Count raw/normalized zero distances and NED decreases, ties (absolute delta <=1e-12) and increases. Overall zero distance means all six pair distances are zero in that unit.

These are **reference-free inter-model disagreement** measures, not WER/CER or semantic distance. Convergence can include shared wrong content and does not demonstrate better recognition or retrieval. The same four-model exact-character identity counts must reproduce **195 raw and 390 normalized** queries; the remaining 85 are text disagreements, not retrieval disagreements.

## Cohorts and uncertainty

Keep the existing `all` (475), `practice` (350), `dev` (125), `pilot` (8), `remaining_dev` (117) and `remaining_all` (467) query sets. Report all six model pairs and the query-wise overall value for both units, giving **84 aggregate rows**.

For each cohort, use **5,000 query-level bootstrap replicates**, base seed **20261117** plus its saved cohort-order index, percentile 95% intervals. One shared query-index vector within a replicate applies to all model pairs, conditions and metrics. Derive normalized-minus-raw intervals from the same replicate means, not independent raw/normalized intervals. The six model pairs are not independent observations. Preserve the actual indices and distributions privately. Intervals are pointwise, unadjusted for multiplicity. Cohorts overlap, and the pilot is small; none is a new independent confirmatory experiment. All queries have been inspected in prior analyses.

## Reuse, dependencies and execution

Before implementation, the real cluster environment, `rag/requirements.txt` and existing analysis constraints were checked with `pip list/show`, import paths and signatures. Python is 3.11.18, NumPy 2.4.6 and SciPy 1.17.1. No edit-distance package was installed. Reuse NumPy/SciPy for aggregation, query resampling and quantiles. Add only **RapidFuzz 3.14.6**, using its Linux CPython 3.11 binary wheel without dependencies; the current environment passes `pip check`.

RapidFuzz's official PyPI metadata, v3.14.6 release, source and API documentation were checked. It is maintained, MIT licensed and directly supports both Unicode strings and generic token sequences with no mandatory runtime dependency. The inspected interpreter had none of RapidFuzz, Levenshtein, editdistance or jiwer installed. RapidFuzz directly meets this symmetric sequence-distance need; reference-based WER aggregation is unnecessary. Do not implement a bespoke edit-distance algorithm. Use `rapidfuzz.distance.Levenshtein.distance(weights=(1,1,1), processor=None)` and verify the normalized-distance API against the explicit denominator. The independent auditor uses the same project's separate Python implementation as an implementation cross-check, plus independently reconstructed aggregates and saved-bootstrap intervals; it is not an independent scientific replication.

All calculations, API examples, tests and audits run inside Edinburgh CPU-only Slurm allocations. A minimal probe covers substitutions/insertions, empty inputs, token sequences, Unicode, symmetry and the normalized-distance convention. Verify the original audited mapping/config/summary hashes, each source transcript/text hash and the unchanged normalizer SHA256 `fff6993b2f1ae85db3a65a6c645a6e2ac3ad0ec415eb82af3f73a763de8e9dd2`. Retain per-query distances, bootstrap indices and source evidence privately. Publish only code, aggregate tables/CSV and audit metadata to PR5; preserve the retrieval experiments unchanged.

References: [RapidFuzz Levenshtein API](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html), [v3.14.6 source](https://github.com/rapidfuzz/RapidFuzz/blob/v3.14.6/src/rapidfuzz/distance/Levenshtein_py.py), [release](https://github.com/rapidfuzz/RapidFuzz/releases/tag/v3.14.6), [PyPI compatibility and license](https://pypi.org/project/RapidFuzz/3.14.6/), [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
