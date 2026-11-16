# Formatting-normalized rerun of the eight-query ASR pilot

This is a post-hoc diagnostic requested after inspection of the raw pilot. The combined formatting rule is frozen before computing the new retrieval metrics. It is not an independent confirmatory evaluation or an ablation separating acronym, punctuation and case effects.

## Frozen scope

- Reuse the exact 32 saved successful transcripts (four ASR models × the same eight dev recordings) from `asr-four-model-pilot-20261116`; make no ASR or API requests. Preserve originals and save every raw-to-normalized mapping and input file hash.
- Apply `acronym-punctuation-v1` identically to every model and query: NFKC; canonical apostrophes; merge bounded dotted ASCII initialisms and whitespace-separated uppercase single letters before Unicode casefold; replace punctuation with spaces, including hyphens; collapse whitespace.
- Retain letters/accents, symbols, bracketed content, word-internal apostrophes, digit-internal dot/comma separators, leading numeric minus signs and numeric percent suffixes. No synonym, number, contraction, spelling or abbreviation expansion; no dictionary keyed by query/model/qrels. This narrow rule is not a universal ambiguity resolver for all initialisms.
- The canonical query form is lower case, with ordinary sentence punctuation removed. Both acronym and punctuation normalization, case normalization and spacing are applied together. They may change the absolute GLM result too.
- Keep the original `enc_a`, CPU device/dtype behavior, full 24,000-vector corpus, float32 ranking, stable corpus-row tie break, original qrels, top-5, cutoff >=0.8925 with top-1 fallback/no cap, pinned Mistral tokenizer and document-only token measure.

## Controls and outputs

Run inside a CPU Slurm allocation using the existing RAG environment. Recompute the original 64 raw retrieval cells first and require exact selected IDs/counts/recalls/token counts/fallback flags to agree with the saved pilot; log score drift. Fail rather than silently treating an environment change as a normalization effect. Then independently encode all 32 normalized transcripts and compute 64 normalized cells. Do not enforce textual equality across models as a success condition.

Freeze policy text/source hashes, input hashes, selection/configuration, environment/import versions and git commit before new metrics. Save normalized strings/hashes, per-query distinct text counts, raw-control checks, complete per-query metrics, means, candidate-minus-GLM contrasts and per-arm raw-to-normalized changes. Recount recalls and document tokens from original qrels/corpus in an independent CPU Slurm audit. Report whether normalization makes the four inputs equal on each query, rather than only whether their means match.

If equal normalized inputs produce equal deterministic retrieval outputs, that is an expected consequence of the controlled input transformation. It does not establish equal general ASR quality, equal WER or irrelevance of ASR choice outside these eight reused dev examples. No answer-quality or inference-latency evaluation is included.

## External reuse decision

The actual environment uses Python 3.11.18 and already contains the required encoder, NumPy and Tokenizers libraries. `pip list/show` was checked; `openai-whisper` and `jiwer` are absent. No dependency is added.

- [Whisper English normalizer](https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py) also rewrites numbers, contractions, titles and spellings; [the basic normalizer](https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py) removes bracketed content. Those policies exceed this diagnostic's scope.
- [JiWER transformations](https://jitsi.github.io/jiwer/reference/transforms/) offer useful generic operations but `RemovePunctuation` deletes punctuation, joining hyphenated words, and custom acronym/protection logic would still be required. Installing its WER/edit-distance dependencies solely for these formatting rules has no benefit here.
- Python 3.11's documented [`unicodedata.normalize/category`](https://docs.python.org/3.11/library/unicodedata.html) and [`re`](https://docs.python.org/3.11/library/re.html) cover the narrow rules directly. Their actual module paths/version and a minimal API check are recorded in Slurm; targeted equivalence, content-preservation and idempotence tests must pass before the run.
