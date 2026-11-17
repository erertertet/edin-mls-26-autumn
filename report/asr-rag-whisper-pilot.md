# Four-model ASR × retrieval pilot

This protocol supersedes the original two-model Whisper pilot. The first run is an eight-query pilot on Edinburgh Slurm compute nodes, following `AGENTS.md`.

## Research question

> How does ASR model choice affect evidence recall and retrieved-context token cost under fixed top-5 and absolute similarity-cutoff retrieval on the same spoken queries?

The comparison tests the effect of changing the upstream transcript with retrieval parameters held fixed. It does not assume that the ASR leaderboard order predicts downstream retrieval quality.

## Frozen design

| ASR arm | Execution / model ID |
| --- | --- |
| GLM-ASR-Nano | Existing course ASR adapter on the Slurm GPU; record the GLM snapshot and `ASR_CUDA` setting |
| Whisper large-v3 | OpenRouter `openai/whisper-large-v3` |
| MAI-Transcribe 2 | OpenRouter `microsoft/mai-transcribe-2` |
| Voxtral Small Transcribe 3 | OpenRouter `mistralai/voxtral-small-transcribe-3-0` |

Each transcript is evaluated with **fixed top-5** and **absolute cosine similarity >= 0.8925**, producing eight ASR/policy settings. The encoder, corpus, qrels, audio and token counter are common to all settings. Do not retune the cutoff for an ASR model or select models/queries from observed retrieval results.

### Sampling and provenance

1. Use the course dev split. Eligible queries must have an existing audio file and at least one positive qrel. Check that all positive qrel record IDs exist in the common corpus.
2. Sort eligible query IDs and select eight without replacement using `random.Random(26).sample(sorted_ids, 8)`.
3. Before transcription or retrieval, save the ordered selection, eligible-ID list/hash, audio paths relative to the dataset, audio SHA256, duration and WAV properties. Save the qrels, manifest, corpus, corpus-vector and record-ID mapping hashes.
4. Freeze the manifest on disk. A resumed run must match it exactly. Do not replace failed queries with convenient alternatives.
5. The original dev data informed the existing cutoff. This is an exploratory pilot on reused dev data, not an independent confirmatory test.

### Transcription

- Run all experiment code, transcription and retrieval inside the Slurm allocation. Use the login node only for login, synchronization and environment preparation. Never use the local computer's GPU.
- Reuse the course `ASRAdapter` for GLM. Prefer a fresh transcription of the eight recordings; a cache is acceptable only when its provenance binds the transcript to the same audio SHA256, model and decoding settings. Mark cached timings separately.
- Send the exact same audio bytes to the three OpenRouter models through `POST /api/v1/audio/transcriptions`. Use transcription rather than translation. Keep the language setting common; for the English course query recordings use `language=en`.
- Request plain transcripts, without domain hints, qrels, reference answers, custom vocabularies, diarization or audio-event tags. Record every effective decoding/formatting option. Request verbatim output where the provider supports it; record unsupported options instead of claiming equivalent decoding implementations.
- Record requested model IDs, the model/endpoint catalogue snapshot, returned provider/version/request IDs when available, raw response JSON, transcript, wall time and usage/cost. A requested slug alone is not proof of an immutable model revision. Record unavailable provenance fields as unavailable.
- Use one saved transcript per audio/model for both retrieval policies. Randomize the 24 external transcription tasks with a fixed recorded seed; no generation or LLM judging is part of this pilot.
- Save completed requests before continuing and reuse them on resume. Disable hidden automatic POST retries; explicitly log any later retry. A timed-out request may still have been billed.

### Retrieval

1. Encode every transcript with the same supplied `enc_a` using the existing `LocalEncoder`; record model files and encoding settings. Do not reuse GLM query vectors for another model's transcript.
2. Reuse the supplied `ExactDenseRetriever` corpus vectors and ID mapping. Compute the full exact cosine ranking over all **24,000** records, as in `prepare_uncapped_rankings.py`.
3. Use stable descending similarity order with corpus row order as the tie breaker.
4. Fixed top-5 returns the first five records. The cutoff returns every record with score >= 0.8925, with **top-1 fallback** if none qualify. No candidate-count cap or silent truncation applies.
5. Score every setting against the same original query's positive qrels, not against a question relabelled from its new transcript.

## Metrics and accounting

For each query, ASR and retrieval rule:

- **Evidence recall:** number of retrieved positive-qrel records / total positive-qrel records for that original query.
- **Context token cost:** sum of the token counts of the individual retrieved document texts, using one pinned Mistral Small 4 tokenizer for every arm, `add_special_tokens=False`. Reuse the report tokenizer metadata and revision `a11f36bebf709121056b1dbcc943d1c6afbe494d`; verify its repository/path and file hash before running.
- This additive document-text measure excludes the question transcript, system text, labels, separators and chat framing. It is not a measured full LLM prompt count or an API bill.
- Also retain selected record IDs/scores, relevant hits, returned-document count, top similarity, cutoff fallback, transcript token count, encoding/retrieval times and ASR service cost.
- Keep GLM GPU inference timings separate from external API wall times, which include network/provider overhead. Cached outputs have no newly measured inference latency.

Use the same complete four-model query intersection for the primary eight-row summary. Report failures, missing cells and the number of complete paired queries. Do not substitute zero recall for failed transcription. If fewer than eight queries complete, mark the pilot partial.

| ASR | Retrieval | Complete paired queries | Mean recall (%) | Recall change vs GLM (pp) | Mean context tokens | Token change vs GLM | Mean documents |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-ASR-Nano | top-5 | | | 0 | | 0 | |
| Whisper large-v3 | top-5 | | | | | | |
| MAI-Transcribe 2 | top-5 | | | | | | |
| Voxtral Small Transcribe 3 | top-5 | | | | | | |
| GLM-ASR-Nano | cutoff 0.8925 | | | 0 | | 0 | |
| Whisper large-v3 | cutoff 0.8925 | | | | | | |
| MAI-Transcribe 2 | cutoff 0.8925 | | | | | | |
| Voxtral Small Transcribe 3 | cutoff 0.8925 | | | | | | |

Changes are **candidate ASR minus GLM**, paired by query and retrieval rule. Positive recall change means more original labelled evidence was recovered; positive token change means more context tokens. Also report each candidate's difference between its cutoff recall change and top-5 recall change, retaining the sign.

## Credentials and spend

Read `OPENROUTER_API_KEY` from the environment or a private dotenv file. Keep it out of Git, command-line arguments, logs and saved configuration. A local coordinator may transfer only this credential over the existing SSH connection to a private cluster file with mode 600. Do not replace the existing ELM credentials.

Use a conservative **USD 1 pilot guard** and the current duration-based prices to preflight the 24 planned external requests. Save actual usage and request IDs separately. The guard is not a provider account spending cap; failed/ambiguous requests may have charges. Do not launch answer generation, judging, model sweeps or the stopped earlier studies.

## Evidence and validation

Save a run directory under ignored `rag/results/` containing frozen sampling/configuration, dependency versions/import paths, source hashes, tokenizer metadata, environment, Slurm job/node/GPU details, command, transcripts, raw API responses, per-query retrieval rows, summary and a completion/error manifest. Restrict private raw outputs appropriately.

Before paid calls, verify the actual cluster interpreters and installed packages, exercise the existing encoder/tokenizer/retrieval APIs in the Slurm allocation, check outbound access to OpenRouter, validate request model IDs and check estimated spend. Record exact versions and consult their official documentation. Reuse the existing data loader, encoder, retriever, ASR adapter and tokenizers rather than replacing their algorithms.

The pilot is complete only when all four models have transcripts for the frozen eight recordings and all 64 retrieval cells are validated. Inspect per-query changes, including numbers/proper names and threshold crossings. Transcript disagreement alone does not establish which ASR is correct without an independent reference. Eight queries cannot establish general model superiority.

## Reproduction

The runner is `rag/extensions/asr_retrieval_pilot.py`; `rag/extensions/run_asr_retrieval_pilot.sbatch` runs targeted validation, builds the existing CUDA library on the allocated GPU and executes the pilot. Use the separate RAG environment with `rag/extensions/asr-pilot-constraints.txt`; preserve the supplied ASR environment. The runner reuses existing `CourseData`, `LocalEncoder`, `ExactDenseRetriever`, `ASRAdapter`, `recall_at_k`, Hugging Face Tokenizers and the declared OpenAI SDK. The new code only composes these components and records the paired experiment.

After course assets and the pinned tokenizer have been prepared on the login node, submit from the clean PR checkout root:

```bash
sbatch rag/extensions/run_asr_retrieval_pilot.sbatch
```

The default output directory is `rag/results/asr-four-model-pilot-20261116/`. Keep the private cluster credential in `$HOME/.config/edin-mls/openrouter.env` with mode 600. It contains only `OPENROUTER_API_KEY`; the runner does not change ELM configuration. The program has no automatic paid POST retries.

## References

- [Course RAG setup](../rag/README.md)
- [Existing full-corpus ranking](../rag/extensions/prepare_uncapped_rankings.py)
- [Current cutoff results and metric boundaries](rag-threshold-controls-results.md)
- [OpenRouter speech-to-text contract](https://openrouter.ai/docs/guides/overview/multimodal/stt)
- [OpenRouter live transcription catalogue](https://openrouter.ai/api/v1/models?output_modalities=transcription)
