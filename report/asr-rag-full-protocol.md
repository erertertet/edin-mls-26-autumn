# Frozen full-audio ASR × normalization × retrieval experiment

The user-authorized expansion covers every supplied recording: **350 practice + 125 dev = 475 queries**. Reuse the exact 32 saved pilot transcripts; acquire four-model transcripts for the other 467 recordings. Full expected coverage is **1,900 transcripts and 7,600 retrieval cells**: four ASR arms × raw/normalized × top-5/cutoff × 475.

## Rules fixed before the remaining audio

- Keep `transcript_normalization.py` byte-for-byte identical to the completed eight-query control. SHA256: `fff6993b2f1ae85db3a65a6c645a6e2ac3ad0ec415eb82af3f73a763de8e9dd2`, policy `acronym-punctuation-v1`. Do not add corrections after inspecting new outputs. Preserve raw text, normalized text and both hashes.
- Same GLM-ASR-Nano snapshot `61ba4e0b3309b6656edea3e93e419f7bd5c61957`, `ASR_CUDA=all`, greedy decoding and 100-new-token ceiling. Same three OpenRouter IDs and English/temperature-zero/JSON options; MAI Azure verbatim style, no diarization. Record service catalogue/returned metadata; slugs do not identify immutable provider weights.
- Same original `enc_a`, CPU/dtype behavior, 24,000 corpus vectors and corpus record texts. Re-encode each raw and normalized transcript independently. Rank all records by float32 cosine; stable corpus-row tie break. Do not normalize the corpus or retune the encoder/threshold.
- Fixed top-5 and absolute cutoff >=0.8925, top-1 fallback if no record qualifies, no count cap. Preserve original positive qrels for each query's own split.
- Same pinned tokenizer/revision/hash as the pilot. Token cost is the sum of individual selected document-text token counts without special tokens or prompt scaffolding; it is not billed generation input tokens.

## Coverage, contrasts and missingness

Freeze manifest, audio SHA/duration, corpus/qrels/model/source hashes, selection and all six cohorts before new inference. Validate every manifest audio and qrel; no replacement sampling or convenient exclusion. Report **all 475**, **practice 350**, **dev 125**, **pilot 8**, **remaining dev 117** and **remaining all 467** separately. The pilot is development evidence, not new replication. Remaining recordings were not used to choose this formatting rule, but the public dev data informed the cutoff; these cohorts are not an independent held-out test.

Each cohort's 16 settings must use one common complete query intersection; publish planned coverage, missing cells, failures and partial status. The requested full experiment is complete only at 475/475 four-model transcripts and all 7,600 cells. Empty/non-idempotent normalized output is an explicit failure of the fixed rule, not permission to change it or silently drop a query.

Report macro recall, mean document-text tokens, document counts/fallbacks; paired candidate-ASR-minus-GLM contrasts within each condition/policy; normalized-minus-raw changes for each ASR/policy, including counts of query-level gains/losses/ties. Report raw/normalized text equality and residual differences. Equal normalized inputs imply equal deterministic outputs; this does not show equal general ASR quality. Do not treat a lower-token, lower-recall cutoff as a matched-recall win.

## Execution, caching and spend

All computation, tests, compilation, inference and audits run on Edinburgh Slurm. Reuse the existing ASR adapter, encoder, exact dense retriever, recall metric, Tokenizers and declared OpenAI SDK; no new dependency. Stage assets/model to node-local scratch and remove only this job's scratch at exit. Preserve supplied CUDA kernels and all existing evidence.

Import the 8-query cache byte-for-byte after checking audio, model, GLM snapshot/decoding options and external request options against its frozen configuration. Never regenerate these pilot transcripts to select a favorable realization. Save every new successful response immediately and reuse it on restart. Disable SDK POST retries; permit only explicitly invoked and separately logged retries of failed/ambiguous requests, at most three attempts per audio/model. Store attempt status/request IDs and raw responses; timeouts may have been charged.

Dedicated JSON POST `/api/v1/audio/transcriptions` reuses the verified OpenRouter contract and SDK `client.post(..., cast_to=httpx2.Response, body=...)`. Use Python 3.11's documented `ThreadPoolExecutor` with three workers, one short-lived SDK client per task, for independent API I/O. This avoids inventing a scheduler or assuming one client is thread-safe. Exact request order is recorded in attempt times; seed 26 shuffles the planned tasks. Existing native-transport and common-pair tests plus normalization and generalized-summary checks run in Slurm.

Read the already configured private cluster key; never commit or print it. Preflight estimated new charges are approximately USD 0.2500, conservative reserve USD 0.5201. Keep a USD 1 program guard including prior attempts on resume; this is not an account-level spending cap. Record original key usage, later usage delta, new attempt response costs, and all successful response costs including cached pilot, as separate quantities. Do not equate ambiguous failed-request costs to zero.

## Evidence and validation

Output under private ignored `rag/results/asr-four-model-full-20261117/`: selection/configuration, cache origins, per-request attempts, transcripts/responses, normalized mappings/hashes, query vectors, all selected IDs/scores/metrics, six-cohort summary, billing, environment, Slurm/library logs and completion/failure manifests.

Reproduce the pilot's original and normalized 128 cells with exact IDs/recall/tokens/fallback and maximum selected-score drift <=1e-6. Save all query vectors for an independent CPU audit of full 24,000-record rankings, uncapped cutoff completeness, qrel recall and retokenized document costs. Audit every cohort summary without rerunning ASR or the encoder. No answer generation, judging, WER/ground-truth-transcript evaluation or cross-model inference-latency claim is included.

## References and reuse

- [Existing raw pilot](asr-rag-four-model-pilot-results.md), [fixed normalization control](asr-rag-normalization-protocol.md), [course RAG interfaces](../rag/README.md).
- [OpenRouter STT contract](https://openrouter.ai/docs/guides/overview/multimodal/stt), [official OpenAI Python SDK](https://github.com/openai/openai-python), [Python 3.11 concurrent futures](https://docs.python.org/3.11/library/concurrent.futures.html).
- Runtime dependency list, import paths and target APIs are verified against the actual cluster environment and targeted Slurm checks; no custom ASR, tokenizer, encoder or retrieval algorithm replaces the existing implementations.
