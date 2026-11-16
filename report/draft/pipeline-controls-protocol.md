# Whole-pipeline answer evaluation with fixed top-5, top-10 and top-20

This supersedes the proposed wider nine-row sweep. The final table has exactly five rows:

1. Fixed top-5
2. Fixed top-10
3. Fixed top-20
4. Absolute similarity cutoff, tau=0.8925
5. Relative top-score gap, delta=0.009

The two cutoff parameters retain the previous dev-only freeze. No new tuning or selection from evaluation-answer scores is allowed. Fixed depths are predeclared controls requested by the user. All five settings receive fresh answers on the same 324 queries, in one shuffled 12-worker batch. All 1,620 answers must finish before the final policy-blind LLM judge starts. Use the same Mistral Small 4 generator, Gemini 2.5 Flash judge, prompts, temperatures and token caps as the earlier completed study. Every context fits without truncation; the largest full prompt is 8,977 tokens.

This evaluates final generated answers, not retrieval metrics alone. Cached ASR transcripts and query embeddings are shared across every method. Reported latency is downstream selection/context construction plus generation API service, excluding the cached upstream stages and queue wait; it must not be labelled newly measured audio-to-answer latency.

The table highlights the best observed number in each column: highest answer scores, lowest full input tokens, latency and generation charges. Bold is descriptive, not a significance declaration. Show correctness, grounding and completeness separately, with score=sum/12. Judge costs are experiment overhead and reported separately from generation cost.

The retrieval figure's **Query-weighted quality** panel uses **x=macro precision, y=macro recall**. Other panels keep their existing axes. Top-k labels are retained. Retrieval frontiers reuse saved complete rankings/metrics, without new retrieval or tuning.

Run: `/home/zcm/ragexp/cutoff-pipeline-20261116`; source: `rag/extensions/cutoff_pipeline_controls.py`. Budget guard: $8. The broader `/home/zcm/ragexp/cutoff-expanded-20261116` run was stopped when the user narrowed the request; its partial answers do not enter this table. The previously completed top-2 comparison remains archived as a separate diagnostic.
