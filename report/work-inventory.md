# Work and evidence inventory — 16 November 2026

The final top-5/top-10/top-20 versus two-cutoff answer evaluation is complete, including blind judging and billing reconciliation. Earlier sufficiency/subset experiments remain stopped. This inventory distinguishes completed results from partial follow-ups.

## Completed work

| Work | Evidence, relative to repository root | Status / use |
| --- | --- | --- |
| Final pipeline depth controls | `rag/results/cutoff-pipeline-20261116/`, `report/draft/pipeline-controls-protocol.md` | CURRENT main answer table: top-5/top-10/top-20 and two frozen cutoffs; 324 queries per row, 1,620 fresh answers, 1,620 final judgements, 3,247 reconciled calls, $1.857334791. One logged explanation-quoting repair preserves identical numeric scores across six attempts. |
| Earlier uncapped cutoff study | `rag/results/cutoff-uncapped-20261116/`, `report/draft/uncapped-protocol.md` | Retrieval/tuning evidence retained; earlier top-2 answer comparison archived separately: 972 answers, 1,945 reconciled calls, $0.845229234. No count cap, no truncated prompts. |
| CUDA RoPE | `report/task1-rope-results.md`, `report/task1-evidence/` | Correctness, performance and rejected fast-math variant |
| CUDA reductions | `report/task2-reductions-results.md`, `report/task2-evidence/` | Official/helper validation, bandwidth and synchronization ablations |
| CUDA GEMM/attention and integrated ASR | `report/task34-kernels-results.md`, `report/task34-evidence/` | Final 33-case correctness, extra cases, limited sanitizer coverage, integrated transcription and ablations; supersedes interim unfinished-task notes |
| Primary five-policy RAG comparison | `report/rag-adaptive-full-results.md`, `rag/results/full-adaptive-20261115T215241Z/` | 467 queries total, 324 locked test queries, 2,335 generated answers and primary judgements; 300 repeat audit judgements; reconciled $2.7480 including audit |
| Retrieval/token frontier, pool 20 | `report/rag-retrieval-frontier-results.md`, primary run `retrieval-frontier/` | Complete saved-data analysis; tokenizer checked against all 2,335 primary prompt counts |
| Prefix oracle, pool 20 | `report/rag-retrieval-oracle-results.md`, primary run `oracle-frontier/` | Complete hindsight bound with independent LP checks |
| Retrieval and oracle, pool 50 | `report/rag-retrieval-top50-results.md`, primary run `retrieval-frontier-top50/` | Complete exact retrieval and offline sweeps, 324 queries; no new generation |
| Precision–recall comparison | `report/plot_saved_retrieval.py`, `report/draft/figures/` | Computed from the existing top-50 metric arrays after experiments stopped; macro/micro distinction documented |

The primary run's `partition.json`, `frozen-parameters.json`, `config.json`, `summary.json`, `validation.json`, request/billing logs and judge-audit files preserve the design and measurements. The main report uses completed evidence and explicitly marks later test-set exploration.

## Stopped follow-ups

| Work | Saved location | Exact limit |
| --- | --- | --- |
| Evidence-sufficiency study | `rag/results/evidence-sufficiency-v3-20261116/` | 125 dev and 101 complete test queries; planned second-model audit not run. Completion-order test subset is not a fresh representative sample. Logged cost $5.7855 includes development; interrupted in-flight usage may be missing. |
| Alternative subset selectors | `rag/results/subset-retrieval-20261116/` | 449 query pools, 22,450 local cross-encoder pair scores and dev-selected parameters saved. Evaluation stopped before a complete validated held-out summary. No paid API calls. |

Both directories contain `stopped-by-user.json`. Do not present scoring completion or dev tuning as completed test results. The report appendix preserves partial observations without a final superiority claim.

## Deliverables and remaining author actions

The revised draft is `report/draft/report.pdf`; the editable package is `report/draft/report-source.zip`. Its main body focuses on the three-method research question, uncapped retrieval/precision/token trade-offs, final answer evaluation, a brief ASR implementation summary and integrated ASR ablation. Detailed kernel material moves to the appendix. Earlier planner/oracle/sufficiency/reranker studies stay archived rather than pooled into the main comparison. Student information appears only on the final page. The full-corpus ranking cache is preserved on fuji3 at `/home/zcm/ragexp/uncapped-rankings-20261116`.

Before submission, the group must review the draft and AI declaration, fill in final-page identity fields, and prepare the separately required video/individual review. Raw evidence directories are preserved locally and on fuji3 but ignored by Git. The source ZIP is a report-authoring package, not an automatically anonymized archive of all coursework evidence.
