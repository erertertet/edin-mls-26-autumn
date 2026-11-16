# Final answer evaluation: top-5, top-10, top-20 and two cutoffs

Completed 16 November 2026 on fuji3. All five rows have 324 queries and fresh answers from one shuffled, 12-worker batch. All 1,620 answers completed before the final policy-blind LLM judge started. The shared ranking covers all 24,000 corpus documents; cutoffs have no artificial document cap. Absolute tau=0.8925 and relative gap delta=0.009 retain their earlier dev-only freeze. Fixed k=5,10,20 are predeclared controls.

| Method | Correctness /4 | Grounding /4 | Completeness /4 | Score /1 | Full input tokens | Downstream s | Generation $/1,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed top-5 | 2.19 | 2.15 | 2.18 | .543 | 1,227 | 2.75 | .363 |
| Fixed top-10 | 2.20 | 2.14 | 2.19 | .544 | 2,391 | 3.08 | .581 |
| Fixed top-20 | 2.29 | 2.23 | 2.29 | .568 | 4,730 | 3.50 | .987 |
| Absolute cutoff | **2.66** | **2.64** | **2.65** | **.663** | **657** | **1.93** | **.215** |
| Relative gap | 2.41 | 2.40 | 2.42 | .602 | 667 | 2.09 | .236 |

Bold denotes best observed values, not statistical significance. Generator: Mistral Small 4; judge: Gemini 2.5 Flash. Scores are rubric judgements, not accuracy. Latency covers selection/context plus generator API service and excludes cached ASR, embedding/ranking and worker queue wait.

Absolute's paired score gains versus top-5/top-10/top-20 are +.119 [.069,.168], +.118 [.067,.168], +.095 [.045,.143]. Relative gains are +.058 [.015,.101], +.057 [.009,.104], +.034 [−.011,.078]. These 5,000-resample query-bootstrap intervals are exploratory and unadjusted for multiple comparisons; this evaluation set was inspected previously.

At matched labelled recall, the existing uncapped retrieval analysis still shows higher macro precision but 24.1%/41.4% more retrieved-text tokens for absolute/relative than fixed-k mixtures. The table above compares different-recall operating points, so lower actual cost against top-5/top-10/top-20 does not contradict that efficiency result.

All 1,620 generator outputs stopped normally, and all predicted prompt counts matched actual usage. There were 1,627 judge calls, including retries. One explanation contained unescaped quotes in all six attempts; numeric scores were identical and retained unchanged by a strict, logged formatting repair. Raw responses and `judge-format-repair.json` preserve the evidence. All 3,247 charges reconciled: generation $0.771871491, judging $1.0854633, total **$1.857334791**.

Evidence is in `rag/results/cutoff-pipeline-20261116/`. The old top-2 study and interrupted broader nine-row sweep are separate; no partial outputs or cross-run timings are pooled into this table. Source: `rag/extensions/cutoff_pipeline_controls.py`. Updated report and graphs are under `report/draft/`; the Query-weighted quality plot uses precision on x and recall on y.
