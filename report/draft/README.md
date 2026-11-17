# Coursework report draft

This revised draft gathers the ASR work, uncapped retrieval analysis and completed final answer evaluation of fixed top-5, top-10, top-20, two dev-selected cutoffs and four rounded-threshold controls. All nine settings use one fresh randomized batch and final blind LLM judging. Earlier sufficiency/subset experiments remain stopped.

The third annotated review uses **macro precision and macro recall throughout
the article**: both average per-query ratios with equal query weight. Formula,
figure, tables, bolded maxima and conclusions use this definition consistently.
The retrieval plot marks upper right as better for precision/recall and upper
left as better for token/recall. Threshold comparisons show both starting and
ending precision; the ASR comparison explicitly assesses GLM transcription
capability. Confidence intervals remain omitted, and Table 2 retains recall
and precision instead of latency and dollar cost. The original dev macro-F1
tuning objective and top-score-relative cutoff remain unchanged.

The fourth review follows the official template's five sections: Introduction;
System and Technical Approach; Experimental Methodology; Results and Analysis;
Discussion and Conclusion. Kernel implementation, hardware and validation
procedure are in Section 3.1. Kernel and E2E numbers are in Section 4.4 / Table 3;
GLM transcript comparisons are in Section 4.5 / Table 4. Macro F1 is explicitly
the average of per-query harmonic means (`mean(2*hits/(selected+relevant))`),
matching the dev-tuning code. The score-distribution discussion leads Section
4.2, and Section 4.3 replaces the old interpretation paragraph with the saved
query-fixed-effects regression. The report's administrative page retains only
the AI declaration and group fields, as in the template.

The pre-restructure snapshot was pushed as commit `88b1c76` on
`codex/asr-whisper-pilot`, including the PDF via Git LFS.

The layout is unchanged: 5.5-inch text width, 10-point body font and approximately
10.96 PDF points between body baselines. Style/support files match both Git and
the supplied template. The earlier annotated `1.pdf` was flattened to A4, while
`2.pdf` and the compiled report are US Letter.

The former Appendix B (full-range retrieval/evidence inventory) and Appendix C
(answer-score/context-size plots) have been removed. Appendix A retains the
CUDA implementation and detailed ablations; the final administrative page is
unnumbered. Supporting experiment files and preview figures remain available.

The ASR comparison now treats normalization as a **diagnostic for GLM transcript
agreement**, not a proposed pipeline improvement. Section 4.5 reports the saved
whole-transcript disagreement counts and the completed word/character
edit-distance results from the audited 475-query saved-data analysis.
The existing normalization rule also needs a leading-decimal audit before it
can be described as meaning-preserving. The final nine-policy answer experiment
continues to use its original transcripts.

- `report.pdf`: compiled report tracked with Git LFS; six main pages. Run `git lfs pull` after cloning to download the PDF. See `format-check.json` for the local format validation.
- `report.tex`, `references.bib`: editable report source.
- `asr-agreement.tex`: GLM comparison, completed edit-distance table, metric definitions and interpretation.
- `figures/uncapped-frontiers.png`: macro precision, pooled precision and token/recall curves, using 324 queries and the entire 24,000-document corpus without a count cap.
- `figures/answer-score-distributions-nine.png` / `.pdf`: the selected preview 01, adapted for report-sized labels; nine exact-score histograms now replace the cost-versus-score graph in Section 4. The older cost graph is retained as an unused asset and excluded from the current source archive.
- `figures/answer-score-distribution.png` / `.pdf`: all 2,916 final scores, with horizontal jitter only and mean diamonds, separately labelled for all nine settings.
- `figures/answer-score-documents.png` / `.pdf`: exact score versus retrieved-document count, split by retrieval family. Bubble area encodes coincident answers within a setting; all panels share a log count axis and include the maximum of 280 documents. Markers/shades distinguish thresholds. These two earlier plots are retained as unused assets; they are no longer included in the PDF or source archive.
- `figures/answer-distribution-data.json`: aggregate count/score frequencies, summary checks, rendering versions and input hashes; no query IDs or text.
- `answer-table.tex`: final nine-row answer table, with judge dimensions, full input tokens, macro recall and macro precision. Best observed values are bolded. The table renderer uses the same columns; older standalone images are retained as previews.
- `threshold-controls-protocol.md`: current nine-row design and measurement boundaries. `pipeline-controls-protocol.md` preserves the earlier five-row study. `uncapped-protocol.md` documents the earlier top-2 study.
- `../../rag/results/cutoff-threshold-controls-20261116/summary.json`: final answer results and paired intervals. Retrieval comparisons remain in `../../rag/results/cutoff-uncapped-20261116/frontiers/summary.json`.
- `format-check.json`: supplied checker output, PASS.
- `report-source.zip`: portable LaTeX source, generated table/analysis inputs, required styles, figure PDFs and this README. It excludes credentials, raw query data, model weights and build intermediates.
- `../work-inventory.md`: evidence locations and completed/incomplete status.

The current report figure has **macro precision on x and macro recall on y**
in the first panel, and retrieved-text tokens versus macro recall in the second.
Diamonds retain the earlier dev-selected references. The standalone macro/micro
comparison and regression previews remain outside the report.

Regenerate only the current report figure from saved data:

```text
python report/plot_uncapped_report.py --run rag/results/cutoff-uncapped-20261116 --evaluation-run rag/results/cutoff-threshold-controls-20261116 --retrieval-only --report-only --out report/draft/figures
```

## Before submission

Fill in Group ID, name and UUN on the final page only. Review every technical claim and the AI-use declaration. The source retains the supplied review format and anonymous main body. The draft is not a replacement for the separate video and individual review deliverables.

The final nine-row study generated 2,916 fresh answers before judging began. All 5,837 calls reconcile to $3.557849731 (generation $1.583303721; judging $1.974546010). Five extra judge attempts resolved a formatting error without manual repair. One answer reached the output cap and is retained. All input prompts fit and all token counts match preflight. The earlier five-row study cost $1.857334791 and the top-2 study $0.845229234; both are archived separately. Partial outputs from interrupted runs are excluded. Sufficiency/subset studies were not resumed.

The added thresholds are absolute tau=0.88/0.87 and relative delta=0.02/0.03, named by parameter rather than target recall. They are exploratory controls motivated by viewed evaluation curves, not dev-tuned selections or exact recall fits. Absolute 0.8925 remains best observed; absolute 0.88 increases recall with an uncertain score difference. See `../rag-threshold-controls-results.md` for the full interpretation.

## ASR comparison provenance and interpretation

Whole-transcript and normalized top-5 set disagreement comes from
[PR #5's saved-data comparison](https://github.com/erertertet/edin-mls-26-autumn/pull/5#issuecomment-6948640302),
based on the audited 475-query run documented in `../asr-rag-full-results.md`.
GLM-versus-Whisper/MAI/Voxtral raw disagreements are 261/155/152 queries;
normalized disagreements are 71/66/62. All denominators are 475. Normalized top-5
set disagreements are 56/52/48, respectively. The separate matched-recall
robustness observation uses **117 dev queries**, as defined in
`../asr-matched-curves-results.md`; it is not the 324-query answer evaluation.

Edit-distance values come from `../asr-edit-distance.csv` at commit `0ba8312`,
with definitions in `../asr-edit-distance-protocol.md`, full uncertainty tables
in `../asr-edit-distance-results.md` and audit job 3726994 in
`../asr-edit-distance-audit.json`. The private transcript archive is not present
in this checkout; the report incorporates the audited public aggregates.
Distances count minimum unit-cost Levenshtein operations, with whitespace word
splitting and Unicode characters including spaces. NED divides each query's
distance by the longer sequence length and then averages queries. This replaces
the earlier draft's proposed reference-denominated corpus rate; neither output
is ground truth. Text formatting normalization and length normalization in NED
are distinct operations. These are agreement measures, not reference-based WER/CER.

Normalized word edits total 164/156/146 for GLM versus Whisper/MAI/Voxtral.
Dividing by 475 gives the table's mean edits per query; dividing by 71/66/62
disagreeing queries gives 2.31/2.36/2.35 edits among disagreements. Word NED
point estimates are 2.456/2.317/2.202%, with the reported 5,000-replicate,
pointwise paired-query bootstrap intervals in the saved analysis. Intervals are omitted from the current report at the author's request. Small edits can still alter meaning.
The leading-decimal normalization edge case remains unresolved; all reported
numbers describe the original frozen normalization, not a corrected rerun.

The plots use the local saved `results.jsonl`, validate all three rubric
components and document counts, require exactly one judgement for each of the
324 shared queries under nine policies, and reconcile means with `summary.json`.
They use no model/API calls. Exactly 263--281 of 324 answers per setting
(81.2--86.7%) score at either endpoint. This is descriptive, not a significance
test or causal estimate of the effect of context length.

Regenerate from the repository root in the report-rendering environment:

```text
python report/plot_answer_distributions.py --run rag/results/cutoff-threshold-controls-20261116 --out report/draft/figures
```

Use `--nine-only` to render only the newly selected Section 4 plot. The standalone
preview 05 (document-count change versus score change, paired with fixed top-5)
remains outside the report pending figure selection.

## Precision/recall regression

Section 4.3 uses the completed saved-data fit in `../answer-regression-results.json`.
It regresses all 2,916 per-query scores on per-query precision and recall, with a
separate intercept for each of the 324 queries. The slopes are 0.1844186 and
0.0508738, or +0.01844/+0.00509 score per +10 percentage points. Adding
configuration intercepts yields +0.01694/+0.00702. Within-query R? is 0.0257501;
its denominator excludes between-query variation. Whole-query bootstrap bounds
support a positive precision association while recall remains uncertain; the
report omits interval numbers as requested. No causal effect is claimed.
`../analyze_answer_regression.py` reproduces the saved-data analysis and preview.

## Kernel numerical correctness

Section 4.4 now lists the largest recorded absolute errors for RoPE, RMSNorm,
LayerNorm, softmax, GEMM and attention, alongside the supplied atol/rtol values.
Sources are `../task1-rope-results.md`, `../task2-reductions-results.md` and
`../task34-kernels-results.md`. These compare against supplied CPU implementations
and cuBLAS for large GEMM, **not direct PyTorch tensor differences**. Large
attention checks sample output rows. The integrated PyTorch/custom comparison
now includes the audited E2E results in `../asr-e2e-error-results.md` and
`../asr-e2e-error-audit.json` (remote commit `41f272f`, GPU job 3727116,
CPU audit 3727120). Main Section 4.4 Table 3 now includes a separate GLM-ASR end-to-end block, alongside the operator checks, reporting all-CUDA maximum discrepancies of
5.245209e-5 in logits and 1.788139e-6 in probabilities, with 0/13 token
mismatches and exact decoded-text agreement on one 3.505-second clip. Appendix A
contains all three custom modes, relative L2 probability discrepancy, audio
projection errors, repeat controls and the independent audit. Each comparison
covers 770,432 logits and probabilities. This separate instrumented IEEE FP32
run adds numerical results; historical timing/memory results remain unchanged.
The detailed ASR timing table stays in Appendix A, with 4.26x runtime and
0.34 GiB lower peak memory summarized in the main text.

## Build

From this directory, with a LaTeX distribution installed:

```text
pdflatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

From the repository root, the supplied checker can inspect the resulting PDF:

```text
python report/format_checker/check_report_format.py report/draft/report.pdf --json report/draft/format-check.json
```

The five supplied style/support files are unchanged from `report/latex_template`. The final checker output records the main-body count, 10 pt body text, required header and review line numbers. Earlier top-50 figures and the prior report are retained for review, but the current source archive includes only the revised report's required files.
