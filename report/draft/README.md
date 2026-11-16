# Coursework report draft

This revised draft gathers the ASR work, uncapped retrieval analysis and completed final answer evaluation of fixed top-5, top-10, top-20, two dev-selected cutoffs and four rounded-threshold controls. All nine settings use one fresh randomized batch and final blind LLM judging. Earlier sufficiency/subset experiments remain stopped.

- `report.pdf`: compiled report tracked with Git LFS; six main pages. Run `git lfs pull` after cloning to download the PDF. See `format-check.json` for the local format validation.
- `report.tex`, `references.bib`: editable report source.
- `figures/uncapped-frontiers.png`: macro precision, pooled precision and token/recall curves, using 324 queries and the entire 24,000-document corpus without a count cap.
- `figures/answer-cost-score.png`: Section 4 scatter plot of reconciled generation USD per 1,000 queries versus final judge score, with all nine settings labelled. Generated from saved results without new API calls.
- `figures/final-answer-evaluation.png`: final nine-row answer table, with separate judge dimensions, full input tokens, downstream latency and generation charges. Best observed values are bolded.
- `threshold-controls-protocol.md`: current nine-row design and measurement boundaries. `pipeline-controls-protocol.md` preserves the earlier five-row study. `uncapped-protocol.md` documents the earlier top-2 study.
- `../../rag/results/cutoff-threshold-controls-20261116/summary.json`: final answer results and paired intervals. Retrieval comparisons remain in `../../rag/results/cutoff-uncapped-20261116/frontiers/summary.json`.
- `format-check.json`: supplied checker output, PASS.
- `report-source.zip`: portable LaTeX source, generated table/analysis inputs, required styles, figure PDFs and this README. It excludes credentials, raw query data, model weights and build intermediates.
- `../work-inventory.md`: evidence locations and completed/incomplete status.

The Query-weighted quality panel has **macro precision on x and macro recall on y**. Each query has equal weight. Higher macro precision does not imply higher pooled precision, recall per token or answer quality. Diamonds retain the earlier dev-selected reference settings; curves are exploratory. The other panels keep their original axes.

## Before submission

Fill in Group ID, name and UUN on the final page only. Review every technical claim and the AI-use declaration. The source retains the supplied review format and anonymous main body. The draft is not a replacement for the separate video and individual review deliverables.

The final nine-row study generated 2,916 fresh answers before judging began. All 5,837 calls reconcile to $3.557849731 (generation $1.583303721; judging $1.974546010). Five extra judge attempts resolved a formatting error without manual repair. One answer reached the output cap and is retained. All input prompts fit and all token counts match preflight. The earlier five-row study cost $1.857334791 and the top-2 study $0.845229234; both are archived separately. Partial outputs from interrupted runs are excluded. Sufficiency/subset studies were not resumed.

The added thresholds are absolute tau=0.88/0.87 and relative delta=0.02/0.03, named by parameter rather than target recall. They are exploratory controls motivated by viewed evaluation curves, not dev-tuned selections or exact recall fits. Absolute 0.8925 remains best observed; absolute 0.88 increases recall with an uncertain score difference. See `../rag-threshold-controls-results.md` for the full interpretation.

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
