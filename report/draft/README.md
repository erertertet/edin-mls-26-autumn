# Coursework report draft

This revised draft gathers the ASR work, uncapped retrieval analysis and completed final answer evaluation of fixed top-5, top-10, top-20 and two cutoffs. All five settings use one fresh randomized batch and final blind LLM judging. Earlier sufficiency/subset experiments remain stopped.

- `report.pdf`: revised focused report; see `format-check.json` for the verified page count.
- `report.tex`, `references.bib`: editable report source.
- `figures/uncapped-frontiers.png`: macro precision, pooled precision and token/recall curves, using 324 queries and the entire 24,000-document corpus without a count cap.
- `figures/final-answer-evaluation.png`: final five-row answer table, with separate judge dimensions, full input tokens, downstream latency and generation charges. Best observed values are bolded.
- `pipeline-controls-protocol.md`: current five-row design and measurement boundaries. `uncapped-protocol.md` documents the earlier top-2 study.
- `../../rag/results/cutoff-pipeline-20261116/summary.json`: final answer results and paired intervals. Retrieval comparisons remain in `../../rag/results/cutoff-uncapped-20261116/frontiers/summary.json`.
- `format-check.json`: supplied checker output, PASS.
- `report-source.zip`: portable LaTeX source, generated table/analysis inputs, required styles, figure PDFs and this README. It excludes credentials, raw query data, model weights and build intermediates.
- `../work-inventory.md`: evidence locations and completed/incomplete status.

The Query-weighted quality panel has **macro precision on x and macro recall on y**. Each query has equal weight. Higher macro precision does not imply higher pooled precision, recall per token or answer quality. Diamonds retain the earlier dev-selected reference settings; curves are exploratory. The other panels keep their original axes.

## Before submission

Fill in Group ID, name and UUN on the final page only. Review every technical claim and the AI-use declaration. The source retains the supplied review format and anonymous main body. The draft is not a replacement for the separate video and individual review deliverables.

The final five-row study generated 1,620 answers and judged them only after generation completed, costing $1.857334791 in reconciled charges. One malformed explanation required an explicitly logged JSON-format repair; its numeric scores were unchanged and identical across six attempts. The earlier completed top-2 study cost $0.845229234 and remains archived separately. A broader nine-row sweep was interrupted and its partial outputs are excluded. Sufficiency/subset studies were not resumed.

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
