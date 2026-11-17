# Frozen ASR recall–token curves, separated calibration and paired intervals

This follow-up uses only the completed 475-audio experiment's saved query vectors, corpus/qrels and pinned tokenizer. It performs **zero ASR, encoder, generation, judging or API calls**. The original raw/normalized transcripts and normalization file remain unchanged. The prior 0.8925/top-5 measurements are retained as fixed controls, not replaced by better-looking tuned results.

## Partition and estimands

The new calibration step uses **350 practice queries** only. Parameters and query-independent mixture weights are frozen before evaluation on **117 dev queries excluding the eight pilot queries**. The eight pilot queries remain historical/descriptive evidence. All 475 queries have already been inspected in previous work, and public dev labels informed the original cutoff; this is operational separation for a new exploratory analysis, **not a newly untouched test set**. Do not retune the normalization rule, partition, grids or targets after seeing the new curves.

Report two distinct estimands:

1. **Calibration-target transfer:** choose each ASR × text-condition × retrieval-family policy using practice only, at common calibration macro-recall targets **0.30, 0.40, 0.50**. Freeze its parameters and probability, apply to the 117 evaluation queries, and report the **actual evaluation recall**, precision and tokens. Evaluation recall may differ from the target and between policies. This table must not be described as exactly matched evaluation recall.
2. **Evaluation-curve standardization:** separately evaluate the predeclared parameter grid and interpolate its evaluation curve at the same common recall targets. This is a descriptive equal-recall metric, analogous to reading a curve, **not selection of a deployable threshold from evaluation labels**. Keep it separate from the frozen transfer table and preserve actual fixed-threshold results.

Descriptive curves also cover all 475, practice, dev, remaining dev, pilot and remaining-all cohorts. Primary paired intervals concern the 350-calibration/117-evaluation design; overlapping cohorts are not independent replications.

## Predeclared grids and metrics

The machine-readable `asr-curve-plan.json` fixes all settings before new curve computation. Four ASR arms × raw/normalized give eight cases. Top-k uses k=1..200 plus 250, 500, 1000, 2000, 4000, 8000, 16000, 24000. Absolute tau uses 0.7500..0.9900 in 0.0025 increments plus -1.000001 and 1.000001 full-range endpoints. Tau=0.8925 and k=5 are explicit controls. Every cutoff uses >=, top-1 fallback and no document-count cap. These produce **307 policies per case** against all 24,000 corpus records.

Recompute each ranking from the saved float32 normalized query vector, **without normalizing it again**, using the supplied exact retriever's corpus vectors and stable corpus-row tie order. Prefix cumulative qrel hits and document tokens give every grid point. Verify that the original 7,600 fixed-control cells reproduce exactly, with maximum selected-score drift <=1e-6. Corpus tokens are computed once for all 24,000 documents with the original pinned tokenizer and `add_special_tokens=False`.

- Macro recall = mean over queries of relevant hits / original positive qrels.
- Macro precision = mean over queries of relevant hits / returned documents.
- Micro precision = pooled relevant hits / pooled returned documents.
- Token cost = mean summed retrieved-document-text tokens; excludes question/system text and chat framing and is not a full billed prompt count.
- Also report mean documents and achieved recall for all transferred policies.

## Equal-recall interpolation

Within each retrieval family, sort policies by macro recall. For recall plateaus within absolute 1e-12, select lowest mean tokens, then mean documents, numeric parameter and original grid index. Interpolate the adjacent distinct-recall representatives bracketing the target. Never extrapolate or silently clip an unsupported target.

The interpolation is an analytic **query-independent randomized policy mixture**, with one common probability for every query; it is not a qrels-based per-query oracle. Expected recall, macro precision, tokens, documents and hits mix linearly. Micro precision is computed as **mixed expected hits / mixed expected documents**, never by interpolating two micro precision ratios. This is adjacent-grid interpolation, not a globally optimized convex-envelope claim. The plateau convention improves on the older direct `np.interp` handling; the new matched-recall table is not claimed to reproduce that older convention bit-for-bit.

## Query-level intervals and pairing

Use **5,000 bootstrap replicates**, seed **20261117**, percentile 95% intervals. Reuse SciPy's verified `stats.bootstrap`, with two independently resampled index arrays for calibration and evaluation (`paired=False` between these disjoint groups). Within each replicate, its calibration indices and evaluation indices are respectively shared by **every** model, text condition, retrieval family and contrast. No retrieval cell or document is treated as an independent observation.

For every replicate, rebuild the calibration mean curves, reselect parameters/weights on calibration only, and evaluate on resampled evaluation queries. This includes calibration uncertainty. Fixed 0.8925/k=5 controls are never tuned. For evaluation-curve standardization, recompute the evaluation curves and interpolation on each replicate; its role remains descriptive.

Report pointwise intervals for normalization-minus-raw, alternative-ASR-minus-GLM, and cutoff-minus-top-k differences in achieved recall, macro/micro precision, tokens and documents. Recall/precision differences are percentage points. Preserve all paired bootstrap outputs and actual resampled indices for audit.

At each panel/target, all 16 policies share one attainable-replicate set. Report the supported count out of 5,000. If fewer than 95% are supported, mark its interval unavailable; otherwise bounds are conditional percentiles on the explicitly reported common supported set. No silent removal of unsupported replicates, no simultaneous/multiplicity-adjusted significance claim, and no correction for qrel incompleteness, prior analysis choices or ASR generation variability.

## Libraries, execution and evidence

Current cluster environment was inspected using requirements/constraints, `pip list/show`, import paths and actual API signatures: Python 3.11.18, NumPy 2.4.6, SciPy 1.17.1 and Tokenizers 0.22.2. Existing `uncapped_frontiers.py` and `prepare_uncapped_rankings.py` supply the full-ranking/prefix approach, but their hard-coded split and direct interpolation are not reused as an independent calibration/evaluation design.

Matplotlib 3.11.2 is the only added direct dependency, in the separate RAG venv; official PyPI metadata, signed release and target docs were checked for Python 3.11/Linux compatibility and PSF license. Existing ASR environment/packages are preserved. Its dependency dry-run and pip-before/after records are private evidence. A minimal SciPy vectorized two-group bootstrap and Matplotlib errorbar/savefig probe runs inside Slurm before the analysis. Domain-specific mixture/support bookkeeping is custom because generic bootstrap/curve APIs do not encode this experiment's estimands; standard library implementations still perform sorting, prefix sums, resampling, quantiles and plotting.

Run all programs, tests and plotting in CPU-only Edinburgh Slurm allocations on saxa, with OMP_NUM_THREADS=4 and OPENBLAS_NUM_THREADS=2. Stage only corpus/tokenizer files to node-local scratch; do not load ASR/encoder models or credentials. Freeze this protocol, plan, source hashes, partition and input artifact hashes. Save per-query grid counts, calibration policies, descriptive curves, point/contrast intervals, bootstrap draws, PNG/PDF figures, runtime/API-probe logs, completion and an independent validation record under ignored `rag/results/asr-matched-curves-20261117/`. Public PR content contains aggregate tables/figures only.

## References

- [Prior full-audio results](asr-rag-full-results.md) and [the frozen normalization rule](../rag/extensions/transcript_normalization.py).
- [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) and [installed-version source](https://github.com/scipy/scipy/blob/v1.17.1/scipy/stats/_resampling.py).
- [Threshold selection/evaluation separation](https://scikit-learn.org/stable/modules/classification_threshold.html).
- [NumPy 2.4 APIs](https://numpy.org/doc/2.4/), [Matplotlib 3.11.2 release](https://github.com/matplotlib/matplotlib/releases/tag/v3.11.2), [Matplotlib package compatibility/license](https://pypi.org/project/matplotlib/).
