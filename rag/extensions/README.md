Use this directory for investigation-specific code if useful. You are not required
to place all modifications here; substantive changes may naturally involve src/.

## Adaptive retrieval preliminary experiment

`adaptive_retrieval_pilot.py` compares fixed top-5 and top-20 with:

- A query-only LLM planner that predicts an integer document count in `[1, 20]`
  before retrieval. The planner and answer generator use the same model.
- Top-20 retrieval followed by an absolute cosine-similarity cutoff.
- Top-20 retrieval followed by a maximum similarity drop from the best hit.

Corpus and query vectors are normalised, so cosine distance is `1 - similarity`.
Both threshold policies keep at least the best document and at most 20. Their
cutoffs are frozen using eight disjoint practice queries: median fifth-hit
similarity and median first-to-fifth similarity gap. No relevance labels or
answer scores are used for calibration or selection. The evaluation is a seeded
sample of 12 other practice queries; dev is untouched.

On fuji3, after configuring the existing private API file:

```bash
source ~/ragexp/activate.sh
python rag/extensions/adaptive_retrieval_pilot.py \
  --out ~/ragexp/adaptive-preliminary-UNIQUE_RUN_NAME \
  --queries 12 --calibration 8 --budget 2
```

Defaults are `mistralai/mistral-small-2603` (Mistral Small 4) for generation and
planning, and `google/gemini-2.5-flash` for the independent judge. These are
run-local overrides; the private `.env` is not rewritten. Temperature, output
limits and requested reasoning settings are fixed across arms and recorded.

Actual warm end-to-end timings include ASR, encoding, the optional planner API
call, retrieval/selection, prompt construction and answer generation. ASR is
repeated for each arm and its transcript must match across paired runs. Loading
and judging are excluded from answer latency. Policy order is randomised per
query. Separate inference and judge charges, token usage, finish reasons and
OpenRouter generation billing metadata are saved. A spend check runs before
each API request; the request count and output caps also bound this small pilot.
This is not an account-level billing cap.

Outputs include `manifest.json`, calibration records, transcripts, candidates,
judge results, request/billing records, and `summary.json`/`summary.csv`. The
summary includes paired exploratory bootstrap intervals against top-5. These
intervals do not include judge uncertainty, model stochasticity, or correction
for multiple comparisons; one small practice sample is not a final result.

### Matching course models

OpenRouter lists the following corresponding model IDs:

| ELM model name | OpenRouter ID |
| --- | --- |
| Mistral Small 4 | `mistralai/mistral-small-2603` |
| Llama 3.3 (70B Instruct variant) | `meta-llama/llama-3.3-70b-instruct` |
| Qwen 3.5 397B A17B | `qwen/qwen3.5-397b-a17b` |

EuroLLM was not listed by the OpenRouter model catalogue at preliminary setup.
The course default is `utter-project/EuroLLM-22B-Instruct-2512`; substituting a
different model is not an exact reproduction of that default. The earlier
Qwen3-235B pilot is not a match for Qwen3.5-397B and must stay separate.

Match the generator and its settings between the baseline and policy variants.
A separate judge can remain fixed across all comparisons. Matching a model name
does not match endpoint latency, quantisation, serving configuration or price:
OpenRouter measurements must not be reported as ELM measurements. The pilot
does not establish that every provider serves exactly the ELM checkpoint.

## Full parallel comparison

`adaptive_retrieval_batch.py` is the resumable full-run runner. The locked
`partition.json` assigns 125 supplied dev queries to tuning, 324 previously unused
practice queries to a researcher-defined held-out test, and 18 previously seen
practice queries to exploratory diagnostics. Eight original calibration queries
are excluded. There is no official `test` split in the supplied archive.

Before evaluation, prepare all dev ASR/query vectors in `queries.jsonl`. The
runner searches 97 absolute cutoffs and 81 relative gaps using **only dev qrels
macro F1**, breaking ties toward fewer documents. The chosen thresholds are
saved in `frozen-parameters.json` before any held-out testing. This objective is
retrieval quality; generated-answer or test scores are not used for selection.
The raw query-only planner is not tuned. Because public qrels may be incomplete,
unjudged-document precision/F1 should be interpreted with that limitation.

```bash
source ~/ragexp/activate.sh
python rag/extensions/adaptive_retrieval_batch.py \
  --out /path/to/prepared-run-directory --workers 8 --budget 40
```

API work is parallel, with at most eight worker calls in flight; each worker has
its own client. ASR uses one persistent GPU worker and each unique transcript and
CPU query vector is reused across policies. Candidates, plans and judgements are
saved incrementally. Re-running the same command resumes missing tasks without
regenerating saved candidates. Frozen configuration changes require a new run
directory. The budget check reserves funds for in-flight requests and records
returned usage; private credentials are never copied into output files.

Every arm uses a 2048-token answer limit (larger than the preliminary run).
Planner and judge limits remain 64 and 768. The same five policies, model IDs,
temperatures, reasoning request and context prompt are used throughout.

`service_e2e_ms` is a **reconstructed** warm audio-to-answer latency: measured ASR
and encoding plus branch planning/retrieval/context/generation service times.
It excludes API worker queueing and judging. `observed_audio_to_answer_ms` is
actual wall time including queueing, available only for freshly transcribed
queries; prepared/cached dev queries have no such measurement. API concurrency
changes serving conditions, so these times must not be presented as directly
comparable to the sequential pilot. Overall batch wall time measures experiment
throughput. `status.json` reports progress and spend; the final summary separates
dev, held-out test and seen exploratory results, with paired query bootstrap
intervals and billing reconciliation.

## Evidence sufficiency and selection

`evidence_sufficiency_experiment.py` compares fixed k=1,2,3,5,10,20,
MMR diversity selection (3 or 5 records), staged evidence-aware stopping,
one-pass subset selection, and the previously frozen absolute cutoff.
It reuses the full experiment's cached ASR transcripts and dense top-20 rankings.
No ASR or new embedding inference is performed. MMR uses the stored corpus
embeddings; selected records retain original ranking order to control context
ordering. Its lambda is chosen from 0.5,0.7,0.9 using 125 dev queries, separately
for each output size, then frozen before the 324 reused test queries.

The main outcome is **estimated context sufficiency**, not document recall or
generated-answer quality. A Gemini evaluator sees only the question and selected
documents, with no reference checklist, policy name, selector decisions or
generated answer. A positive label requires no declared missing requirement and
at least one supporting quotation found in the context. This checks quotation
provenance, not semantic entailment or evaluator accuracy. Labels are model
estimates, not human ground truth. A locked 30-query audit uses Llama 3.3 on
fixed-3, stopping and one-pass contexts; agreement is not proof of correctness.

A separate Gemini annotation pass constructs necessary-fact blueprints from
public qrels and reference documents. Fact coverage and reference-complete
sufficiency are secondary, restricted to assessable blueprints; all exclusion
counts are reported. Broad questions and overlapping alternative advice make
these checklists imperfect. They do not enter selection or the main evaluator.

The stopping model checks prefixes of 3,5,10 records and otherwise returns 20.
It stops only on a sufficient decision with no missing requirements and a
valid evidence quotation. The one-pass model sees all 20 records and selects a
subset. Both use Mistral Small 4; reading documents and making decisions have
real API cost. Every unique selected context also receives a fresh generated
answer for cost/latency measurement. Identical contexts within a query reuse
the same evaluation and generation calls across policies.

```bash
python rag/extensions/evidence_sufficiency_experiment.py \
  --base ~/ragexp/full-adaptive-20261115T215241Z \
  --out ~/ragexp/evidence-sufficiency-UNIQUE_RUN \
  --stage pilot --workers 16 --budget 15
# Inspect the dev-only pilot before continuing the same configuration:
python rag/extensions/evidence_sufficiency_experiment.py \
  --base ~/ragexp/full-adaptive-20261115T215241Z \
  --out ~/ragexp/evidence-sufficiency-UNIQUE_RUN \
  --stage full --workers 16 --budget 15
python rag/extensions/summarize_evidence_sufficiency.py \
  --run ~/ragexp/evidence-sufficiency-UNIQUE_RUN
```

All stages are resumable. A failed JSON call retries with twice the output cap;
all returned usage is retained. The budget guard reserves funds for concurrent
requests. Interrupted requests may incur provider charges without returned
usage, so logged spend is not an account-level cap or balance.

Graphs show sufficient-context rate against generator input tokens, actual
selector-plus-generator charges, and summed API service time. Evaluation costs
are experiment overhead, excluded from deployment cost. Service timing excludes
local CPU selection/context-building and is measured under concurrency. Adding
cached ASR/embedding/retrieval yields a reconstructed estimate, never a fresh
audio-to-answer wall-clock benchmark. Generator answers are saved but these
context labels do not establish answer correctness or factual grounding.

The previous test results informed this follow-up's design. Despite keeping
new tuning on dev, describe these results as **follow-up evaluation on a reused
test set**, not a pristine confirmatory holdout. Bootstrap intervals resample
queries; they exclude systematic label error and are not multiplicity-adjusted.

## Oracle stopping frontier

`retrieval_oracle_frontier.py` is an offline CPU analysis over the saved
`retrieval-frontier/query-prefix-metrics.npz`. It computes query-specific top-k
and similarity-threshold oracle bounds, plus the best shared-rule envelopes.
The per-query rules have identical expressive power except for tied scores;
the script checks ties explicitly. It optimises macro qrels recall under a
mean generator-input-token budget, retaining the same ranking and top-20 pool.
Continuous curves allow budget-boundary randomisation and are checked against
a general linear-program solver. All labels are used in hindsight, so these
are diagnostic upper bounds rather than deployable policies. No new API calls
or inference runs are needed. See
[`rag-retrieval-oracle-results.md`](../../report/rag-retrieval-oracle-results.md)
for definitions, numerical results, constraints and reproduction commands.

`retrieval_frontier_top50.py` extends the cached query rankings to top-50 using
CPU exact search over the existing enc_a corpus embeddings. Fixed-k points use
5,10,...,50 plus a top-1 fallback anchor; absolute and relative cutoffs enumerate
all distinct selection thresholds within the expanded pool. The old top-20
rankings and token counts are checked for consistency. The script also computes
the expanded per-query oracle bounds and checks them with linear programming.
It makes no model/API calls and does not resume paused inference. See
[`rag-retrieval-top50-results.md`](../../report/rag-retrieval-top50-results.md)
for results and reproduction.

## Uncapped cutoffs and final answer-depth controls

`prepare_uncapped_rankings.py` ranks all 24,000 corpus records against the saved
125 dev and 324 evaluation query vectors, on CPU. `cutoff_followup.py` preserves
the initial dev-selected top-2 versus two-cutoff experiment. Its cutoff parameters
are reused unchanged by `cutoff_pipeline_controls.py`, which compares fixed
**top-5, top-10 and top-20**, absolute cutoff and relative gap. Both cutoffs search
the full corpus without a document-count cap; an empty selection falls back to
top-1. Fixed depths are predeclared controls rather than retuned choices.

The final five-row study generates all 1,620 fresh answers before starting blind
LLM judging, with the same generator, judge, decoding settings and 12 workers.
All rows share a randomized batch, cached ASR/transcripts and full-corpus ranking.
Generation and judging are independently resumable, usage is metered, and billing
is reconciled. Downstream latency excludes cached ASR, embedding/ranking and queue
wait; it is not a fresh audio-to-answer measurement. These are reused evaluation
queries, so results remain follow-up evidence rather than a new holdout.

`uncapped_frontiers.py` produces saved-data retrieval curves with explicit macro
and micro metrics. Its additive token metric counts retrieved document texts;
the answer run records actual full API input tokens separately. The final report
renderer accepts separate frontier and answer-evaluation directories:

```bash
python report/plot_uncapped_report.py \
  --run ~/ragexp/cutoff-uncapped-20261116 \
  --evaluation-run ~/ragexp/cutoff-pipeline-20261116 \
  --out ~/ragexp/cutoff-pipeline-20261116/report/figures
```

The query-weighted panel has precision on x and recall on y. The final answer
table bolds best observed values, not significance. The broader nine-row
`cutoff_expanded_controls.py` run was interrupted when scope narrowed; its partial
results do not enter the five-row table. See `report/draft/pipeline-controls-protocol.md`.

## Four-ASR recall–token follow-up

The saved 475-audio experiment is extended without ASR, encoder or API calls by
`prepare_asr_curve_cache.py`, `analyze_asr_curves.py` and
`render_asr_curve_analysis.py`. The frozen JSON plan declares 307 policies per
case, 30/40/50% recall targets and 5,000 paired query bootstrap replicates.
Calibration uses 350 practice queries; evaluation uses the 117 dev queries
outside the original pilot. Practice-selected transfer and descriptive
evaluation-curve recall standardization are separate estimands. These reused
data are exploratory, not a fresh holdout.

Submit `run_prepare_asr_curves.sbatch`, then `run_analyze_asr_curves.sbatch`,
then `run_audit_asr_curves.sbatch` from the repository root in Edinburgh Slurm.
The independent auditor checks all operating points, paired intervals and
aggregate curves, plus selected saved bootstrap replicates, without importing
the producer's statistical kernel or drawing new query samples. The original
top-5/cutoff 0.8925 controls remain unchanged. Private inputs/results are required;
the wrappers do not provision them or resume ASR work.

See the [frozen protocol](../../report/asr-matched-curves-protocol.md) and
[results with PNG/PDF figures](../../report/asr-matched-curves-results.md).
The [aggregate curves CSV](../../report/asr-matched-curves.csv) and
[independent validation record](../../report/asr-matched-curves-audit.json)
are public; the per-query count cache and bootstrap indices remain private.

## Transcript differences by edit distance

`analyze_asr_edit_distance.py` quantifies all six unordered ASR-model pairs on
the same 475 saved raw/normalized transcripts using RapidFuzz 3.14.6 unit-cost
Levenshtein distance. It reports character and whitespace-word absolute edits
and the macro mean of `distance / max(lengths)`, with paired query-bootstrap
intervals. These reference-free distances measure disagreement, not WER/CER.
The query-wise overall value averages the six pairs before query resampling.

Run `run_asr_edit_distance.sbatch`, followed by
`run_audit_asr_edit_distance.sbatch`, in Edinburgh Slurm. Render the audited
aggregates with `render_asr_edit_distance.py --run RUN --output RUN/report.md`
inside an allocation. This follow-up reads existing transcripts, makes no
ASR/encoder/API calls and leaves the retrieval results and normalizer unchanged.
The [protocol](../../report/asr-edit-distance-protocol.md),
[result tables](../../report/asr-edit-distance-results.md),
[aggregate CSV](../../report/asr-edit-distance.csv) and
[audit](../../report/asr-edit-distance-audit.json) document the calculation.
