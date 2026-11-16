# Retrieval-only evidence/token frontier

The sweep reuses fixed top-20 rankings for the same 324 held-out queries, from
the common 24,000-document corpus. It evaluates fixed k=1..20, 99 absolute
similarity settings and 85 relative gaps, including common endpoint settings.
No ASR, encoder, planner, answer-generation or judge inference was run again.
Additional inference API cost: **$0**.

## Measurement

X is mean full generator input tokens (system, question, evidence and chat
overhead). Y is macro recall of qrels-labelled relevant records, including all
324 queries. It measures evidence selection, not generated-answer quality.
All policies select prefixes from exactly the same ranking and return at least
one and at most 20 records.

The official Mistral Small 4 tokenizer was pinned to revision
`a11f36bebf709121056b1dbcc943d1c6afbe494d`. A constant 17-token chat overhead
was calibrated from dev usage logs. Offline counts matched **all 2,335** prior
generator API input counts exactly, including the held-out cases. There were
no prefix-ranking mismatches. No model weights were downloaded for this sweep.

The dashed baseline is the upper concave envelope of fixed-k points. It permits
query-independent random mixtures of fixed k values, so fractional average
budgets do not unfairly favour adaptive policies. Lines connect discrete
settings; the full test curves are descriptive and no new threshold was selected
from them. Diamonds highlight the earlier dev-selected settings.

## Fixed-k results

| k | Mean input tokens | Macro evidence recall |
| ---: | ---: | ---: |
| 1 | 302.5 | 0.2303 |
| 2 | 531.9 | 0.3209 |
| 3 | 772.9 | 0.3772 |
| 5 | 1,227.0 | 0.4240 |
| 10 | 2,391.2 | 0.4846 |
| 20 | 4,730.3 | 0.5775 |

## Matched-token comparisons at the existing operating points

| Policy | Mean input tokens | Policy recall | Fixed-k mixture recall at same cost | Recall difference (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| Dev-selected absolute cutoff | 644.4 | 0.3239 | 0.3472 | -2.33 |
| Dev-selected relative gap | 666.5 | 0.3060 | 0.3524 | -4.64 |
| Cached LLM-predicted k | 724.7 | 0.3504 | 0.3660 | -1.55 |

All three matched-cost baselines mix k=2 and k=3. The paired 5,000-resample
bootstrap intervals for recall differences are respectively [-4.71, -0.16],
[-6.97, -2.38], and [-3.28, +0.10] percentage points. Resampling recomputes the
fixed-k mixture envelope. These are exploratory, unadjusted intervals.

The absolute cutoff does **not** improve the recall/token frontier at its
dev-selected point. The relative gap also underperforms there. The cached
model prediction does not show a clear matched-cost coverage advantage; its
planning tokens and sequential latency are excluded here because this is a
retrieval-selection experiment. Include those in a deployment-cost comparison.

The relative-gap sweep has a small positive bump at high budgets, but it is not
a newly validated operating point or evidence of a general advantage. The
dominant pattern is that fixed k recovers at least as much relevant evidence
for most observed budgets.

This qualifies the earlier end-to-end finding: higher answer-judge scores with
smaller filtered contexts do not establish more efficient recovery of relevant
documents. The gains may reflect less distracting context, shorter answers or
abstention behaviour. A matched-small-k answer-generation control would be
needed before attributing downstream gains specifically to adaptive stopping.

## Graphs, data and reproduction

Graphs and data are under
`rag/results/full-adaptive-20261115T215241Z/retrieval-frontier/` locally and in
the corresponding run directory on fuji3:

- `retrieval-frontier-main.png` / `.svg`: readable single-panel graph.
- `retrieval-frontier.png` / `.svg`: full graph plus matched-cost recall gains.
- `sweep.csv`: every setting, cost, recall, precision and baseline difference.
- `summary.json`: validation, frozen points and bootstrap intervals.
- `query-prefix-metrics.npz`: paired per-query prefix costs and metrics.

```bash
source ~/ragexp/activate.sh
python rag/extensions/retrieval_frontier.py \
  --run ~/ragexp/full-adaptive-20261115T215241Z \
  --out ~/ragexp/full-adaptive-20261115T215241Z/NEW_SWEEP_DIRECTORY
```

The run needs the pinned tokenizer and Matplotlib, not model weights or API
credentials. The [runner](../rag/extensions/retrieval_frontier.py) is in the
repository. Earlier dev/test partition definitions remain unchanged.
