"""CPU-only inspection of saved judgements; no model calls or report edits."""
from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'report/draft/build/answer-regression'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = REPO / 'rag/results/cutoff-threshold-controls-20261116/results.jsonl'
SEED, REPS = 20261117, 5000
ARMS = ['fixed5', 'fixed10', 'fixed20', 'absolute_radius', 'absolute_088',
        'absolute_087', 'relative_gap', 'relative_002', 'relative_003']
MODELS = ['pooled', 'query', 'query_policy']
LABELS = ['All observations', 'Same-query comparison', 'Same query + configuration control']


def residualize(x, y, model):
    x, y = x.copy(), y.copy()
    if model != 'pooled':
        x -= x.mean(axis=1, keepdims=True)
        y -= y.mean(axis=1, keepdims=True)
    if model == 'query_policy':
        x -= x.mean(axis=0, keepdims=True)
        y -= y.mean(axis=0, keepdims=True)
    x, y = x.reshape(-1, 2), y.ravel()
    if model == 'pooled':
        x = np.column_stack([np.ones(len(y)), x])
    return x, y


def fit(x, y, model):
    a, b = residualize(x, y, model)
    beta, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    assert rank == a.shape[1]
    error = b - a @ beta
    assert np.max(np.abs(a.T @ error)) < 1e-8
    return {
        'beta_precision': float(beta[-2]),
        'beta_recall': float(beta[-1]),
        'intercept': float(beta[0]) if model == 'pooled' else None,
        'r2': float(1 - error @ error / np.sum((b - b.mean()) ** 2)),
        'regressor_correlation': float(np.corrcoef(a[:, -2:].T)[0, 1]),
        'condition_number': float(np.linalg.cond(a)),
    }


def main():
    # Record all report files to verify that this preview leaves them unchanged.
    report_files = [p for p in (REPO / 'report/draft').rglob('*')
                    if p.is_file() and 'build' not in p.relative_to(REPO / 'report/draft').parts]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in report_files}
    rows = [json.loads(line) for line in SOURCE.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 2916
    by_key = {(r['query_id'], r['arm']): r for r in rows}
    assert len(by_key) == len(rows)
    queries = sorted({r['query_id'] for r in rows})
    assert len(queries) == 324
    assert set(r['arm'] for r in rows) == set(ARMS)
    for r in rows:
        assert r['k'] == len(r['record_ids']) == len(set(r['record_ids']))
        assert np.isclose(r['precision'], r['hits'] / r['k'])
        assert np.isclose(r['recall'], r['hits'] / r['relevant_documents'])
        assert np.isclose(r['score'], sum(r['scores'][k] for k in
                          ['correctness', 'grounding', 'completeness']) / 12)
    summary = json.loads(SOURCE.with_name('summary.json').read_text(encoding='utf-8'))
    for arm in summary['by_arm']:
        selected = [r for r in rows if r['arm'] == arm['arm']]
        assert np.isclose(np.mean([r['score'] for r in selected]), arm['mean_score'])
        assert np.isclose(sum(r['hits'] for r in selected) / sum(r['k'] for r in selected), arm['micro_precision'])

    x = np.array([[[by_key[q, a]['precision'], by_key[q, a]['recall']]
                   for a in ARMS] for q in queries])
    y = np.array([[by_key[q, a]['score'] for a in ARMS] for q in queries])
    fitted = {m: fit(x, y, m) for m in MODELS}
    # Resample whole queries with all nine settings intact. Re-estimate the
    # configuration means for each two-way fixed-effects bootstrap sample.
    rng = np.random.default_rng(SEED)
    bootstrap = {m: np.empty((REPS, 2)) for m in MODELS}
    for i in range(REPS):
        ix = rng.integers(len(queries), size=len(queries))
        for m in MODELS:
            a, b = residualize(x[ix], y[ix], m)
            bootstrap[m][i] = np.linalg.lstsq(a, b, rcond=None)[0][-2:]
    for m in MODELS:
        for j, name in enumerate(['precision', 'recall']):
            fitted[m][name + '_ci95'] = np.percentile(bootstrap[m][:, j], [2.5, 97.5]).tolist()
            fitted[m][name + '_per_10pp'] = fitted[m]['beta_' + name] * .1

    # Partial R2: additional explained variation from each regressor, given the
    # other regressor and the same fixed effects.
    for m in MODELS:
        a, b = residualize(x, y, m)
        full = b - a @ np.linalg.lstsq(a, b, rcond=None)[0]
        for j, name in enumerate(['precision', 'recall']):
            reduced = np.delete(a, a.shape[1] - 2 + j, axis=1)
            residual = b - reduced @ np.linalg.lstsq(reduced, b, rcond=None)[0]
            fitted[m][name + '_partial_r2'] = float(1 - (full @ full) / (residual @ residual))

    result = {
        'source': SOURCE.relative_to(REPO).as_posix(), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'observations': len(rows), 'queries': len(queries), 'configurations': ARMS,
        'equation': 'score_qa = intercept / fixed effects + beta_P * precision_qa + beta_R * recall_qa + residual',
        'precision': 'Per-query relevant retrieved documents / selected documents; not an arm-level macro-precision predictor.',
        'recall': 'Per-query relevant retrieved documents / labelled relevant documents.',
        'score': 'Sum of correctness, grounding and completeness divided by 12; range 0 to 1.',
        'uncertainty': f'{REPS} whole-query bootstrap resamples; percentile 95% intervals; seed {SEED}.',
        'r2_definition': 'Pooled R2 uses total score variation; query R2 uses within-query variation; query_policy R2 uses variation after removing query and configuration means.',
        'models': fitted,
        'diagnostics': {
            'queries_with_precision_variation': int(np.sum(np.ptp(x[:, :, 0], axis=1) > 0)),
            'queries_with_recall_variation': int(np.sum(np.ptp(x[:, :, 1], axis=1) > 0)),
            'zero_hit_rows': sum(r['hits'] == 0 for r in rows),
            'full_score_zero_hit_rows': sum(r['hits'] == 0 and r['score'] == 1 for r in rows),
            'endpoint_scores': sum(r['score'] in (0, 1) for r in rows),
        },
        'interpretation': 'Descriptive conditional linear associations, not causal treatment effects. Query controls remove fixed query difficulty; configuration controls remove mean policy differences. Precision and recall share a hit numerator and change jointly. Scores are bounded and concentrated at endpoints; OLS is a descriptive projection.',
    }
    (OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharex=True, sharey=True)
    colors = ['#8296aa', '#186b8c', '#b7632d']
    for ax, feature in zip(axes, ['precision', 'recall']):
        for i, m in enumerate(MODELS):
            v = fitted[m][feature + '_per_10pp']
            lo, hi = np.array(fitted[m][feature + '_ci95']) * .1
            pos = 2 - i
            ax.errorbar(v, pos, xerr=[[v-lo], [hi-v]], fmt='o', color=colors[i],
                        capsize=4, markersize=7, linewidth=2)
            ax.annotate(f'{v:+.3f}', (v, pos), xytext=(0, 12),
                        textcoords='offset points', ha='center', color=colors[i], fontsize=10)
        ax.axvline(0, color='#555', linestyle='--', linewidth=1)
        ax.set_title(f'+10 percentage points in {feature}', fontsize=11)
        ax.set_xlabel('Associated change in final score (0-1)')
        ax.set_yticks([2, 1, 0], LABELS)
        ax.set_ylim(-.45, 2.6)
        ax.grid(axis='x', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Precision and recall predict less once query differences are controlled', fontsize=13, y=.98)
    fig.text(.5, .015, '2,916 answers | 324 queries × 9 settings | both predictors fitted together | 95% whole-query bootstrap intervals',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .91))
    for ext in ['png', 'svg']:
        fig.savefig(OUT / ('01-regression-effects.' + ext), dpi=200)
    plt.close(fig)

    table = ['| Model | Precision: +10 pp | Recall: +10 pp | R² |', '|---|---:|---:|---:|']
    for m, label in zip(MODELS, LABELS):
        cells = []
        for feature in ['precision', 'recall']:
            lo, hi = np.array(fitted[m][feature + '_ci95']) * .1
            cells.append(f"{fitted[m][feature + '_per_10pp']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        table.append(f"| {label} | {' | '.join(cells)} | {fitted[m]['r2']:.4f} |")
    markdown = '''# Precision/recall regression preview

This analysis reuses all 2,916 saved answers from the
same 324 queries under nine retrieval settings. It makes no model or API calls.

Fit both per-query precision and recall together against final score (0–1).
The main comparison includes a separate intercept for each query, so its slopes
come from changes across settings on the same question. A second model also
controls for each retrieval configuration's mean score. Pooled OLS is a baseline.

## Results

Values are score changes for a +10 percentage-point change in the predictor,
holding the other predictor fixed. Brackets show percentile 95% intervals from
5,000 bootstrap samples of whole queries, keeping all nine settings together.

''' + '\n'.join(table) + '''

R² uses total variation in the first row, within-query variation in the second,
and residual variation after removing query/configuration means in the third.
These denominators differ; the latter two show how little variation the two
retrieval measures explain beyond the controls.

## Interpretation

The pooled associations combine differences between easy/hard questions with
differences between retrieval settings. Query controls greatly reduce both
coefficients. Precision retains a positive conditional association; inspect the
bootstrap bounds for recall before concluding it has a clear association.

These are associations rather than causal estimates. Precision and recall share
the hit numerator, and actual policy changes generally change both. The score
distribution is concentrated at 0 and 1; OLS describes an average linear pattern.
At zero labelled hits, 287 of 1,109 answers still receive full score. The supplied
rubric can reward insufficient-evidence responses; missing qrels or judge errors
are also possible. Retrieval labels alone therefore do not determine this score.

Per-query precision is the relevant fraction of that query's selected list.
The regression does not use the nine configurations' macro-precision means.
No document count/token control is included: those are closely coupled to the
precision/recall ratios and could change the interpretation of their slopes.

## Files and reproduction

- `01-regression-effects.png` / `.svg`: coefficient comparison.
- `results.json`: exact coefficients, bootstrap bounds, partial R² and diagnostics.
- `analyze.py`: saved-data validation, regression and plotting.

Run with NumPy and Matplotlib available. The existing report plot-deps directory
can be supplied through PYTHONPATH. All output remains in this preview directory.
'''
    (OUT / 'README.md').write_text(markdown, encoding='utf-8')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
    print(json.dumps({'models': fitted, 'diagnostics': result['diagnostics'],
                      'report_files_unchanged': len(hashes)}, indent=2))


if __name__ == '__main__':
    main()
