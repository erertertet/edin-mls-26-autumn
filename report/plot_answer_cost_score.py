"""Plot reconciled generation cost against final judge score; no API calls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run/'summary.json').read_text())
    assert summary['complete']
    assert summary['answers'] == summary['judged'] == 324*len(summary['by_arm'])
    assert summary['billed_requests'] == summary['api_requests']
    rows = {r['arm']: r for r in summary['by_arm']}
    groups = [
        ('Fixed top-k', '#24476c', 'o', ['fixed5', 'fixed10', 'fixed20']),
        ('Absolute cutoff', '#d65f2b', 's', ['absolute_radius', 'absolute_088', 'absolute_087']),
        ('Relative gap', '#16877d', '^', ['relative_gap', 'relative_002', 'relative_003']),
    ]
    labels = {
        'fixed5': r'$k=5$', 'fixed10': r'$k=10$', 'fixed20': r'$k=20$',
        'absolute_radius': r'$\tau=0.8925$', 'absolute_088': r'$\tau=0.88$',
        'absolute_087': r'$\tau=0.87$', 'relative_gap': r'$\delta=0.009$',
        'relative_002': r'$\delta=0.02$', 'relative_003': r'$\delta=0.03$',
    }
    offsets = {
        'fixed5': (5, -12), 'fixed10': (6, -10), 'fixed20': (-35, 8),
        'absolute_radius': (6, 5), 'absolute_088': (6, 5), 'absolute_087': (6, 5),
        'relative_gap': (6, 5), 'relative_002': (-38, 6), 'relative_003': (-39, -13),
    }
    assert set(rows) == set(labels)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
    fig, ax = plt.subplots(figsize=(5.5, 2.55))
    exported = []
    for name, color, marker, arms in groups:
        ax.scatter([rows[a]['inference_usd_per_1000'] for a in arms],
                   [rows[a]['mean_score'] for a in arms],
                   color=color, marker=marker, s=33, edgecolor='white', linewidth=.5,
                   label=name, zorder=4)
        for arm in arms:
            r = rows[arm]
            x, y = r['inference_usd_per_1000'], r['mean_score']
            assert abs(x-r['generator_usd']/r['queries']*1000) < 1e-9
            ax.annotate(labels[arm], (x, y), xytext=offsets[arm],
                        textcoords='offset points', fontsize=7.5, color=color,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.3),
                        arrowprops=dict(arrowstyle='-', color=color, lw=.45))
            exported.append(dict(arm=arm, queries=r['queries'],
                                 generation_usd_per_1000=x, mean_judge_score=y))
    ax.set(xlim=(.15, 1.12), ylim=(.51, .665),
           xlabel='Generation cost (USD per 1,000 queries)',
           ylabel='Mean final judge score (0-1)')
    ax.xaxis.set_major_locator(MultipleLocator(.2))
    ax.yaxis.set_major_locator(MultipleLocator(.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', frameon=False, fontsize=7.5)
    fig.tight_layout(pad=.6)
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(args.out/f'answer-cost-score.{ext}', dpi=220)
    plt.close(fig)
    (args.out/'answer-cost-score-data.json').write_text(json.dumps(dict(
        source_run=str(args.run), points=exported,
        cost_definition='Reconciled generation charges per 1,000 queries; excludes judging and cached upstream work.',
        score_definition='Mean of (correctness + grounding + completeness)/12; observed means, not confidence bounds.'
    ), indent=2))
    print('Plotted all nine reconciled cost/score pairs; no new inference.')


if __name__ == '__main__':
    main()
