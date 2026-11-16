"""Plot and summarise already-saved retrieval metrics; no experiments/inference."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    source=args.run/'retrieval-frontier-top50'
    a=np.load(source/'query-prefix-metrics.npz');cost=a['input_tokens'];recall=a['recall'];precision=a['precision'];scores=a['similarities']
    x=cost.mean(axis=0);r=recall.mean(axis=0);pp=precision.mean(axis=0);n=len(cost)
    sweep=list(csv.DictReader((source/'sweep.csv').open()))
    summary=json.loads((source/'summary.json').read_text())
    rng=np.random.default_rng(20261120);samples=rng.integers(0,n,size=(5000,n));checks=[]
    for item in summary['frozen_policy_points']:
        kk=np.maximum(1,(scores>=item['parameter']).sum(axis=1)) if item['family']=='absolute_radius' else np.maximum(1,(scores[:,:1]-scores<=item['parameter']).sum(axis=1))
        c=cost[np.arange(n),kk-1];rr=recall[np.arange(n),kk-1];pval=precision[np.arange(n),kk-1]
        same_recall_p=float(np.interp(rr.mean(),r,pp));same_recall_c=float(np.interp(rr.mean(),r,x))
        same_token_p=float(np.interp(c.mean(),x,pp));same_token_r=float(np.interp(c.mean(),x,r))
        d_recall=[];d_token=[]
        for s in samples:
            fx=cost[s].mean(axis=0);fr=recall[s].mean(axis=0);fp=precision[s].mean(axis=0)
            if fr[0]<=rr[s].mean()<=fr[-1]:d_recall.append(float(pval[s].mean()-np.interp(rr[s].mean(),fr,fp)))
            if fx[0]<=c[s].mean()<=fx[-1]:d_token.append(float(pval[s].mean()-np.interp(c[s].mean(),fx,fp)))
        checks.append({**item,'fixed_at_same_recall':{'macro_precision':same_recall_p,'mean_tokens':same_recall_c},
                       'fixed_at_same_tokens':{'macro_precision':same_token_p,'macro_recall':same_token_r},
                       'precision_gain_same_recall_pp':100*(pval.mean()-same_recall_p),
                       'precision_gain_same_recall_bootstrap95_pp':(100*np.percentile(d_recall,[2.5,97.5])).tolist(),
                       'precision_gain_same_tokens_pp':100*(pval.mean()-same_token_p),
                       'precision_gain_same_tokens_bootstrap95_pp':(100*np.percentile(d_token,[2.5,97.5])).tolist(),
                       'valid_bootstrap_counts':[len(d_recall),len(d_token)],
                       'micro_precision':float((pval*kk).sum()/kk.sum())})
    result={'queries':n,'pool':50,'macro_metrics':True,'comparisons':checks,
            'notes':['All values are computed from previously saved arrays. No new retrieval, inference or tuning.',
                     'Equal-cost/equal-recall baselines interpolate adjacent fixed-k points from k=1..50, equivalent to query-independent random mixtures.',
                     'Paired query bootstrap recomputes the baseline mixture. Exploratory intervals are not multiplicity-adjusted.',
                     'Precision counts unjudged documents as non-relevant; incomplete qrels can understate useful evidence.',
                     'Macro precision improvement is not a guarantee of higher recall, lower token cost or better generated answers.']}
    (args.out/'precision-summary.json').write_text(json.dumps(result,indent=2))
    colors={'fixed_k':'#24476c','absolute_radius':'#d65f2b','relative_gap':'#16877d'}
    labels={'fixed_k':'Fixed top-k','absolute_radius':'Absolute similarity cutoff','relative_gap':'Relative score-gap cutoff'}
    def curves(ax,kind,small=False):
        for family in ['fixed_k','absolute_radius','relative_gap']:
            if family=='fixed_k':xx,yy=(r,pp) if kind=='pr' else (x,r)
            else:
                rows=[v for v in sweep if v['family']==family]
                xx=np.array([float(v['macro_recall' if kind=='pr' else 'mean_input_tokens']) for v in rows]);yy=np.array([float(v['macro_precision' if kind=='pr' else 'macro_recall']) for v in rows])
            ax.plot(xx*(100 if kind=='pr' else 1),100*yy,color=colors[family],lw=1.5 if small else 2.4,label=labels[family])
        for k in [1,2,3,5,10,20,50] if kind=='pr' else [5,10,20,30,40,50]:
            xx=100*r[k-1] if kind=='pr' else x[k-1];yy=100*(pp[k-1] if kind=='pr' else r[k-1])
            ax.scatter(xx,yy,color=colors['fixed_k'],s=12 if small else 24,zorder=5)
            if not small or k in ([1,3,5,20,50] if kind=='pr' else [5,20,50]):
                ax.annotate(f'k={k}',(xx,yy),xytext=(3,-11 if small else -15),textcoords='offset points',fontsize=6.5 if small else 10,color=colors['fixed_k'])
        for v in summary['frozen_policy_points']:
            xx=v['macro_recall']*100 if kind=='pr' else v['mean_input_tokens'];yy=100*v['macro_precision' if kind=='pr' else 'macro_recall']
            ax.scatter(xx,yy,color=colors[v['family']],s=20 if small else 80,marker='D',edgecolors='white',linewidths=.6,zorder=7)
        ax.grid(alpha=.18)
        if kind=='pr':ax.set(xlabel='Macro document recall (%)',ylabel='Macro document precision (%)',xlim=(20,71),ylim=(0,48))
        else:ax.set(xlabel='Mean input tokens',ylabel='Macro document recall (%)',xlim=(0,12500),ylim=(20,73))
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(11.5,7.5));curves(ax,'pr');ax.legend(loc='upper right',fontsize=10)
    fig.suptitle('Precision–recall trade-off: top-k versus cutoffs',fontsize=18,weight='bold',y=.97)
    fig.text(.09,.905,'324 queries · same top-50 pool · higher and farther right is better',fontsize=11,color='#555')
    fig.text(.09,.035,'Diamonds: previously dev-selected cutoffs. Curves use existing saved results; no new experiment.\nMacro averages weight each query equally. Unjudged documents count as non-relevant; this is not answer accuracy.',fontsize=9.5,color='#555',linespacing=1.5)
    fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.86)
    for ext in ['png','svg','pdf']:fig.savefig(args.out/f'retrieval-precision-recall.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    # Compact vector figures sized for the supplied 5.5-inch report text width.
    plt.rcParams.update({'font.size':8})
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.65));curves(axes[0],'cost',True);curves(axes[1],'pr',True)
    axes[0].set_title('(a) Coverage and input cost',fontsize=8);axes[1].set_title('(b) Precision and recall',fontsize=8)
    handles,legend_labels=axes[0].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='lower center',ncol=3,fontsize=7,frameon=False)
    fig.subplots_adjust(left=.09,right=.99,bottom=.27,top=.89,wspace=.38)
    for ext in ['pdf','png']:fig.savefig(args.out/f'report-retrieval-frontiers.{ext}',dpi=220)
    plt.close(fig)
    # Oracle figure from the saved top-50 curve, with no recomputation.
    orows=list(csv.DictReader((source/'oracle-curves.csv').open()))
    o=[v for v in orows if v['family']=='query_topk'];ox=[float(v['mean_input_tokens']) for v in o];oy=[100*float(v['macro_recall']) for v in o]
    fig,ax=plt.subplots(figsize=(5.5,2.1));ax.plot(x,100*r,color=colors['fixed_k'],lw=1.6,label='Fixed top-k')
    ax.plot(ox+[x[-1]],oy+[oy[-1]],color='#222',lw=1.7,label='Query-specific oracle (k or threshold)')
    ax.axhline(100*r[19],color='#777',ls=':',lw=1,label='Top-20 pool ceiling')
    ax.set(xlabel='Mean generator input tokens',ylabel='Macro recall (%)',xlim=(0,12000),ylim=(20,73));ax.grid(alpha=.18);ax.legend(loc='lower right',fontsize=7)
    fig.tight_layout(pad=.4)
    for ext in ['pdf','png']:fig.savefig(args.out/f'report-oracle.{ext}',dpi=220)
    plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
