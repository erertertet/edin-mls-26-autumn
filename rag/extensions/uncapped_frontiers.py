"""Offline full-corpus frontier plots for the frozen three-policy follow-up.

Cost here is additive retrieved-text tokens, not a claim of exact API input size.
The generation experiment separately records actual full prompt usage.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data import CourseData


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--pool',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    a=p.parse_args();out=a.run/'frontiers';out.mkdir(exist_ok=True)
    config=json.loads((a.run/'config.json').read_text());params=json.loads((a.run/'frozen-parameters.json').read_text())
    ranked=np.load(a.pool/'rankings.npz');allq=ranked['query_ids'].tolist();qids=config['evaluation_ids'];take=[allq.index(q) for q in qids]
    indices=ranked['indices'][take];scores=ranked['similarities'][take].astype(np.float64);ids=ranked['corpus_ids'].tolist();n,m=scores.shape
    data=CourseData(ROOT/'data/rag_corpus');qrels=data.load_qrels('practice');mapping={rid:i for i,rid in enumerate(ids)}
    hits=np.empty((n,m),np.int32);ngold=np.array([len(qrels[q]) for q in qids])
    for i,qid in enumerate(qids):
        mask=np.zeros(m,dtype=bool);mask[[mapping[r] for r in qrels[qid]]]=True;hits[i]=np.cumsum(mask[indices[i]])
    meta=json.loads((a.base/'tokenizer-metadata.json').read_text());tok=Tokenizer.from_file(meta['path'])
    doc_tokens=np.array([len(tok.encode(data.record_text(rid),add_special_tokens=False).ids) for rid in ids])
    tokens=np.cumsum(doc_tokens[indices],axis=1);ks=np.arange(1,m+1)
    recall=hits/ngold[:,None];precision=hits/ks
    fixed_r=recall.mean(axis=0);fixed_p=precision.mean(axis=0);fixed_c=tokens.mean(axis=0)
    assert np.all(hits[:,-1]==ngold)
    def selection(arm,value):
        if arm=='topk':return np.full(n,int(value))
        thresholds=np.full(n,value) if arm=='absolute_radius' else scores[:,0]-value
        return np.array([max(1,int(np.searchsorted(-scores[i],-thresholds[i],side='right'))) for i in range(n)])
    def row(arm,value):
        k=selection(arm,value);h=hits[np.arange(n),k-1];r=recall[np.arange(n),k-1];pr=h/k;c=tokens[np.arange(n),k-1]
        return dict(arm=arm,parameter=float(value),mean_documents=float(k.mean()),macro_recall=float(r.mean()),macro_precision=float(pr.mean()),
                    micro_precision=float(h.sum()/k.sum()),micro_recall=float(h.sum()/ngold.sum()),mean_evidence_tokens=float(c.mean()),max_documents=int(k.max()))
    # Full-range quantiles plus a dense high-similarity grid; no cap at any state.
    abs_grid=np.unique(np.r_[np.quantile(scores,np.linspace(0,1,1001)),np.linspace(.75,.99,481),scores.max()+1e-8,params['tau']])
    gaps=scores[:,0,None]-scores
    rel_grid=np.unique(np.r_[np.quantile(gaps,np.linspace(0,1,1001)),np.linspace(0,.1,501),params['delta']])
    fixed_grid=np.unique(np.r_[np.arange(1,51),np.arange(55,201,5),np.arange(250,1001,50),np.arange(2000,m+1,1000),m])
    rows=[row('topk',k) for k in fixed_grid]
    rows += [row('absolute_radius',v) for v in abs_grid]
    rows += [row('relative_gap',v) for v in rel_grid]
    with (out/'sweep.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    frozen=[row('topk',params['fixed_k']),row('absolute_radius',params['tau']),row('relative_gap',params['delta'])]
    # Same-recall fixed-k mixtures, recomputed in each paired query bootstrap.
    comparisons=[];rng=np.random.default_rng(20261124)
    # Frozen points are near rank 2. Limit bootstrap work to a broad bracket,
    # asserting coverage in every replicate rather than silently extrapolating.
    bracket=200;fr=recall[:,:bracket];fp=precision[:,:bracket];fc=tokens[:,:bracket];fh=hits[:,:bracket]
    for point in frozen[1:]:
        k=selection(point['arm'],point['parameter']);r=recall[np.arange(n),k-1];pr=precision[np.arange(n),k-1];c=tokens[np.arange(n),k-1];h=hits[np.arange(n),k-1]
        target=r.mean();mix_k=float(np.interp(target,fixed_r,ks));mix_h=float(np.interp(target,fixed_r,hits.mean(axis=0)))
        base_p=float(np.interp(target,fixed_r,fixed_p));base_c=float(np.interp(target,fixed_r,fixed_c))
        changes=[];invalid=0
        for _ in range(5000):
            ix=rng.integers(0,n,n);mr=fr[ix].mean(axis=0);target_i=r[ix].mean()
            if not mr[0]<=target_i<=mr[-1]:invalid+=1;continue
            bp=np.interp(target_i,mr,fp[ix].mean(axis=0));bc=np.interp(target_i,mr,fc[ix].mean(axis=0))
            bh=np.interp(target_i,mr,fh[ix].mean(axis=0));bk=np.interp(target_i,mr,np.arange(1,bracket+1))
            changes.append([100*(pr[ix].mean()-bp),c[ix].mean()-bc,100*(h[ix].sum()/k[ix].sum()-bh/bk)])
        comparisons.append(dict(arm=point['arm'],matched_macro_recall=float(target),cutoff_macro_precision=point['macro_precision'],
                                fixed_macro_precision=base_p,cutoff_micro_precision=point['micro_precision'],fixed_micro_precision=mix_h/mix_k,
                                cutoff_evidence_tokens=point['mean_evidence_tokens'],fixed_evidence_tokens=base_c,
                                delta_columns=['macro_precision_pp','evidence_tokens','micro_precision_pp'],
                                delta_bootstrap95=np.percentile(changes,[2.5,97.5],axis=0).T.tolist(),valid_bootstraps=len(changes),outside_bracket=invalid))
    summary=dict(queries=n,corpus_size=m,no_document_count_cap=True,frozen=frozen,matched_recall=comparisons,
                 maximum_recall=float(fixed_r[-1]),sweep_points=len(rows),
                 cost_definition='Sum of individually tokenized retrieved document texts; excludes question, system, labels and separators. Actual API input tokens are recorded separately.',
                 notes=['All curves are exploratory on a reused evaluation set. No threshold selected from test curves.',
                        'Fixed-k interpolation is a query-independent mixture, not query-adaptive hindsight selection.',
                        'Cutoffs are prefixes of the embedding ranking, not arbitrary noncontiguous subsets.',
                        'Fallback top-1, common full-corpus ranking, stable corpus-row tie order.',
                        'Bootstrap intervals are unadjusted for multiple comparisons; qrels and judge errors are not captured.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(out/'fixed-prefix-means.npz',k=ks,recall=fixed_r,precision=fixed_p,evidence_tokens=fixed_c)
    colors={'topk':'#24476c','absolute_radius':'#d65f2b','relative_gap':'#16877d'}
    labels={'topk':'Fixed top-k','absolute_radius':'Absolute similarity cutoff','relative_gap':'Relative score-gap cutoff'}
    def plot(ax,x,y,zoom=False):
        for arm in labels:
            rr=sorted([r for r in rows if r['arm']==arm],key=lambda r:r[x])
            scale_x=100 if 'recall' in x else 1;scale_y=100 if y!='mean_evidence_tokens' else 1
            ax.plot([r[x]*scale_x for r in rr],[r[y]*scale_y for r in rr],color=colors[arm],label=labels[arm],lw=1.5)
        for r in frozen:ax.scatter(r[x]*scale_x,r[y]*scale_y,s=30,marker='D',color=colors[r['arm']],edgecolor='white',lw=.5,zorder=4)
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
        if zoom:
            if x=='mean_evidence_tokens':ax.set_xlim(0,12000);ax.set_ylim(20,75)
            else:ax.set_xlim(20,75);ax.set_ylim(0,46)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axs=plt.subplots(1,3,figsize=(14,4.5))
    plot(axs[0],'macro_recall','macro_precision',True);plot(axs[1],'macro_recall','micro_precision',True);plot(axs[2],'mean_evidence_tokens','macro_recall',True)
    for ax in axs[:2]:ax.set_xlabel('Macro document recall (%)')
    axs[0].set_ylabel('Macro precision (%)');axs[1].set_ylabel('Micro precision (%)');axs[2].set_ylabel('Macro recall (%)');axs[2].set_xlabel('Mean retrieved-text tokens')
    for ax,t in zip(axs,['Equal weight per query','Pooled retrieved documents','Evidence coverage per token']):ax.set_title(t)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Uncapped similarity selection: 324 queries, all 24,000 corpus documents')
    fig.tight_layout(rect=(0,.08,1,.94))
    for ext in ['png','pdf','svg']:fig.savefig(out/('uncapped-frontiers.'+ext),dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    plot(axs[0],'macro_recall','macro_precision');plot(axs[1],'mean_evidence_tokens','macro_recall')
    axs[0].set(xlabel='Macro recall (%)',ylabel='Macro precision (%)');axs[1].set(xlabel='Mean retrieved-text tokens (log scale)',ylabel='Macro recall (%)',xscale='log')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False);fig.tight_layout(rect=(0,.1,1,1))
    fig.savefig(out/'full-range.png',dpi=180);fig.savefig(out/'full-range.pdf');plt.close(fig)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
