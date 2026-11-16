"""Render compact report figures and tables from completed saved follow-up data."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--evaluation-run',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    summary=json.loads(((a.evaluation_run or a.run)/'summary.json').read_text());front=json.loads((a.run/'frontiers/summary.json').read_text())
    assert summary['complete'] and summary['answers']==summary['judged']==324*len(summary['by_arm'])
    rows=list(csv.DictReader((a.run/'frontiers/sweep.csv').open()))
    colors={'topk':'#24476c','absolute_radius':'#d65f2b','relative_gap':'#16877d'};names={'topk':'Fixed top-k','absolute_radius':'Absolute cutoff','relative_gap':'Relative gap'}
    names.update({f'fixed{k}':f'Fixed top-{k}' for k in [1,3,5,10,20,50]})
    fixed={int(float(r['parameter'])):r for r in rows if r['arm']=='topk'}
    def label_k(ax,x,y,values,small=False):
        sx=100 if 'recall' in x or 'precision' in x else 1
        for k in values:
            r=fixed[k];xx=sx*float(r[x]);yy=100*float(r[y])
            ax.scatter(xx,yy,s=10 if small else 20,color=colors['topk'],zorder=5)
            # Alternate offsets in crowded regions; label the curve, not the cutoffs.
            if x=='mean_evidence_tokens':
                offsets={1:(4,4),2:(-5,5),3:(4,-11),5:(-4,7),10:(-5,7),20:(-8,7),30:(-6,7),40:(-5,7),50:(-12,-14)}
            else:
                offsets={1:(3,6),2:(3,6),3:(3,6),5:(3,6),10:(3,6),20:(-4,7),30:(-12,-13),40:(-3,9),50:(7,-12)}
            dx,dy=offsets.get(k,(-12,8))
            if k>=1000:dx,dy=-5,8
            if k==1000 and x=='macro_recall':dy=22
            if small and k==50 and x=='macro_recall':dx,dy=-13,6
            if x=='macro_precision':
                dx,dy={1:(-24,0),2:(4,0),3:(4,0),5:(5,-2),10:(5,-2),20:(4,-3),30:(6,-6),40:(6,0),50:(10,5)}.get(k,(5,0))
            ax.annotate(f'k={k:,}',(xx,yy),xytext=(dx,dy),textcoords='offset points',
                        fontsize=6 if small else 8,color=colors['topk'],ha='right' if k>=1000 and x!='macro_precision' else 'left',
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.15),
                        arrowprops=dict(arrowstyle='-',color=colors['topk'],lw=.45),zorder=6)
    def draw(ax,x,y,small=False):
        sx=100 if 'recall' in x or 'precision' in x else 1
        for arm in colors:
            rr=sorted([r for r in rows if r['arm']==arm],key=lambda r:float(r[x]))
            ax.plot([sx*float(r[x]) for r in rr],[100*float(r[y]) for r in rr],color=colors[arm],lw=1.2 if small else 1.7,label=names[arm])
        for r in front['frozen']:ax.scatter(sx*r[x],100*r[y],s=18 if small else 32,marker='D',color=colors[r['arm']],edgecolor='white',lw=.5,zorder=7)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    plt.rcParams.update({'font.size':8,'font.family':'DejaVu Sans','axes.titlesize':8})
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.65))
    for ax,x,y in [(axs[0],'macro_precision','macro_recall'),(axs[1],'mean_evidence_tokens','macro_recall')]:
        draw(ax,x,y,True)
        label_k(ax,x,y,[1,2,3,5,10,20,50],True)
    axs[0].set(xlim=(0,46),ylim=(20,75),xlabel='Macro precision (%)',ylabel='Macro recall (%)',title='(a) Query-weighted quality')
    axs[1].set(xlim=(0,12000),ylim=(20,75),xlabel='Mean retrieved-text tokens',ylabel='Macro recall (%)',title='(b) Coverage per token')
    axs[1].tick_params(axis='x',labelsize=7)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=7)
    fig.tight_layout(rect=(0,.1,1,1),pad=.5)
    fig.savefig(a.out/'uncapped-report-frontiers.pdf');fig.savefig(a.out/'uncapped-report-frontiers.png',dpi=220);plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.titlesize':11})
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    fields=[('macro_precision','macro_recall'),('macro_recall','micro_precision'),('mean_evidence_tokens','macro_recall')]
    for ax,(x,y) in zip(axs,fields):
        draw(ax,x,y);label_k(ax,x,y,[1,2,3,5,10,20,30,40,50])
        if x=='macro_precision':ax.set(xlim=(0,46),ylim=(20,75),xlabel='Macro document precision (%)')
        elif x=='macro_recall':ax.set(xlim=(20,75),ylim=(0,46),xlabel='Macro document recall (%)')
        else:ax.set(xlim=(0,12500),ylim=(20,75),xlabel='Mean retrieved-text tokens')
    for ax,label,title in zip(axs,['Macro recall (%)','Micro precision (%)','Macro recall (%)'],['Query-weighted quality','Pooled retrieved documents','Evidence coverage per token']):ax.set(ylabel=label,title=title)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Uncapped retrieval: labelled top-k settings and similarity cutoffs',fontsize=15)
    fig.tight_layout(rect=(0,.08,1,.94))
    for ext in ['png','pdf','svg']:fig.savefig(a.out/('uncapped-frontiers.'+ext),dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    for ax,(x,y) in zip(axs,[fields[0],fields[2]]):
        draw(ax,x,y)
        if x=='mean_evidence_tokens':ax.set_xscale('log')
        label_k(ax,x,y,[2,10,50,100,1000,24000])
    axs[0].set(xlabel='Macro precision (%)',ylabel='Macro recall (%)');axs[1].set(xlabel='Mean retrieved-text tokens (log scale)',ylabel='Macro recall (%)')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False);fig.tight_layout(rect=(0,.1,1,1))
    fig.savefig(a.out/'uncapped-full-range.pdf');fig.savefig(a.out/'uncapped-full-range.png',dpi=180);plt.close(fig)
    # Tables are generated directly from the final reconciled summary.
    table=[]
    fields=['mean_correctness','mean_grounding','mean_completeness','mean_score','mean_input_tokens','mean_downstream_ms','inference_usd_per_1000']
    best={f:(max if i<4 else min)(r[f] for r in summary['by_arm']) for i,f in enumerate(fields)}
    def formatted(r):
        return [f"{r[f]:.2f}" for f in fields[:3]]+[f"{r['mean_score']:.3f}",f"{r['mean_input_tokens']:.0f}",f"{r['mean_downstream_ms']/1000:.2f}",f"{r['inference_usd_per_1000']:.3f}"]
    for r in summary['by_arm']:
        name='Fixed top-2' if r['arm']=='topk' else names[r['arm']]
        values=[r'\textbf{'+v+'}' if r[f]==best[f] else v for f,v in zip(fields,formatted(r))]
        table.append(' & '.join([name]+values)+r' \\')
    (a.out.parent/'answer-table.tex').write_text('\n'.join(table)+'\n')
    # Standalone readable final answer table.
    headers=['Method','Correctness\n/4','Grounding\n/4','Completeness\n/4','Score\n/1','Input\ntokens','Downstream\nseconds','Generation\n$/1,000']
    cells=[]
    for r in summary['by_arm']:
        cells.append(['Fixed top-2' if r['arm']=='topk' else names[r['arm']]]+formatted(r))
    fig,ax=plt.subplots(figsize=(12,2.8+.4*(len(cells)-3)));ax.axis('off')
    t=ax.table(cellText=cells,colLabels=headers,loc='center',cellLoc='center',colWidths=[.18,.11,.11,.12,.09,.1,.13,.16]);t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,2.0)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor('#d6dce4')
        if i==0:cell.set_facecolor('#24476c');cell.set_text_props(color='white',weight='bold')
        elif i%2:cell.set_facecolor('#f2f5f8')
        if i>0 and j>0 and summary['by_arm'][i-1][fields[j-1]]==best[fields[j-1]]:cell.set_text_props(weight='bold')
    fig.suptitle('Final blind LLM evaluation: 324 queries × '+str(len(cells))+' retrieval settings',fontsize=15,weight='bold')
    fig.text(.5,.84,'Mistral Small 4 generator · Gemini 2.5 Flash judge',ha='center',fontsize=10)
    fig.text(.02,.02,'Bold: best observed value (not significance). Higher scores; lower tokens, time and cost. Latency excludes cached ASR/embedding/retrieval.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(a.out/'final-answer-evaluation.png',dpi=180);fig.savefig(a.out/'final-answer-evaluation.pdf');plt.close(fig)
    print(json.dumps({'tables_generated':True,'api_cost':summary['total_api_usd']}))


if __name__=='__main__':main()
