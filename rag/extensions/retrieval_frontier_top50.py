"""CPU-only top-50 frontier: coarse fixed-k points and exact cutoff sweeps.

Reuse cached ASR/query vectors. Do not call any model/API or resume inference.
All prefix costs are needed internally for variable cutoff/oracle selections;
fixed-k comparison points are only 5,10,...,50 plus the top-1 fallback anchor.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, kron
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import CourseData
from src.context_builder import SimpleContextBuilder
from src.retriever import ExactDenseRetriever
from retrieval_oracle_frontier import hull, query_oracle

SYSTEM = "Use the supplied evidence to answer the user's question. Do not invent facts that are unsupported by the evidence."


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def threshold_sweep(values, cost, recall, precision, descending, family):
    """Exact global threshold breakpoints, updating tied scores together."""
    n, width = cost.shape
    events = sorted([(float(values[q,k]),q,k) for q in range(n) for k in range(1,width)], reverse=descending)
    selected = np.zeros(n, dtype=int)
    output = [{"family":family,"parameter":float(values.max()+1e-6 if descending else -1e-6),
               "mean_input_tokens":float(cost[:,0].mean()),"macro_recall":float(recall[:,0].mean()),
               "macro_precision":float(precision[:,0].mean()),"mean_documents":1.0}]
    pos = 0
    while pos < len(events):
        threshold = events[pos][0]
        updates = {}
        while pos < len(events) and events[pos][0] == threshold:
            _,q,k=events[pos]; updates[q]=max(updates.get(q,0),k);pos+=1
        dc=dr=dp=dk=0.0
        for q,k in updates.items():
            old=selected[q]
            dc+=cost[q,k]-cost[q,old];dr+=recall[q,k]-recall[q,old]
            dp+=precision[q,k]-precision[q,old];dk+=k-old
            selected[q]=k
        previous=output[-1]
        output.append({"family":family,"parameter":threshold,
                       "mean_input_tokens":previous["mean_input_tokens"]+dc/n,
                       "macro_recall":previous["macro_recall"]+dr/n,
                       "macro_precision":previous["macro_precision"]+dp/n,
                       "mean_documents":previous["mean_documents"]+dk/n})
    assert np.allclose([output[-1]["mean_input_tokens"],output[-1]["macro_recall"]],[cost[:,-1].mean(),recall[:,-1].mean()])
    for i in np.linspace(0,len(output)-1,21,dtype=int):
        threshold=output[i]["parameter"]
        k=np.clip((values>=threshold if descending else values<=threshold).sum(axis=1),1,width)-1
        assert np.allclose([cost[np.arange(n),k].mean(),recall[np.arange(n),k].mean()],
                           [output[i]["mean_input_tokens"],output[i]["macro_recall"]])
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    old=np.load(args.run/"retrieval-frontier/query-prefix-metrics.npz")
    ids=old["query_ids"].tolist();n=len(ids);width=50
    queries={q["query_id"]:q for q in read_rows(args.run/"queries.jsonl")}
    meta=json.loads((args.run/"tokenizer-metadata.json").read_text())
    tokenizer=Tokenizer.from_file(hf_hub_download(meta["repo"],"tokenizer.json",revision=meta["revision"]))
    overhead=len(tokenizer.encode(SYSTEM,add_special_tokens=False).ids)+17
    builder=SimpleContextBuilder();data=CourseData(ROOT/"data/rag_corpus")
    qrels=data.load_qrels("practice")
    emb=ROOT/"data/rag_corpus/embeddings"
    retriever=ExactDenseRetriever(emb/"corpus_enc_a.npy",emb/"corpus_enc_a_ids.json")
    cost=np.zeros((n,width),dtype=np.int32);recall=np.zeros((n,width));precision=np.zeros((n,width));scores=np.zeros((n,width))
    retokenized_errors=[];ranking_tie_reorders=0;max_score_error=0.0
    for i,qid in enumerate(ids):
        q=queries[qid]
        found=retriever.retrieve(np.array(q["vector"],dtype=np.float32),k=width)
        score_map=dict(zip(found["record_ids"],found["scores"]))
        assert set(q["record_ids"]).issubset(score_map)
        for j,rid in enumerate(q["record_ids"]):
            max_score_error=max(max_score_error,abs(score_map[rid]-q["similarities"][j]))
        assert max_score_error<1e-6
        if found["record_ids"][:20] != q["record_ids"]:
            # np.argpartition may order equal scores differently for k=20 vs 50.
            for j,rid in enumerate(q["record_ids"]):
                assert abs(score_map[rid]-found["scores"][j])<1e-6
            ranking_tie_reorders+=1
        # Preserve all previously evaluated top-20 IDs/order, append the next 30.
        selected=q["record_ids"]+[rid for rid in found["record_ids"] if rid not in set(q["record_ids"])]
        assert len(selected)==width and len(set(selected))==width
        scores[i]=q["similarities"]+[score_map[rid] for rid in selected[20:]]
        assert np.all(np.diff(scores[i])<=1e-7)
        if not np.all(np.diff(scores[i])<=0):
            raise RuntimeError("Score rounding changes a prefix boundary; inspect before proceeding")
        cost[i,:20]=old["input_tokens"][i]
        k20=overhead+len(tokenizer.encode(builder.build(q["transcript"],selected[:20],data)["prompt"],add_special_tokens=False).ids)
        retokenized_errors.append(k20-int(cost[i,19]))
        for k in range(21,width+1):
            prompt=builder.build(q["transcript"],selected[:k],data)["prompt"]
            cost[i,k-1]=overhead+len(tokenizer.encode(prompt,add_special_tokens=False).ids)
        relevant=set(qrels[qid]);hits=0
        for k,rid in enumerate(selected,1):
            hits+=rid in relevant;recall[i,k-1]=hits/len(relevant);precision[i,k-1]=hits/k
        with (args.out/"queries-top50.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps({"query_id":qid,"record_ids":selected,"similarities":scores[i].tolist(),"retrieval_ms":found["latency_ms"]})+"\n")
        if (i+1)%25==0 or i+1==n:
            print(json.dumps({"prepared":i+1,"total":n,"api_calls":0}),flush=True)
    assert max(abs(v) for v in retokenized_errors)==0
    assert np.allclose(recall[:,:20],old["recall"])
    assert np.allclose(precision[:,:20],old["precision"])
    assert np.all(np.diff(cost,axis=1)>0)
    np.savez_compressed(args.out/"query-prefix-metrics.npz",query_ids=np.array(ids),input_tokens=cost,recall=recall,precision=precision,similarities=scores)
    fixed=[]
    for k in [1]+list(range(5,51,5)):
        fixed.append({"family":"fixed_k","parameter":k,"mean_input_tokens":float(cost[:,k-1].mean()),
                      "macro_recall":float(recall[:,k-1].mean()),"macro_precision":float(precision[:,k-1].mean()),"mean_documents":float(k)})
    absolute=threshold_sweep(scores,cost,recall,precision,True,"absolute_radius")
    gaps=scores[:,:1]-scores
    relative=threshold_sweep(gaps,cost,recall,precision,False,"relative_gap")
    ox,oy,allocation,_=query_oracle(cost,recall)
    allowed=[np.array(sorted(set([0,width-1]+[k for k in range(width-1) if scores[q,k]>scores[q,k+1]]))) for q in range(n)]
    sx,sy,sallocation,_=query_oracle(cost,recall,allowed)
    grid=np.unique(np.r_[ox,sx]);oracle_difference=float(np.max(np.abs(np.interp(grid,ox,oy)-np.interp(grid,sx,sy))))
    # Independent LP validation of the expanded oracle, using three budgets.
    eq=kron(eye(n,format="csr"),csr_matrix(np.ones((1,width))),format="csr")
    lp_checks=[]
    for k in [5,10,25]:
        budget=float(cost[:,k-1].mean())
        result=linprog(-recall.flatten()/n,A_ub=csr_matrix(cost.reshape(1,-1)/n),b_ub=[budget],A_eq=eq,b_eq=np.ones(n),bounds=(0,None),method="highs")
        expected=float(np.interp(budget,ox,oy));assert result.success and abs(-result.fun-expected)<1e-8
        lp_checks.append({"budget":budget,"curve_recall":expected,"lp_recall":float(-result.fun)})
    params=json.loads((args.run/"frozen-parameters.json").read_text())
    frozen=[]
    for family,param,kk in [("absolute_radius",params["tau"],np.clip((scores>=params["tau"]).sum(axis=1),1,width)),
                            ("relative_gap",params["delta"],np.clip((gaps<=params["delta"]).sum(axis=1),1,width))]:
        index=kk-1
        frozen.append({"family":family,"parameter":param,"mean_input_tokens":float(cost[np.arange(n),index].mean()),
                       "macro_recall":float(recall[np.arange(n),index].mean()),"macro_precision":float(precision[np.arange(n),index].mean()),
                       "mean_documents":float(kk.mean()),"queries_selecting_more_than_20":int((kk>20).sum())})
    fx=np.array([r["mean_input_tokens"] for r in fixed]);fy=np.array([r["macro_recall"] for r in fixed]);fh=hull(fx,fy)
    for r in fixed+frozen:
        r["oracle_recall_same_budget"]=float(np.interp(r["mean_input_tokens"],ox,oy))
        r["oracle_gap_pp"]=100*(r["oracle_recall_same_budget"]-r["macro_recall"])
    comparison={}
    for name,rs in [("absolute",absolute),("relative",relative)]:
        x=np.array([r["mean_input_tokens"] for r in rs]);y=np.array([r["macro_recall"] for r in rs]);hh=hull(x,y)
        comparison[name]=[{"fixed_k":int(r["parameter"]),"tokens":r["mean_input_tokens"],
                            "fixed_recall":r["macro_recall"],"cutoff_envelope_recall":float(np.interp(r["mean_input_tokens"],x[hh],y[hh]))} for r in fixed]
    summary={"queries":n,"candidate_pool":width,"fixed_k_values":list(range(5,51,5)),"top1_fallback_anchor":True,
             "absolute_settings":len(absolute),"relative_settings":len(relative),"fixed_points":fixed,"frozen_policy_points":frozen,
             "pool_ceiling":float(recall[:,-1].mean()),"previous_top20_ceiling":float(recall[:,19].mean()),
             "oracle_saturation_tokens":float(ox[-1]),"oracle_saturation_mean_documents":float((allocation+1).mean()),
             "oracle_top50_token_saving_fraction":float(1-ox[-1]/cost[:,-1].mean()),"cutoff_envelopes_at_fixed_budgets":comparison,
             "validation":{"retokenized_top20_queries":n,"max_token_error":max(abs(v) for v in retokenized_errors),
                           "cached_top20_recall_precision_match":True,"top20_tie_order_restorations":ranking_tie_reorders,
                           "max_recomputed_score_error":max_score_error,"adjacent_score_ties":int((np.diff(scores,axis=1)==0).sum()),
                           "max_topk_vs_similarity_oracle_difference":oracle_difference,"linear_program_checks":lp_checks},
             "additional_api_calls":0,"notes":[
                 "Same 324 original retrieval-test queries and enc_a embeddings. CPU exact search extends each cached top-20 list to 50.",
                 "Fixed-k comparison points use step 5; top-1 retained only as common fallback anchor. Internal prefix arrays support variable cutoffs and exact oracle calculation.",
                 "Absolute: score>=tau; relative: top-score minus score<=delta. Both use the top-50 pool and top-1 fallback.",
                 "All distinct threshold selection states are enumerated. Curves are descriptive test sweeps; old dev-selected settings are retained without retuning.",
                 "Input token costs use the same pinned Mistral tokenizer and 17-token chat overhead. Counts for k>20 are offline estimates with the previously API-validated format; no new API validation.",
                 "Oracle curves use qrels in hindsight and allow fractional mean-budget allocation. They do not skip documents within a ranking prefix.",
                 "Recall is qrels document recall, not necessary-fact coverage or answer sufficiency. No planner, output-generation or judge cost included.",
                 "The paused evidence-sufficiency inference experiment is separate and is not resumed."]}
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2))
    with (args.out/"sweep.csv").open("w",newline="") as f:
        fields=["family","parameter","mean_input_tokens","macro_recall","macro_precision","mean_documents"]
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(fixed+absolute+relative)
    with (args.out/"oracle-curves.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["family","mean_input_tokens","macro_recall"])
        for name,x,y in [("query_topk",ox,oy),("query_similarity",sx,sy)]:
            w.writerows((name,float(xx),float(yy)) for xx,yy in zip(x,y))
    np.savez_compressed(args.out/"oracle-allocation.npz",query_ids=np.array(ids),topk_saturation_k=allocation+1,similarity_saturation_k=sallocation+1)
    plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
    colors={"absolute_radius":"#d65f2b","relative_gap":"#16877d"}
    xmax=float(cost[:,-1].mean())*1.045
    for zoom,name in [(False,"retrieval-frontier-top50"),(True,"retrieval-frontier-top50-zoom")]:
        fig,ax=plt.subplots(figsize=(12,7.8))
        ax.plot(fx,100*fy,"o-",color="#24476c",lw=2.4,ms=5,label="Fixed top-k: 5, 10, ... , 50 (+ top-1 anchor)",zorder=4)
        for family,rs,label in [("absolute_radius",absolute,"Absolute similarity cutoff, top-50 pool"),("relative_gap",relative,"Relative score-gap cutoff, top-50 pool")]:
            ax.plot([r["mean_input_tokens"] for r in rs],[100*r["macro_recall"] for r in rs],color=colors[family],lw=2,label=label)
        ax.plot(np.r_[ox,xmax],100*np.r_[oy,oy[-1]],color="#20262e",lw=2.6,label="Per-query top-k oracle")
        ax.plot(np.r_[sx,xmax],100*np.r_[sy,sy[-1]],color="#bc4c82",lw=1.8,ls=(0,(4,4)),label="Per-query similarity oracle")
        ax.axhline(100*recall[:,19].mean(),ls=":",lw=1,color="#888",label="Previous top-20 recall ceiling")
        for r in fixed:
            if r["parameter"]!=1 and (not zoom or r["mean_input_tokens"]<5100):
                ax.annotate(f"k={int(r['parameter'])}",(r["mean_input_tokens"],100*r["macro_recall"]),xytext=(5,-17),textcoords="offset points",fontsize=9,color="#24476c")
        for r in frozen:
            ax.scatter(r["mean_input_tokens"],100*r["macro_recall"],marker="D",s=65,color=colors[r["family"]],edgecolors="white",zorder=7)
        ax.set(xlabel="Mean generator input tokens per query",ylabel="Relevant-document recall (%)",xlim=(0,5100 if zoom else xmax),ylim=(20,min(100,100*recall[:,-1].mean()+5)))
        ax.grid(alpha=.18);ax.legend(loc="lower right",fontsize=9.5,framealpha=.97)
        fig.suptitle("Retrieval frontier extended to top-50",fontsize=19,weight="bold",y=.97)
        fig.text(.09,.905,"324 queries · same embedding model and corpus · fixed-k step 5 · exact cutoff sweeps",fontsize=11,color="#555")
        fig.text(.09,.035,"Diamonds retain the earlier dev-selected cutoffs; no new tuning. Oracle curves use relevance labels in hindsight.\nCost is generator input tokens, not full pipeline cost. No new API calls; the sufficiency experiment remains paused.",fontsize=9.5,color="#555",linespacing=1.5)
        fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.86)
        for ext in ["png","svg"]:fig.savefig(args.out/f"{name}.{ext}",dpi=180,facecolor="white")
        plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
