"""Offline subset selectors over the same top-50 candidates, dev tuning first.

prepare: cached query vectors -> candidate pool and lexical/token features.
evaluate: consume separately cached local cross-encoder scores; tune on dev,
freeze settings, then evaluate the previously used 324-query test set.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data import CourseData
from src.context_builder import SimpleContextBuilder
from src.retriever import ExactDenseRetriever
from retrieval_oracle_frontier import hull

SYSTEM="Use the supplied evidence to answer the user's question. Do not invent facts that are unsupported by the evidence."
KS=[1,2,3,5,10,15,20,25,30,40,50]
TOKEN_GRID=[500,750,1000,1500,2000,3000,5000]


def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def write(p,v):p.write_text(json.dumps(v,indent=2))
def append(p,v):
    with p.open("a") as f:f.write(json.dumps(v)+"\n")
def terms(s):return re.findall(r"[a-z0-9]+",s.lower())
def descending(s):return np.argsort(-np.asarray(s),kind="stable")


class Experiment:
    def __init__(self,args):
        self.args=args;self.out=args.out;self.out.mkdir(parents=True,exist_ok=True)
        self.data=CourseData(ROOT/"data/rag_corpus");self.builder=SimpleContextBuilder()
        self.partition=json.loads((args.base/"partition.json").read_text())
        self.queries={r["query_id"]:r for r in rows(args.base/"queries.jsonl")}
        meta=json.loads((args.base/"tokenizer-metadata.json").read_text())
        self.tokenizer=Tokenizer.from_file(hf_hub_download(meta["repo"],"tokenizer.json",revision=meta["revision"]))
        self.overhead=len(self.tokenizer.encode(SYSTEM,add_special_tokens=False).ids)+17
        self.cache={}

    def prepare(self):
        if (self.out/"prepare-complete.json").exists():return
        if (self.out/"candidates.jsonl").exists():raise RuntimeError("Partial prepare output; inspect before retrying")
        corpus=self.data.records
        counters={rid:Counter(terms(text)) for rid,text in corpus.items()}
        df=Counter()
        for c in counters.values():df.update(c.keys())
        lengths={rid:sum(c.values()) for rid,c in counters.items()};avg=float(np.mean(list(lengths.values())));N=len(corpus)
        idf={t:np.log1p((N-n+.5)/(n+.5)) for t,n in df.items()}
        token_lengths={rid:len(self.tokenizer.encode(text,add_special_tokens=False).ids) for rid,text in corpus.items()}
        test={q["query_id"]:q for q in rows(self.args.base/"retrieval-frontier-top50/queries-top50.jsonl")}
        emb=ROOT/"data/rag_corpus/embeddings"
        ret=ExactDenseRetriever(emb/"corpus_enc_a.npy",emb/"corpus_enc_a_ids.json")
        all_ids=self.partition["tuning_query_ids"]+self.partition["heldout_test_query_ids"]
        for i,qid in enumerate(all_ids):
            q=self.queries[qid];begin=time.perf_counter()
            if qid in test:
                ids=test[qid]["record_ids"];sim=test[qid]["similarities"]
            else:
                found=ret.retrieve(np.array(q["vector"]),k=50)
                mapping=dict(zip(found["record_ids"],found["scores"]))
                assert set(q["record_ids"]).issubset(mapping)
                ids=q["record_ids"]+[rid for rid in found["record_ids"] if rid not in set(q["record_ids"])]
                sim=q["similarities"]+[mapping[rid] for rid in ids[20:]]
            retrieval_ms=1000*(time.perf_counter()-begin)
            begin=time.perf_counter();qt=set(terms(q["transcript"]))-ENGLISH_STOP_WORDS
            bm=[]
            for rid in ids:
                c=counters[rid];den=1.5*(1-.75+.75*lengths[rid]/avg)
                bm.append(sum(idf.get(t,0)*c[t]*2.5/(c[t]+den) for t in qt if c[t]))
            lexical_ms=1000*(time.perf_counter()-begin)
            append(self.out/"candidates.jsonl",{"query_id":qid,"phase":"dev" if q["source_split"]=="dev" else "test",
                "transcript":q["transcript"],"record_ids":ids,"similarities":sim,"bm25":bm,
                "document_tokens":[token_lengths[rid] for rid in ids],"bm25_ms":lexical_ms,"prepare_retrieval_ms":retrieval_ms})
            if (i+1)%100==0:print(json.dumps({"prepared":i+1,"total":len(all_ids)}),flush=True)
        config={"dev_ids":self.partition["tuning_query_ids"],"test_ids":self.partition["heldout_test_query_ids"],
                "k_values":KS,"tuning_token_grid":TOKEN_GRID,
                "rrf_weights":[0,.25,.5,.75,1],"rrf_constant":60,
                "length_penalties":[.25,.5,1,2,4],"diversity_penalties":[.25,1,4],"score_temperature":1,
                "tuning_objective":"mean dev recall-envelope gain versus dense at fixed token budgets 500..5000; budgets outside a method cost range excluded consistently",
                "notes":["All methods select from the same enc_a top-50 pool. Selected contexts retain original embedding rank order.",
                         "Token-aware scores use log(sigmoid(cross-encoder logit)) minus alpha*log(document tokens). These are heuristic scores, not calibrated probabilities.",
                         "Relevance labels never enter feature generation or cross-encoder inference. Only dev qrels select hyperparameters.",
                         "Test has been used in earlier experiments; results are exploratory follow-up, not a fresh confirmatory test.",
                         "Cost is actual tokenizer count for the selected full prompt; all local scoring latency is reported separately. No generator/API calls.",
                         "Incomplete qrels can penalize unjudged useful documents. Recall does not measure answer sufficiency."]}
        write(self.out/"config.json",config);write(self.out/"prepare-complete.json",{"queries":len(all_ids),"corpus_records":N,"api_calls":0})

    def token_count(self,q,selected):
        selected=tuple(sorted(int(i) for i in selected))
        key=(q["query_id"],selected)
        if key not in self.cache:
            prompt=self.builder.build(q["transcript"],[q["record_ids"][i] for i in selected],self.data)["prompt"]
            self.cache[key]=self.overhead+len(self.tokenizer.encode(prompt,add_special_tokens=False).ids)
        return self.cache[key]

    def orders(self,q,method,param=0):
        dense=np.arange(50)
        if method=="dense":return dense
        if method=="bm25":return descending(q["bm25"])
        if method=="rrf":
            lexical=np.empty(50);lexical[descending(q["bm25"])]=np.arange(1,51)
            return descending((1-param)/(60+np.arange(1,51))+param/(60+lexical))
        if method=="crossencoder":return descending(q["ce"])
        if method=="ce_token":return descending(-np.logaddexp(0,-np.array(q["ce"]))-param*np.log(np.maximum(1,q["document_tokens"])))
        if method=="dense_token":return descending(np.array(q["similarities"])-param*np.log(np.maximum(1,q["document_tokens"])))
        if method=="ce_diverse":
            selected=[int(np.argmax(q["ce"]))]
            while len(selected)<50:
                score=np.array(q["ce"])-param*q["redundancy"][:,selected].max(axis=1)
                score[selected]=-np.inf;selected.append(int(np.argmax(score)))
            return np.array(selected)
        raise ValueError(method)

    def evaluate_order(self,qs,method,param=0):
        c=np.zeros((len(qs),len(KS)));r=np.zeros_like(c);p=np.zeros_like(c)
        selections=[]
        for i,q in enumerate(qs):
            order=self.orders(q,method,param);relevant=set(self.qrels[q["phase"]][q["query_id"]]);selrow=[]
            for j,k in enumerate(KS):
                chosen=sorted(order[:k].tolist());hits=sum(q["record_ids"][v] in relevant for v in chosen)
                c[i,j]=self.token_count(q,chosen);r[i,j]=hits/len(relevant);p[i,j]=hits/k;selrow.append(chosen)
            selections.append(selrow)
        return {"tokens":c,"recall":r,"precision":p,"selections":selections}

    def area(self,value):
        x=value["tokens"].mean(axis=0);y=value["recall"].mean(axis=0);hh=hull(x,y)
        # A token budget permits spending less: extend the maximum-recall plateau.
        if x[hh[0]]>min(TOKEN_GRID):return -1
        return float(np.mean(np.interp(TOKEN_GRID,x[hh],y[hh])))

    def evaluate(self):
        config=json.loads((self.out/"config.json").read_text())
        candidates=rows(self.out/"candidates.jsonl");ce={r["query_id"]:r for r in rows(self.out/"crossencoder-scores.jsonl")}
        assert len(ce)==len(candidates)==449
        embroot=ROOT/"data/rag_corpus/embeddings"
        emb=np.load(embroot/"corpus_enc_a.npy",mmap_mode="r")
        emap={rid:i for i,rid in enumerate(json.loads((embroot/"corpus_enc_a_ids.json").read_text()))}
        for q in candidates:
            q["ce"]=ce[q["query_id"]]["scores"]
            assert len(q["ce"])==50 and np.isfinite(q["ce"]).all()
            v=np.asarray(emb[[emap[rid] for rid in q["record_ids"]]],dtype=float)
            v/=np.linalg.norm(v,axis=1,keepdims=True);q["redundancy"]=v@v.T
        self.qrels={"dev":self.data.load_qrels("dev"),"test":self.data.load_qrels("practice")}
        dev=[q for q in candidates if q["phase"]=="dev"]
        test=[q for q in candidates if q["phase"]=="test"]
        assert len(dev)==125 and len(test)==324
        options={"rrf":config["rrf_weights"],"ce_token":config["length_penalties"],"dense_token":[.001,.003,.01,.03,.1],"ce_diverse":config["diversity_penalties"]}
        tuning=[];frozen={}
        for method,values in options.items():
            for param in values:
                result=self.evaluate_order(dev,method,param)
                tuning.append({"method":method,"parameter":param,"objective":self.area(result)})
                print(json.dumps({"dev_method":method,"parameter":param,"objective":tuning[-1]["objective"]}),flush=True)
            frozen[method]=max([r for r in tuning if r["method"]==method],key=lambda r:r["objective"])["parameter"]
        frozen.update({"frozen_at":datetime.now(timezone.utc).isoformat(),"test_evaluated_at_freeze":0,"dev_grid":tuning})
        write(self.out/"frozen-parameters.json",frozen)
        methods=["dense","bm25","rrf","dense_token","crossencoder","ce_token","ce_diverse"]
        arrays={};table=[]
        for method in methods:
            arrays[method]=self.evaluate_order(test,method,frozen.get(method,0))
            v=arrays[method]
            for j,k in enumerate(KS):
                table.append({"method":method,"k":k,"mean_tokens":float(v["tokens"][:,j].mean()),"macro_recall":float(v["recall"][:,j].mean()),"macro_precision":float(v["precision"][:,j].mean())})
            print(json.dumps({"test_method_complete":method}),flush=True)
        # Check the dense baseline against the previous complete top-50 run.
        old=np.load(self.args.base/"retrieval-frontier-top50/query-prefix-metrics.npz")
        oi={q:i for i,q in enumerate(old["query_ids"].tolist())};ii=[oi[q["query_id"]] for q in test]
        columns=np.array(KS)-1
        assert np.array_equal(arrays["dense"]["tokens"],old["input_tokens"][ii][:,columns])
        assert np.allclose(arrays["dense"]["recall"],old["recall"][ii][:,columns])
        for v in arrays.values():assert np.allclose(v["recall"][:,-1],arrays["dense"]["recall"][:,-1])
        # Paired comparison with the original dense fixed-k mixture envelope.
        rng=np.random.default_rng(20261119);boot=rng.integers(0,len(test),size=(3000,len(test)))
        baseline=arrays["dense"];bx=baseline["tokens"].mean(axis=0);by=baseline["recall"].mean(axis=0);bh=hull(bx,by)
        paired=[]
        for method in methods[1:]:
            v=arrays[method]
            for k in [3,5,10,20]:
                j=KS.index(k);x=v["tokens"][:,j];y=v["recall"][:,j]
                if x.mean()<bx[bh[0]]:continue
                gain=float(y.mean()-np.interp(x.mean(),bx[bh],by[bh]));diff=[]
                for sample in boot:
                    sx=baseline["tokens"][sample].mean(axis=0);sy=baseline["recall"][sample].mean(axis=0);hh=hull(sx,sy)
                    if x[sample].mean()>=sx[hh[0]]:
                        diff.append(float(y[sample].mean()-np.interp(x[sample].mean(),sx[hh],sy[hh])))
                paired.append({"method":method,"k":k,"tokens":float(x.mean()),"recall":float(y.mean()),
                               "dense_mixture_recall_same_tokens":float(np.interp(x.mean(),bx[bh],by[bh])),
                               "gain_pp":100*gain,"gain_bootstrap_95pct_pp":(100*np.percentile(diff,[2.5,97.5])).tolist(),"bootstrap_samples":len(diff)})
        metadata=json.loads((self.out/"crossencoder-metadata.json").read_text())
        latency=np.array([ce[q["query_id"]]["service_ms"] for q in test])
        summary={"dev_queries":len(dev),"test_queries":len(test),"candidate_pool":50,"additional_api_calls":0,
                 "fixed_k_values":KS,"frozen_parameters":frozen,"points":table,"matched_budget_comparisons":paired,
                 "crossencoder":{**metadata,"mean_ms":float(latency.mean()),"median_ms":float(np.median(latency)),"p95_ms":float(np.percentile(latency,95)),
                                 "truncated_pairs":sum(ce[q["query_id"]]["truncated_pairs"] for q in test),"total_pairs":len(test)*50},
                 "bm25_mean_ms":float(np.mean([q["bm25_ms"] for q in test])),
                 "validation":{"dense_tokens_exact_match":True,"dense_recall_match":True,"same_top50_ceiling_all_methods":True,"source_labels_never_in_reranker":True},
                 "notes":config["notes"]+["Matched-cost bootstrap resamples paired queries and recomputes the dense envelope. Intervals are exploratory and not multiplicity-adjusted.","No generated-answer or sufficiency claim is supported by this retrieval-only experiment."]}
        write(self.out/"summary.json",summary)
        with (self.out/"points.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
        with (self.out/"selections.jsonl").open("w") as f:
            for method,v in arrays.items():
                for i,q in enumerate(test):
                    f.write(json.dumps({"query_id":q["query_id"],"method":method,"selected_indices_by_k":dict(zip(map(str,KS),v["selections"][i]))})+"\n")
        np.savez_compressed(self.out/"metrics.npz",query_ids=np.array([q["query_id"] for q in test]),k_values=np.array(KS),**{f"{m}_{field}":v[field] for m,v in arrays.items() for field in ["tokens","recall","precision"]})
        colors={"dense":"#24476c","bm25":"#b09c7c","rrf":"#d65f2b","dense_token":"#999999","crossencoder":"#16877d","ce_token":"#8c4ba0","ce_diverse":"#c2466b"}
        labels={"dense":"Original dense ranking","bm25":"BM25 subset","rrf":"Dense + BM25 fusion","dense_token":"Dense + length penalty","crossencoder":"Cross-encoder subset","ce_token":"Cross-encoder + length penalty","ce_diverse":"Cross-encoder + diversity"}
        plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
        for zoom,name in [(False,"subset-retrieval-frontier"),(True,"subset-retrieval-frontier-zoom")]:
            fig,ax=plt.subplots(figsize=(12,7.8))
            for method in methods:
                v=arrays[method];xx=v["tokens"].mean(axis=0);yy=v["recall"].mean(axis=0)*100
                ax.plot(xx,yy,"o-",lw=2.4 if method in ["dense","crossencoder","ce_token"] else 1.6,ms=4,color=colors[method],label=labels[method],alpha=1 if method in ["dense","crossencoder","ce_token"] else .75)
                if method in ["dense","crossencoder","ce_token"]:
                    for k in [3,5,10,20]:
                        j=KS.index(k)
                        if not zoom or xx[j]<3000:ax.annotate(f"k={k}",(xx[j],yy[j]),xytext=(4,-15 if method=="dense" else 6),textcoords="offset points",fontsize=9,color=colors[method])
            ax.axhline(100*by[-1],color="#666",lw=1,ls=":")
            ax.set(xlabel="Mean generator input tokens per query",ylabel="Relevant-document recall (%)",xlim=(0,3000 if zoom else bx[-1]*1.03),ylim=(0,min(100,100*by[-1]+5)))
            ax.grid(alpha=.18);ax.legend(loc="lower right",fontsize=10,framealpha=.97)
            fig.suptitle("Selecting subsets from the same top-50 candidates",fontsize=18,weight="bold",y=.97)
            fig.text(.09,.905,"125 dev queries for tuning · 324 reused test queries · exact selected-context token counts",fontsize=11,color="#555")
            fig.text(.09,.035,f"Cross-encoder overhead: {latency.mean():.0f} ms/query on fuji3 GPU 1 (50 pairs); not included in the token axis.\nNo paid API calls. Recall is qrels coverage, not answer sufficiency. Curves join tested subset sizes.",fontsize=9.5,color="#555",linespacing=1.5)
            fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.86)
            for ext in ["png","svg"]:fig.savefig(self.out/f"{name}.{ext}",dpi=180,facecolor="white")
            plt.close(fig)
        print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--stage",choices=["prepare","evaluate"],required=True)
    args=p.parse_args();exp=Experiment(args)
    getattr(exp,args.stage)()


if __name__=="__main__":main()
