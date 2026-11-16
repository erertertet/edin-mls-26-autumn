"""CPU exact full-corpus ranking of cached query vectors; no model/API calls."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.retriever import ExactDenseRetriever


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((a.base/'partition.json').read_text())
    ids=plan['tuning_query_ids']+plan['heldout_test_query_ids']
    queries={r['query_id']:r for r in map(json.loads,(a.base/'queries.jsonl').read_text().splitlines())}
    emb=ROOT/'data/rag_corpus/embeddings'
    ret=ExactDenseRetriever(emb/'corpus_enc_a.npy',emb/'corpus_enc_a_ids.json')
    n=len(ret.record_ids);indices=np.empty((len(ids),n),np.int32);scores=np.empty((len(ids),n),np.float32)
    times=[];changed=0
    for i,qid in enumerate(ids):
        q=np.asarray(queries[qid]['vector'],dtype=np.float32);q=q/np.linalg.norm(q)
        start=time.perf_counter();s=ret.embeddings@q;idx=np.argsort(-s,kind='stable')
        times.append(1000*(time.perf_counter()-start));indices[i]=idx;scores[i]=s[idx]
        changed += [ret.record_ids[int(j)] for j in idx[:20]] != queries[qid]['record_ids']
        if (i+1)%100==0:print(json.dumps({'ranked':i+1,'queries':len(ids),'corpus_size':n}),flush=True)
    assert np.all(np.diff(scores,axis=1)<=0)
    np.savez_compressed(a.out/'rankings.npz',query_ids=np.array(ids),corpus_ids=np.array(ret.record_ids),
                        indices=indices,similarities=scores,retrieval_ms=np.array(times))
    meta={'queries':len(ids),'corpus_size':n,'document_count_cap':None,'tie_break':'stable corpus row order',
          'top20_order_differences_vs_prior':changed,'mean_cpu_full_ranking_ms':float(np.mean(times)),
          'cached_queries_sha256':hashlib.sha256((a.base/'queries.jsonl').read_bytes()).hexdigest(),
          'notes':['Full exact cosine ranking on CPU. No ASR/encoder/reranker inference.',
                   'Fresh ranking reused by all three policies. Any old tie ordering is not restored.']}
    (a.out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
