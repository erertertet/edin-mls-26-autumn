"""Score existing top-50 candidates on the explicitly assigned remote GPU.

No API calls or relevance labels. Uses a pinned public cross-encoder checkpoint.
Run with the GPU-enabled mls-asr Python on fuji3, not the CPU RAG environment.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-cc1fade4-7632-01cc-a9ea-77e9dcb519cd"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--corpus",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Assigned remote GPU unavailable; no CPU/local fallback")
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REVISION,use_safetensors=True).to("cuda").eval()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    corpus={r["record_id"]:r["text"] for r in [json.loads(l) for l in args.corpus.read_text().splitlines()]}
    queries=[json.loads(l) for l in (args.run/"candidates.jsonl").read_text().splitlines()]
    output=args.run/"crossencoder-scores.jsonl"
    complete={r["query_id"] for r in [json.loads(l) for l in output.read_text().splitlines()]} if output.exists() else set()
    def predict(query,docs):
        result=[]
        for start in range(0,len(docs),25):
            batch=tokenizer([query]*len(docs[start:start+25]),docs[start:start+25],padding=True,truncation=True,max_length=512,return_tensors="pt")
            batch={k:v.to("cuda") for k,v in batch.items()}
            with torch.inference_mode():
                logits=model(**batch).logits.reshape(-1)
            result.extend(logits.float().cpu().tolist())
        return result
    predict("Warmup query",["Warmup document."]*25)
    started=time.perf_counter()
    for i,q in enumerate(queries):
        if q["query_id"] in complete:
            continue
        docs=[corpus[rid] for rid in q["record_ids"]]
        lengths=tokenizer([q["transcript"]]*len(docs),docs,truncation=False,add_special_tokens=True,return_length=True)["length"]
        torch.cuda.synchronize();begin=time.perf_counter()
        scores=predict(q["transcript"],docs)
        torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-begin)
        row={"query_id":q["query_id"],"scores":scores,"service_ms":elapsed,
             "truncated_pairs":sum(n>512 for n in lengths),"pair_lengths":lengths}
        with output.open("a") as f:f.write(json.dumps(row)+"\n")
        if (i+1)%25==0 or i+1==len(queries):
            print(json.dumps({"scored_queries":i+1,"total":len(queries),"elapsed_s":time.perf_counter()-started}),flush=True)
    metadata={"model":MODEL,"revision":REVISION,"gpu":torch.cuda.get_device_name(0),
              "gpu_uuid":os.environ["CUDA_VISIBLE_DEVICES"],"dtype":"float32","max_pair_tokens":512,"batch_size":25,
              "torch_version":torch.__version__,"api_calls":0,"notes":"Latency includes tokenization, GPU transfers and forward passes over 50 pairs; excludes loading and diagnostic untruncated token-length counting."}
    (args.run/"crossencoder-metadata.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)


if __name__=="__main__":main()
