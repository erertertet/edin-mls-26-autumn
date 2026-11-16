import json, sys, time
from pathlib import Path
r=Path.home()/"ragexp"
repo=r/"edin-mls-26-autumn"
sys.path.insert(0,str(repo/"rag"))
from src.asr_adapter import ASRAdapter
from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever
from src.data import CourseData
out=Path((r/"latest-full-adaptive-path.txt").read_text().strip())
plan=json.loads((out/"partition.json").read_text())
data_dir=repo/"rag/data/rag_corpus"
data=CourseData(data_dir)
items={x["query_id"]:x for x in data.load_audio_manifest()}
cache=out/"queries.jsonl"
seen={json.loads(x)["query_id"] for x in cache.read_text().splitlines()} if cache.exists() else set()
encoder=LocalEncoder(repo/"rag/models/enc_a",device="cpu")
ret=ExactDenseRetriever(data_dir/"embeddings/corpus_enc_a.npy",data_dir/"embeddings/corpus_enc_a_ids.json")
with ASRAdapter() as asr:
    asr.prepare()
    warm=asr.transcribe(data_dir/items[plan["tuning_query_ids"][0]]["audio_file"])
    encoder.encode(warm)
    for n,qid in enumerate(plan["tuning_query_ids"],1):
        if qid in seen: continue
        item=items[qid]
        t=time.perf_counter();transcript=asr.transcribe(data_dir/item["audio_file"]);asr_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter();v=encoder.encode(transcript);embedding_ms=(time.perf_counter()-t)*1000
        retrieval=ret.retrieve(v,k=20)
        row={"query_id":qid,"source_split":item["split"],"transcript":transcript,"asr_ms":asr_ms,"embedding_ms":embedding_ms,"vector":v.tolist(),"record_ids":retrieval["record_ids"],"similarities":retrieval["scores"],"retrieval_ms":retrieval["latency_ms"],"asr_cuda":"all"}
        with cache.open("a") as f:f.write(json.dumps(row)+"\n")
        print(f"Dev transcription {n}/{len(plan['tuning_query_ids'])}: {qid}",flush=True)
