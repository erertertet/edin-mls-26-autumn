"""Recount saved pilot evidence; run in Slurm without new ASR/API inference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

from tokenizers import Tokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--tokenizer", type=Path, required=True)
args = parser.parse_args()
if not os.getenv("SLURM_JOB_ID"):
    raise SystemExit("Run in Slurm")
sys.path.insert(0, str(args.root / "rag"))
from src.data import CourseData

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

selection = json.loads((args.run / "selection.json").read_text())
config = json.loads((args.run / "config.json").read_text())
summary = json.loads((args.run / "summary.json").read_text())
rows = [json.loads(line) for line in (args.run / "retrieval.jsonl").read_text().splitlines()]
ids = selection["selected_ids"]
models = config["models"]
data = CourseData(args.root / "rag/data/rag_corpus")
gold = data.load_qrels("dev")
assert len(ids) == len(set(ids)) == 8
assert len(rows) == 64
assert digest(args.tokenizer) == config["tokenizer"]["sha256"]
assert digest(args.root / "rag/extensions/asr_retrieval_pilot.py") == config["sources"]["rag/extensions/asr_retrieval_pilot.py"]
tokenizer = Tokenizer.from_file(str(args.tokenizer))
cells = {(r["query_id"], r["arm"], r["rule"]): r for r in rows}
assert len(cells) == 64
assert set(cells) == {(q, a, p) for q in ids for a in models for p in ["top5", "cutoff"]}
documents = {}
for row in rows:
    retrieved = row["record_ids"]
    expected_recall = len(set(retrieved).intersection(gold[row["query_id"]])) / len(gold[row["query_id"]])
    assert math.isclose(row["recall"], expected_recall, abs_tol=1e-12)
    assert len(retrieved) == len(set(retrieved)) == row["documents"]
    for rid in retrieved:
        if rid not in documents:
            documents[rid] = len(tokenizer.encode(data.record_text(rid), add_special_tokens=False).ids)
    assert row["context_tokens"] == sum(documents[rid] for rid in retrieved)
    if row["rule"] == "top5":
        assert len(retrieved) == 5 and not row["fallback"]
    elif row["fallback"]:
        assert len(retrieved) == 1 and row["top_similarity"] < config["tau"]
    else:
        assert min(row["selected_scores"]) >= config["tau"]
for row in summary["table"]:
    values = [cells[q, row["arm"], row["rule"]] for q in ids]
    assert math.isclose(row["mean_recall"], math.fsum(r["recall"] for r in values) / 8, abs_tol=1e-12)
    assert row["mean_context_tokens"] == math.fsum(r["context_tokens"] for r in values) / 8
transcripts = {}
api_costs = []
request_ids = []
for path in (args.run / "transcripts").glob("*/*.json"):
    result = json.loads(path.read_text())
    arm = path.parent.name
    qid = result["query_id"]
    assert result["status"] == "complete" and result["model"] == models[arm]
    audio = next(a for a in selection["audio"] if a["query_id"] == qid)
    assert result["audio_sha256"] == audio["audio_sha256"]
    transcripts[qid, arm] = result["transcript"]
    if arm != "glm":
        raw = result["raw_response"]
        cost = (raw.get("usage") or {}).get("cost")
        if cost is not None:
            api_costs.append(float(cost))
        request_id = raw.get("id") or next((v for k, v in result["response_headers"].items() if k in ["x-request-id", "x-generation-id"] and v), None)
        if request_id:
            request_ids.append(request_id)
assert len(transcripts) == 32
assert len(request_ids) == len(set(request_ids))
changed = []
for q in ids:
    if any(cells[q, a, p]["recall"] != cells[q, "glm", p]["recall"] for a in models for p in ["top5", "cutoff"]):
        changed.append({"query_id": q, "transcripts": {a: transcripts[q, a] for a in models},
                        "cells": {a + ":" + p: {k: cells[q, a, p][k] for k in ["recall", "context_tokens", "documents", "top_similarity", "fallback", "record_ids"]} for a in models for p in ["top5", "cutoff"]}})
audit = {"status": "pass", "slurm_job_id": os.environ["SLURM_JOB_ID"], "paired_queries": 8, "transcripts": 32, "recounted_retrieval_cells": 64,
         "unique_documents_retokenized": len(documents), "api_responses_with_cost": len(api_costs), "response_cost_sum_usd": math.fsum(api_costs),
         "distinct_request_ids": len(request_ids), "recall_change_cases": changed,
         "artifact_sha256": {p.name: digest(p) for p in [args.run / "selection.json", args.run / "config.json", args.run / "retrieval.jsonl", args.run / "summary.json"]}}
(args.run / "independent-audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(audit, ensure_ascii=False), flush=True)
