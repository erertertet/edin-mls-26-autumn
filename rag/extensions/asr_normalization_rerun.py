"""Re-encode saved raw/normalized pilot transcripts in Slurm, without ASR/API calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import unicodedata

import numpy as np
from tokenizers import Tokenizer

from asr_retrieval_pilot import MODELS, RULES, TAU, freeze, sha, summarize, write_json
from transcript_normalization import POLICY, POLICY_VERSION, normalize_transcript

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import CourseData
from src.encoder import LocalEncoder
from src.metrics import recall_at_k
from src.retriever import ExactDenseRetriever


def retrieve_rows(texts, selected, data, encoder, retriever, tokenizer):
    rows = []
    token_cache = {}
    qrels = data.load_qrels("dev")
    for qid in selected:
        for arm in MODELS:
            text = texts[qid, arm]
            # Encode each cell independently, including identical normalized strings.
            vector = np.asarray(encoder.encode(text), dtype=np.float32)
            norm = np.linalg.norm(vector)
            if not np.isfinite(vector).all() or not norm > 0:
                raise RuntimeError("Invalid query vector")
            similarities = retriever.embeddings @ (vector / norm)
            order = np.argsort(-similarities, kind="stable")
            scores = similarities[order].astype(np.float64)
            if not np.isfinite(scores).all() or not np.all(np.diff(scores) <= 0):
                raise RuntimeError("Invalid ranking")
            matching = int(np.count_nonzero(scores >= TAU))
            for rule, count in [("top5", 5), ("cutoff", max(1, matching))]:
                ids = [retriever.record_ids[int(index)] for index in order[:count]]
                for rid in ids:
                    if rid not in token_cache:
                        token_cache[rid] = len(tokenizer.encode(data.record_text(rid), add_special_tokens=False).ids)
                hits = len(set(ids) & set(qrels[qid]))
                recall = recall_at_k(ids, qrels[qid], count)
                if recall != hits / len(qrels[qid]) or len(set(ids)) != count:
                    raise RuntimeError("Retrieval bookkeeping failed")
                rows.append({"query_id": qid, "arm": arm, "model": MODELS[arm], "rule": rule,
                             "recall": recall, "relevant_hits": hits, "total_relevant": len(qrels[qid]),
                             "context_tokens": sum(token_cache[rid] for rid in ids), "documents": count,
                             "record_ids": ids, "selected_scores": scores[:count].tolist(),
                             "top_similarity": float(scores[0]), "fallback": rule == "cutoff" and matching == 0,
                             "transcript_tokens": len(tokenizer.encode(text, add_special_tokens=False).ids)})
        print(json.dumps({"stage": "retrieved", "query_id": qid}), flush=True)
    return rows


def save_condition(rows, selected, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if not summarize(rows, selected, out, [], {"new_api_requests": 0, "new_api_cost_usd": 0}):
        raise RuntimeError("Incomplete paired retrieval output")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-file", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not socket.getfqdn().endswith(".inf.ed.ac.uk"):
        raise RuntimeError("Run only in an Edinburgh Slurm allocation")
    os.umask(0o077)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original = json.loads((args.raw_run / "config.json").read_text())
    selection = json.loads((args.raw_run / "selection.json").read_text())
    selected = selection["selected_ids"]
    if len(selected) != len(set(selected)) or len(selected) != 8 or original["models"] != MODELS:
        raise RuntimeError("Unexpected original pilot selection/models")
    if original["tau"] != TAU or original["topk"] != 5 or original["query_ids"] != selected:
        raise RuntimeError("Original retrieval parameters changed")
    inputs = {}
    for relative, expected in original["sources"].items():
        if relative.startswith("rag/data/rag_corpus/"):
            path = args.data_dir / relative.removeprefix("rag/data/rag_corpus/")
        elif relative.startswith("rag/models/enc_a/"):
            path = args.encoder_dir / relative.removeprefix("rag/models/enc_a/")
        else:
            path = ROOT.parent / relative
        if sha(path) != expected:
            raise RuntimeError(f"Original source/input changed: {relative}")
        inputs[relative] = expected
    if sha(args.tokenizer_file) != original["tokenizer"]["sha256"]:
        raise RuntimeError("Original tokenizer changed")
    inputs["tokenizer.json"] = sha(args.tokenizer_file)
    for name in ["config.json", "selection.json", "retrieval.jsonl", "summary.json"]:
        inputs["raw-run/" + name] = sha(args.raw_run / name)
    texts = {}
    mappings = []
    for path in sorted((args.raw_run / "transcripts").glob("*/*.json")):
        value = json.loads(path.read_text())
        arm = path.parent.name
        qid = value["query_id"]
        audio = next(item for item in selection["audio"] if item["query_id"] == qid)
        if (arm not in MODELS or value["model"] != MODELS[arm] or value["status"] != "complete"
                or value["audio_sha256"] != audio["audio_sha256"] or (qid, arm) in texts):
            raise RuntimeError("Transcript provenance mismatch")
        text = value["transcript"]
        normalized = normalize_transcript(text)
        if not normalized or normalize_transcript(normalized) != normalized:
            raise RuntimeError("Empty or non-idempotent normalized transcript")
        texts[qid, arm] = text
        relative = str(path.relative_to(args.raw_run))
        inputs["raw-run/" + relative] = sha(path)
        mappings.append({"query_id": qid, "arm": arm, "model": MODELS[arm],
                         "audio_sha256": audio["audio_sha256"], "source_path": relative,
                         "source_sha256": sha(path), "raw": text, "normalized": normalized})
    if set(texts) != {(q, a) for q in selected for a in MODELS}:
        raise RuntimeError("Need all 32 original transcripts")
    sources = {str(p.relative_to(ROOT.parent)): sha(p) for p in [Path(__file__),
               Path(__file__).with_name("transcript_normalization.py"),
               ROOT.parent / "report/asr-rag-normalization-protocol.md",
               ROOT / "tests/test_transcript_normalization.py",
               Path(__file__).with_name("run_asr_normalization_rerun.sbatch")]}
    freeze(out / "config.json", {"policy_version": POLICY_VERSION, "policy": POLICY,
           "query_ids": selected, "models": MODELS, "tau": TAU, "topk": 5,
           "sources": sources, "input_sha256": inputs, "tokenizer": original["tokenizer"],
           "original_source_commit": "75d1a1e0a84a3bbbac294c839888dc90a15ffa93",
           "new_api_requests": 0, "new_asr_inferences": 0, "conditions": ["raw-control", "normalized"]})
    write_json(out / "transcript-mappings.json", mappings)
    versions = {n: {"version": importlib.metadata.version(n), "path": importlib.util.find_spec(m).origin}
                for n, m in [("numpy", "numpy"), ("sentence-transformers", "sentence_transformers"),
                             ("tokenizers", "tokenizers"), ("transformers", "transformers"), ("torch", "torch")]}
    write_json(out / "environment.json", {"utc": datetime.now(timezone.utc).isoformat(),
               "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.getfqdn(),
               "python_version": sys.version, "python": sys.executable, "versions": versions,
               "stdlib": {name: importlib.util.find_spec(name).origin for name in ["re", "unicodedata"]},
               "unicode_version": unicodedata.unidata_version,
               "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
               "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode(),
               "command": sys.argv})
    assert unicodedata.normalize("NFKC", "ＵＫ") == "UK"
    assert unicodedata.category("?") == "Po"
    data = CourseData(args.data_dir)
    retriever = ExactDenseRetriever(args.data_dir / "embeddings/corpus_enc_a.npy",
                                    args.data_dir / "embeddings/corpus_enc_a_ids.json")
    if len(retriever.record_ids) != len(set(retriever.record_ids)) or len(retriever.record_ids) != 24000:
        raise RuntimeError("Unexpected corpus")
    encoder = LocalEncoder(args.encoder_dir, device="cpu")
    tokenizer = Tokenizer.from_file(str(args.tokenizer_file))
    raw = retrieve_rows(texts, selected, data, encoder, retriever, tokenizer)
    saved = {(r["query_id"], r["arm"], r["rule"]): r
             for r in map(json.loads, (args.raw_run / "retrieval.jsonl").read_text().splitlines())}
    maximum_drift = 0.0
    for row in raw:
        previous = saved[row["query_id"], row["arm"], row["rule"]]
        for key in ["record_ids", "documents", "relevant_hits", "total_relevant", "recall", "context_tokens", "fallback"]:
            if row[key] != previous[key]:
                raise RuntimeError(f"Raw reproduction mismatch: {row['query_id']}/{row['arm']}/{row['rule']}/{key}")
        maximum_drift = max(maximum_drift, float(np.max(np.abs(np.array(row["selected_scores"]) - previous["selected_scores"]))))
    write_json(out / "raw-control-check.json", {"status": "pass", "cells": len(raw),
               "exact_ids_metrics_tokens_and_fallback": True, "max_abs_selected_score_drift": maximum_drift})
    save_condition(raw, selected, out / "raw-control")
    normalized_texts = {(m["query_id"], m["arm"]): m["normalized"] for m in mappings}
    normalized = retrieve_rows(normalized_texts, selected, data, encoder, retriever, tokenizer)
    save_condition(normalized, selected, out / "normalized")
    raw_cells = {(r["query_id"], r["arm"], r["rule"]): r for r in raw}
    norm_cells = {(r["query_id"], r["arm"], r["rule"]): r for r in normalized}
    per_query = []
    for qid in selected:
        per_query.append({"query_id": qid,
            "raw_distinct_texts": len({texts[qid, arm] for arm in MODELS}),
            "normalized_distinct_texts": len({normalized_texts[qid, arm] for arm in MODELS}),
            "normalized_recall_range": {p: max(norm_cells[qid, a, p]["recall"] for a in MODELS) - min(norm_cells[qid, a, p]["recall"] for a in MODELS) for p in RULES},
            "normalized_selected_ids_equal": {p: all(norm_cells[qid, a, p]["record_ids"] == norm_cells[qid, "glm", p]["record_ids"] for a in MODELS) for p in RULES}})
    changes = []
    for rule in RULES:
        for arm in MODELS:
            changes.append({"arm": arm, "rule": rule, "paired_queries": 8,
                "recall_change_pp": statistics.mean(100 * (norm_cells[q, arm, rule]["recall"] - raw_cells[q, arm, rule]["recall"]) for q in selected),
                "context_token_change": statistics.mean(norm_cells[q, arm, rule]["context_tokens"] - raw_cells[q, arm, rule]["context_tokens"] for q in selected),
                "queries_with_changed_selected_ids": sum(norm_cells[q, arm, rule]["record_ids"] != raw_cells[q, arm, rule]["record_ids"] for q in selected)})
    write_json(out / "comparison.json", {"status": "complete", "policy_version": POLICY_VERSION,
               "per_query": per_query, "normalized_minus_raw": changes, "new_api_requests": 0,
               "new_api_cost_usd": 0, "slurm_job_id": os.environ["SLURM_JOB_ID"]})
    write_json(out / "completion.json", {"status": "complete", "raw_cells": len(raw),
               "normalized_cells": len(normalized), "slurm_job_id": os.environ["SLURM_JOB_ID"]})


if __name__ == "__main__":
    main()
