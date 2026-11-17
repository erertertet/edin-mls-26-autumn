"""Prepare query-paired retrieval curves from audited saved vectors, only in Slurm."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
from tokenizers import Tokenizer

NORMALIZER_SHA256 = "fff6993b2f1ae85db3a65a6c645a6e2ac3ad0ec415eb82af3f73a763de8e9dd2"
ARMS = ("glm", "whisper", "mai", "voxtral")
CONDITIONS = ("raw", "normalized")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def freeze(path, value):
    if path.exists():
        require(read(path) == value, "Refuse different frozen configuration: " + path.name)
    else:
        write_json(path, value)


def write_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, allow_pickle=False, **arrays)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "full-run", "out", "data-dir", "tokenizer", "plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"), "Run only in Edinburgh Slurm")
    os.umask(0o077)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.root / "rag"))
    from src.data import CourseData
    from src.retriever import ExactDenseRetriever

    plan = read(args.plan)
    topk_grid = plan["topk_grid"]
    tau_grid = np.asarray(plan["tau_grid"], dtype=np.float64)
    cases = [tuple(case) for case in plan["cases"]]
    fixed_tau = float(plan["fixed_tau"])
    require(len(topk_grid) == len(set(topk_grid)) == 208
            and all(type(k) is int and 1 <= k <= 24000 for k in topk_grid)
            and topk_grid == sorted(topk_grid) and 5 in topk_grid and 24000 in topk_grid,
            "Unexpected fixed top-k grid")
    require(tau_grid.shape == (99,) and len(set(tau_grid.tolist())) == 99 and np.isfinite(tau_grid).all()
            and np.all(np.diff(tau_grid) > 0), "Unexpected fixed cutoff grid")
    require(math.isclose(float(tau_grid[0]), -1.000001, abs_tol=1e-12, rel_tol=0)
            and math.isclose(float(tau_grid[-1]), 1.000001, abs_tol=1e-12, rel_tol=0), "Cutoff grid endpoints changed")
    require(len(cases) == len(set(cases)) == 8 and set(cases) == {(a, c) for a in ARMS for c in CONDITIONS},
            "Unexpected eight ASR/condition cases")
    require(plan["fixed_k"] == 5 and fixed_tau == 0.8925
            and any(math.isclose(t, fixed_tau, abs_tol=1e-12, rel_tol=0) for t in tau_grid),
            "Original cutoff absent from grid")
    require(plan["targets"] == [0.3, 0.4, 0.5], "Frozen recall targets changed")
    fixed_topk_index = topk_grid.index(5)
    fixed_tau_local_index = next(i for i, t in enumerate(tau_grid) if math.isclose(t, fixed_tau, abs_tol=1e-12, rel_tol=0))
    fixed_tau_index = len(topk_grid) + fixed_tau_local_index

    full_audit = read(args.full_run / "audit.json")
    require(full_audit["status"] == "pass" and not full_audit["errors"], "Full source experiment audit did not pass")
    core_files = ("config.json", "selection.json", "query-vectors.npz", "retrieval.jsonl")
    inputs = {"full-run/audit.json": sha(args.full_run / "audit.json"), "plan.json": sha(args.plan)}
    for name in core_files:
        actual = sha(args.full_run / name)
        require(actual == full_audit["artifact_sha256"][name], "Audited full-run artifact changed: " + name)
        inputs["full-run/" + name] = actual
    original = read(args.full_run / "config.json")
    selection = read(args.full_run / "selection.json")
    normalizer = args.root / "rag/extensions/transcript_normalization.py"
    require(sha(normalizer) == original["normalizer_sha256"] == NORMALIZER_SHA256, "Frozen normalization rule changed")
    inputs["rag/extensions/transcript_normalization.py"] = NORMALIZER_SHA256
    require(sha(args.tokenizer) == original["tokenizer"]["sha256"], "Pinned tokenizer bytes changed")
    inputs["tokenizer.json"] = sha(args.tokenizer)
    # Recheck every original input; this includes corpus/vectors/IDs, qrels, encoder bytes and original code.
    for relative, expected in original["input_sources"].items():
        if relative.startswith("rag/data/rag_corpus/"):
            path = args.data_dir / relative.removeprefix("rag/data/rag_corpus/")
        else:
            path = args.root / relative
        require(sha(path) == expected, "Original source/input changed: " + relative)
        inputs[relative] = expected

    query_ids = selection["query_ids"]
    audios = {row["query_id"]: row for row in selection["audio"]}
    require(len(query_ids) == len(set(query_ids)) == len(audios) == 475 and set(query_ids) == set(audios),
            "Expected complete 475-query selection")
    splits = [audios[q]["split"] for q in query_ids]
    calibration_ids = [q for q in query_ids if audios[q]["split"] == "practice"]
    pilot_ids = selection["pilot_ids"]
    evaluation_ids = [q for q in query_ids if audios[q]["split"] == "dev" and q not in pilot_ids]
    require(len(calibration_ids) == 350 and len(evaluation_ids) == 117 and len(set(pilot_ids)) == 8
            and set(calibration_ids).isdisjoint(evaluation_ids)
            and set(calibration_ids).isdisjoint(pilot_ids) and set(evaluation_ids).isdisjoint(pilot_ids)
            and set(calibration_ids + evaluation_ids + pilot_ids) == set(query_ids), "Partition membership mismatch")
    partition = {"calibration_ids": calibration_ids, "evaluation_ids": evaluation_ids, "pilot_ids": pilot_ids}
    sources = {"prepare_asr_curve_cache.py": sha(__file__),
               "rag/src/data.py": sha(args.root / "rag/src/data.py"),
               "rag/src/retriever.py": sha(args.root / "rag/src/retriever.py")}
    configuration = {"schema_version": 1, "plan": plan, "plan_sha256": sha(args.plan), "source_sha256": sources,
                     "input_sha256": inputs, "normalizer_sha256": NORMALIZER_SHA256,
                     "tokenizer": original["tokenizer"], "query_ids": query_ids, "cases": [list(c) for c in cases],
                     "partition": partition, "asr_calls": 0, "encoder_calls": 0, "api_calls": 0}
    freeze(args.out / "config.json", configuration)
    freeze(args.out / "partition.json", partition)

    data = CourseData(args.data_dir)
    qrels = {split: data.load_qrels(split) for split in set(splits)}
    retriever = ExactDenseRetriever(args.data_dir / "embeddings/corpus_enc_a.npy",
                                    args.data_dir / "embeddings/corpus_enc_a_ids.json")
    require(retriever.embeddings.shape == (24000, 384) and retriever.embeddings.dtype == np.float32
            and len(retriever.record_ids) == len(set(retriever.record_ids)) == 24000, "Unexpected corpus vectors")
    record_rows = {rid: index for index, rid in enumerate(retriever.record_ids)}
    gold = np.empty(len(query_ids), dtype=np.int32)
    gold_rows = []
    for index, qid in enumerate(query_ids):
        positives = qrels[audios[qid]["split"]][qid]
        require(positives and set(positives) <= set(record_rows), "Invalid original positive qrels")
        gold[index] = len(positives)
        gold_rows.append(np.asarray([record_rows[rid] for rid in positives], dtype=np.intp))

    with np.load(args.full_run / "query-vectors.npz", allow_pickle=False) as saved:
        vector_keys, vectors = saved["keys"], saved["vectors"]
    require(vector_keys.shape == (3800, 3) and vectors.shape == (3800, 384) and vectors.dtype == np.float32,
            "Unexpected saved query-vector schema")
    vector_rows = {tuple(key): index for index, key in enumerate(vector_keys.tolist())}
    require(len(vector_rows) == len(vector_keys) and set(vector_rows) == {(q, a, c) for q in query_ids for a, c in cases},
            "Saved query-vector coverage mismatch")
    require(np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5, rtol=0),
            "Saved vectors are not finite normalized float32 vectors")
    previous_rows = [json.loads(line) for line in (args.full_run / "retrieval.jsonl").read_text().splitlines() if line.strip()]
    previous = {(r["query_id"], r["arm"], r["condition"], r["rule"]): r for r in previous_rows}
    require(len(previous) == len(previous_rows) == 7600
            and set(previous) == {(q, a, c, r) for q in query_ids for a, c in cases for r in ("top5", "cutoff")},
            "Audited fixed-policy retrieval coverage mismatch")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    document_tokens = np.empty(24000, dtype=np.int32)
    for index, rid in enumerate(retriever.record_ids):
        count = len(tokenizer.encode(data.record_text(rid), add_special_tokens=False).ids)
        require(0 <= count <= np.iinfo(np.int32).max, "Document token count exceeds storage range")
        document_tokens[index] = count
        if (index + 1) % 4000 == 0:
            print(json.dumps({"stage": "tokenized_documents", "completed": index + 1, "total": 24000}), flush=True)
    write_npz(args.out / "doc-token-cache.npz", record_ids=np.asarray(retriever.record_ids), tokens=document_tokens)

    families = np.asarray(["topk"] * len(topk_grid) + ["cutoff"] * len(tau_grid))
    parameters = np.asarray(list(topk_grid) + tau_grid.tolist(), dtype=np.float64)
    shape = (len(query_ids), len(cases), len(parameters))
    hits = np.empty(shape, dtype=np.int32)
    docs = np.empty(shape, dtype=np.int32)
    tokens = np.empty(shape, dtype=np.int64)
    topk_counts = np.asarray(topk_grid, dtype=np.int32)
    fixed_checked, maximum_score_drift, rankings_checked = 0, 0.0, 0
    fallback_counts = {arm + "/" + condition: 0 for arm, condition in cases}
    # Minimal actual-API check of inclusive threshold semantics before the full replay.
    probe = np.asarray([0.9, 0.8, 0.8, 0.7], dtype=np.float64)
    require(int(np.searchsorted(-probe, -0.8, side="right")) == 3, "Inclusive cutoff API check failed")
    for query_index, qid in enumerate(query_ids):
        relevance = np.zeros(24000, dtype=np.int32)
        relevance[gold_rows[query_index]] = 1
        for case_index, (arm, condition) in enumerate(cases):
            vector = vectors[vector_rows[qid, arm, condition]]
            # These are the audited final float32 vectors. Do not normalize them again.
            similarities = retriever.embeddings @ vector
            order = np.argsort(-similarities, kind="stable")
            scores = similarities[order].astype(np.float64)
            require(np.isfinite(scores).all() and np.all(np.diff(scores) <= 0), "Invalid full ranking")
            prefix_hits = np.cumsum(relevance[order], dtype=np.int32)
            prefix_tokens = np.cumsum(document_tokens[order], dtype=np.int64)
            qualifying = np.searchsorted(-scores, -tau_grid, side="right")
            cutoff_counts = np.maximum(1, qualifying).astype(np.int32)
            counts = np.concatenate((topk_counts, cutoff_counts))
            hits[query_index, case_index] = prefix_hits[counts - 1]
            docs[query_index, case_index] = counts
            tokens[query_index, case_index] = prefix_tokens[counts - 1]
            require(prefix_hits[-1] == gold[query_index] and prefix_tokens[-1] == np.sum(document_tokens, dtype=np.int64),
                    "Full-prefix qrel/token bookkeeping mismatch")
            fixed_fallback = bool(qualifying[fixed_tau_local_index] == 0)
            fallback_counts[arm + "/" + condition] += int(fixed_fallback)
            for rule, point_index in (("top5", fixed_topk_index), ("cutoff", fixed_tau_index)):
                row = previous[qid, arm, condition, rule]
                count = int(docs[query_index, case_index, point_index])
                expected_ids = [retriever.record_ids[int(i)] for i in order[:count]]
                require(row["split"] == audios[qid]["split"] and row["record_ids"] == expected_ids
                        and row["documents"] == count and row["relevant_hits"] == int(hits[query_index, case_index, point_index])
                        and row["total_relevant"] == int(gold[query_index])
                        and row["context_tokens"] == int(tokens[query_index, case_index, point_index])
                        and row["recall"] == int(hits[query_index, case_index, point_index]) / int(gold[query_index])
                        and row["fallback"] == (fixed_fallback if rule == "cutoff" else False),
                        "Original fixed-policy cell did not reproduce exactly: " + qid + "/" + arm + "/" + condition + "/" + rule)
                require(len(row["selected_scores"]) == count, "Original selected-score count mismatch")
                drift = float(np.max(np.abs(scores[:count] - row["selected_scores"])))
                require(drift <= 1e-6 and math.isclose(row["top_similarity"], float(scores[0]), rel_tol=0, abs_tol=1e-6),
                        "Original fixed-policy selected-score drift exceeds tolerance")
                maximum_score_drift = max(maximum_score_drift, drift)
                fixed_checked += 1
            rankings_checked += 1
        if (query_index + 1) % 25 == 0 or query_index + 1 == len(query_ids):
            print(json.dumps({"stage": "curve_rankings", "queries": query_index + 1, "total_queries": len(query_ids),
                              "rankings_checked": rankings_checked, "fixed_cells_checked": fixed_checked}), flush=True)
    require(fixed_checked == 7600 and rankings_checked == 3800, "Incomplete original-result replay")
    require(np.all(hits >= 0) and np.all(hits <= gold[:, None, None]) and np.all(hits <= docs)
            and np.all(docs >= 1) and np.all(docs <= 24000) and np.all(tokens >= 0), "Invalid cached curve values")
    write_npz(args.out / "curve-cache.npz", query_ids=np.asarray(query_ids), splits=np.asarray(splits),
              cases=np.asarray(cases), families=families, parameters=parameters,
              hits=hits, docs=docs, tokens=tokens, gold=gold)
    versions = {name: {"version": importlib.metadata.version(name), "path": importlib.util.find_spec(module).origin}
                for name, module in (("numpy", "numpy"), ("tokenizers", "tokenizers"))}
    artifacts = ("config.json", "partition.json", "curve-cache.npz", "doc-token-cache.npz")
    metadata = {"status": "complete", "schema_version": 1, "utc": datetime.now(timezone.utc).isoformat(),
                "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.getfqdn(),
                "plan_sha256": sha(args.plan), "source_sha256": sources, "input_sha256": inputs,
                "normalizer_sha256": NORMALIZER_SHA256, "queries": len(query_ids), "cases": len(cases),
                "curve_points_per_case": len(parameters), "topk_points": len(topk_grid), "cutoff_points": len(tau_grid),
                "saved_query_vectors_replayed": rankings_checked, "corpus_records": 24000,
                "all_document_texts_tokenized": len(document_tokens), "array_shape": list(shape),
                "dtypes": {"hits": str(hits.dtype), "docs": str(docs.dtype), "tokens": str(tokens.dtype), "gold": str(gold.dtype)},
                "partition_sizes": {key: len(value) for key, value in partition.items()},
                "fixed_policy_reproduction": {"status": "pass", "cells": fixed_checked,
                    "exact_ids_recall_tokens_docs_hits_and_fallback": True,
                    "max_abs_selected_score_drift": maximum_score_drift, "score_tolerance": 1e-6,
                    "fallback_counts_at_fixed_tau": fallback_counts},
                "asr_calls": 0, "encoder_calls": 0, "api_calls": 0,
                "environment": {"python": sys.version, "executable": sys.executable, "packages": versions,
                    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.root).decode().strip(),
                    "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=args.root).decode(),
                    "command": sys.argv},
                "artifact_sha256": {name: sha(args.out / name) for name in artifacts},
                "limitations": ["Practice calibration and remaining-dev evaluation are historical exploratory partitions, not fresh holdout data.",
                    "Uses audited saved vectors; does not independently re-encode text or assess ASR recognition accuracy.",
                    "Document-text tokens exclude question/system text and prompt framing; downstream answer quality is not measured."]}
    write_json(args.out / "metadata.json", metadata)
    print(json.dumps({"status": "complete", "queries": len(query_ids), "shape": list(shape),
                      "fixed_cells_checked": fixed_checked, "max_score_drift": maximum_score_drift}), flush=True)


if __name__ == "__main__":
    main()
