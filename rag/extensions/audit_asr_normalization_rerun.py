"""Audit saved normalization evidence in Slurm; no model or network execution."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(actual, expected, message):
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def keyed(rows, fields):
    result = {tuple(row[field] for field in fields): row for row in rows}
    require(len(result) == len(rows), "Duplicate cells")
    return result


def mean(values):
    return math.fsum(values) / 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "raw-run", "run", "data-dir", "tokenizer"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"),
            "Run only inside an Edinburgh Slurm allocation")
    os.umask(0o077)
    sys.path.insert(0, str(args.root / "rag"))
    sys.path.insert(0, str(args.root / "rag/extensions"))
    from src.data import CourseData
    from transcript_normalization import POLICY, POLICY_VERSION, normalize_transcript
    from tokenizers import Tokenizer

    config = read(args.run / "config.json")
    original = read(args.raw_run / "config.json")
    selection = read(args.raw_run / "selection.json")
    ids = selection["selected_ids"]
    models = original["models"]
    rules = ["top5", "cutoff"]
    tau = original["tau"]
    require(len(ids) == len(set(ids)) == 8 and len(models) == 4 and "glm" in models,
            "Expected eight unique queries and four models")
    require(config["models"] == models and config["query_ids"] == ids == original["query_ids"],
            "Changed query/model selection")
    require(config["tau"] == tau == 0.8925 and config["topk"] == original["topk"] == 5,
            "Changed retrieval parameters")
    require(config["policy_version"] == POLICY_VERSION and config["policy"] == POLICY,
            "Changed normalization policy")
    require(config["conditions"] == ["raw-control", "normalized"]
            and config["new_api_requests"] == config["new_asr_inferences"] == 0,
            "Unexpected conditions or inference declaration")
    require(config["tokenizer"] == original["tokenizer"], "Changed tokenizer metadata")
    require(digest(args.tokenizer) == config["tokenizer"]["sha256"], "Changed tokenizer bytes")
    require(config["raw_score_tolerance"] == 1e-6, "Changed score tolerance")
    require(config["original_source_commit"] == "75d1a1e0a84a3bbbac294c839888dc90a15ffa93",
            "Unexpected original source commit")

    def input_path(relative):
        if relative.startswith("raw-run/"):
            return args.raw_run / relative.removeprefix("raw-run/")
        if relative.startswith("rag/data/rag_corpus/"):
            return args.data_dir / relative.removeprefix("rag/data/rag_corpus/")
        if relative == "tokenizer.json":
            return args.tokenizer
        return args.root / relative

    for relative, expected in original["sources"].items():
        require(config["input_sha256"].get(relative) == expected, "Changed source declaration: " + relative)
    for relative, expected in config["input_sha256"].items():
        require(digest(input_path(relative)) == expected, "Input hash mismatch: " + relative)
    for relative, expected in config["sources"].items():
        require(digest(args.root / relative) == expected, "Rerun source hash mismatch: " + relative)
    expected_raw_inputs = {"raw-run/" + name for name in ["config.json", "selection.json", "retrieval.jsonl", "summary.json"]}
    mappings = read(args.run / "transcript-mappings.json")
    mapped = keyed(mappings, ["query_id", "arm"])
    pairs = {(q, a) for q in ids for a in models}
    require(set(mapped) == pairs and len(mappings) == 32, "Need 32 unique transcript mappings")
    audio = {item["query_id"]: item["audio_sha256"] for item in selection["audio"]}
    source_paths = set()
    texts = {"raw-control": {}, "normalized": {}}
    for pair, item in mapped.items():
        qid, arm = pair
        path = args.raw_run / item["source_path"]
        require(path.parent.name == arm, "Source arm directory mismatch")
        value = read(path)
        require(value["query_id"] == qid and value["model"] == item["model"] == models[arm]
                and value["status"] == "complete", "Transcript model/status mismatch")
        require(value["audio_sha256"] == item["audio_sha256"] == audio[qid], "Audio provenance mismatch")
        require(value["transcript"] == item["raw"] and digest(path) == item["source_sha256"],
                "Raw transcript mismatch")
        require(item["normalized"] == normalize_transcript(item["raw"])
                and item["normalized"] and normalize_transcript(item["normalized"]) == item["normalized"],
                "Normalized mapping mismatch/non-idempotence")
        for field in ["raw", "normalized"]:
            require(text_digest(item[field]) == item[field + "_text_sha256"], "Text hash mismatch")
        relative = "raw-run/" + item["source_path"]
        require(config["input_sha256"].get(relative) == item["source_sha256"], "Missing transcript input hash")
        source_paths.add(path.resolve())
        expected_raw_inputs.add(relative)
        texts["raw-control"][pair] = item["raw"]
        texts["normalized"][pair] = item["normalized"]
    require(source_paths == {p.resolve() for p in (args.raw_run / "transcripts").glob("*/*.json")},
            "Unaccounted original transcript files")
    require({name for name in config["input_sha256"] if name.startswith("raw-run/")} == expected_raw_inputs,
            "Raw input manifest mismatch")

    data = CourseData(args.data_dir)
    gold = data.load_qrels("dev")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokens = {}
    expected_cells = {(q, a, p) for q in ids for a in models for p in rules}

    def load_rows(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        cells = keyed(rows, ["query_id", "arm", "rule"])
        require(len(rows) == 64 and set(cells) == expected_cells, "Expected 64 complete unique retrieval cells")
        return cells

    def audit_summary(summary, cells):
        require(summary["status"] == "complete" and not summary["failures"]
                and summary["frozen_queries"] == summary["complete_paired_queries"] == ids
                and summary["retrieval_cells"] == 64, "Incomplete/mismatched summary")
        table = keyed(summary["table"], ["arm", "rule"])
        require(set(table) == {(a, p) for a in models for p in rules}, "Expected eight summary rows")
        for (arm, rule), row in table.items():
            require(row["model"] == models[arm] and row["paired_queries"] == 8, "Summary provenance mismatch")
            values = [cells[q, arm, rule] for q in ids]
            baseline = [cells[q, "glm", rule] for q in ids]
            expected = {
                "mean_recall": mean(r["recall"] for r in values),
                "mean_context_tokens": mean(r["context_tokens"] for r in values),
                "mean_documents": mean(r["documents"] for r in values),
                "fallback_queries": sum(r["fallback"] for r in values),
                "delta_recall_pp_vs_glm": mean(100 * (r["recall"] - b["recall"]) for r, b in zip(values, baseline)),
                "delta_context_tokens_vs_glm": mean(r["context_tokens"] - b["context_tokens"] for r, b in zip(values, baseline)),
            }
            for name, value in expected.items():
                close(row[name], value, "Summary mismatch: " + name)
        interaction = keyed(summary["interaction"], ["arm"])
        require(set(interaction) == {(a,) for a in models if a != "glm"}, "Interaction rows mismatch")
        for (arm,), row in interaction.items():
            close(row["cutoff_minus_top5_delta_recall_pp"],
                  table[arm, "cutoff"]["delta_recall_pp_vs_glm"] - table[arm, "top5"]["delta_recall_pp_vs_glm"],
                  "Interaction mismatch")

    conditions = {}
    for condition in texts:
        cells = load_rows(args.run / condition / "retrieval.jsonl")
        conditions[condition] = cells
        for (qid, arm, rule), row in cells.items():
            retrieved = row["record_ids"]
            scores = row["selected_scores"]
            require(row["model"] == models[arm] and len(retrieved) == len(set(retrieved)) == row["documents"]
                    and len(scores) == len(retrieved) > 0, "Invalid selected records")
            require(all(math.isfinite(s) for s in scores) and all(a >= b for a, b in zip(scores, scores[1:])),
                    "Invalid selected scores")
            close(row["top_similarity"], scores[0], "Top score mismatch")
            hits = len(set(retrieved) & set(gold[qid]))
            require(row["relevant_hits"] == hits and row["total_relevant"] == len(gold[qid]), "Qrel count mismatch")
            close(row["recall"], hits / len(gold[qid]), "Recall mismatch")
            for rid in retrieved:
                if rid not in tokens:
                    tokens[rid] = len(tokenizer.encode(data.record_text(rid), add_special_tokens=False).ids)
            require(row["context_tokens"] == sum(tokens[rid] for rid in retrieved), "Context token mismatch")
            require(row["transcript_tokens"] == len(tokenizer.encode(texts[condition][qid, arm], add_special_tokens=False).ids),
                    "Transcript token mismatch")
            if rule == "top5":
                require(row["documents"] == 5 and not row["fallback"], "Invalid top-5 selection")
            elif row["fallback"]:
                require(row["documents"] == 1 and scores[0] < tau, "Invalid cutoff fallback")
            else:
                require(min(scores) >= tau, "Selected score below cutoff")
        for qid, arm in pairs:
            top = cells[qid, arm, "top5"]
            cutoff = cells[qid, arm, "cutoff"]
            prefix = min(top["documents"], cutoff["documents"])
            require(top["record_ids"][:prefix] == cutoff["record_ids"][:prefix]
                    and top["selected_scores"][:prefix] == cutoff["selected_scores"][:prefix], "Policy prefix mismatch")
        summary = read(args.run / condition / "summary.json")
        audit_summary(summary, cells)
        require(summary["billing"] == {"new_api_requests": 0, "new_api_cost_usd": 0}, "Unexpected billing declaration")

    raw, normalized = conditions["raw-control"], conditions["normalized"]
    previous = load_rows(args.raw_run / "retrieval.jsonl")
    audit_summary(read(args.raw_run / "summary.json"), previous)
    maximum_drift = 0.0
    exact_fields = ["model", "record_ids", "documents", "relevant_hits", "total_relevant", "recall",
                    "context_tokens", "fallback", "transcript_tokens"]
    for key, row in raw.items():
        require(all(row[field] == previous[key][field] for field in exact_fields), "Raw reproduction mismatch")
        require(len(row["selected_scores"]) == len(previous[key]["selected_scores"]), "Raw score length mismatch")
        maximum_drift = max(maximum_drift, *(abs(a - b) for a, b in zip(row["selected_scores"], previous[key]["selected_scores"])))
    require(maximum_drift <= config["raw_score_tolerance"], "Raw scores exceed frozen tolerance")
    control = read(args.run / "raw-control-check.json")
    require(control["status"] == "pass" and control["cells"] == 64
            and control["exact_ids_metrics_tokens_and_fallback"], "Failed raw-control declaration")
    close(control["max_abs_selected_score_drift"], maximum_drift, "Raw score drift declaration mismatch")

    comparison = read(args.run / "comparison.json")
    require(comparison["status"] == "complete" and comparison["policy_version"] == POLICY_VERSION
            and comparison["new_api_requests"] == comparison["new_api_cost_usd"] == 0, "Comparison declaration mismatch")
    per_query = keyed(comparison["per_query"], ["query_id"])
    require(set(per_query) == {(q,) for q in ids}, "Per-query comparison selection mismatch")
    for qid in ids:
        row = per_query[qid,]
        require(row["raw_distinct_texts"] == len({texts["raw-control"][qid, a] for a in models})
                and row["normalized_distinct_texts"] == len({texts["normalized"][qid, a] for a in models}),
                "Distinct transcript count mismatch")
        for rule in rules:
            for left in models:
                for right in models:
                    if texts["normalized"][qid, left] == texts["normalized"][qid, right]:
                        fields = ["record_ids", "documents", "recall", "context_tokens", "selected_scores", "fallback"]
                        require(all(normalized[qid, left, rule][field] == normalized[qid, right, rule][field]
                                    for field in fields), "Equal normalized inputs produced unequal retrieval outputs")
            equal_ids = all(normalized[qid, a, rule]["record_ids"] == normalized[qid, "glm", rule]["record_ids"] for a in models)
            require(row["normalized_selected_ids_equal"][rule] == equal_ids, "ID-equality declaration mismatch")
            recalls = [normalized[qid, a, rule]["recall"] for a in models]
            close(row["normalized_recall_range"][rule], max(recalls) - min(recalls), "Recall-range mismatch")
            for first in models:
                for second in models:
                    if texts["normalized"][qid, first] == texts["normalized"][qid, second]:
                        left, right = normalized[qid, first, rule], normalized[qid, second, rule]
                        require(all(left[field] == right[field] for field in exact_fields if field != "model"),
                                "Equal normalized inputs have inconsistent retrieval metrics/IDs")
                        require(all(abs(a - b) <= 1e-6 for a, b in zip(left["selected_scores"], right["selected_scores"])),
                                "Equal normalized inputs have inconsistent selected scores")
    changes = keyed(comparison["normalized_minus_raw"], ["arm", "rule"])
    require(set(changes) == {(a, p) for a in models for p in rules}, "Change rows mismatch")
    for (arm, rule), row in changes.items():
        require(row["paired_queries"] == 8, "Incomplete change pairs")
        close(row["recall_change_pp"], mean(100 * (normalized[q, arm, rule]["recall"] - raw[q, arm, rule]["recall"]) for q in ids), "Recall-change mismatch")
        close(row["context_token_change"], mean(normalized[q, arm, rule]["context_tokens"] - raw[q, arm, rule]["context_tokens"] for q in ids), "Token-change mismatch")
        require(row["queries_with_changed_selected_ids"] == sum(normalized[q, arm, rule]["record_ids"] != raw[q, arm, rule]["record_ids"] for q in ids), "Changed-ID count mismatch")
    completion = read(args.run / "completion.json")
    require(completion["status"] == "complete" and completion["raw_cells"] == completion["normalized_cells"] == 64,
            "Incomplete rerun")
    require(comparison["slurm_job_id"] == completion["slurm_job_id"] == read(args.run / "environment.json")["slurm_job_id"],
            "Rerun job identifiers disagree")
    artifacts = ["config.json", "transcript-mappings.json", "comparison.json", "completion.json", "raw-control-check.json",
                 "raw-control/retrieval.jsonl", "raw-control/summary.json", "normalized/retrieval.jsonl", "normalized/summary.json"]
    result = {"status": "pass", "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "rerun_slurm_job_id": completion["slurm_job_id"], "policy_version": POLICY_VERSION,
              "paired_queries": 8, "transcript_mappings_checked": 32, "text_hashes_checked": 64,
              "recounted_retrieval_cells": 128, "summary_rows_checked": 16,
              "unique_documents_retokenized": len(tokens), "verified_input_hashes": len(config["input_sha256"]),
              "verified_rerun_source_hashes": len(config["sources"]),
              "max_abs_raw_selected_score_drift": maximum_drift,
              "queries_with_identical_normalized_text": sum(per_query[q,]["normalized_distinct_texts"] == 1 for q in ids),
              "queries_with_equal_normalized_selected_ids": {p: sum(per_query[q,]["normalized_selected_ids_equal"][p] for q in ids) for p in rules},
              "audit_scope": "Saved-source, mapping and accounting recount; no model execution or independent full-ranking reconstruction.",
              "audit_script_sha256": digest(__file__), "tokenizers_version": importlib.metadata.version("tokenizers"),
              "artifact_sha256": {name: digest(args.run / name) for name in artifacts}}
    (args.run / "independent-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
