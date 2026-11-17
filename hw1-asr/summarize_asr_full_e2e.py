"""Aggregate frozen full-ASR numerical traces; render prose only after audit."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket

import numpy as np
import rapidfuzz
from rapidfuzz.distance import Levenshtein


SCHEMA = "asr-full-e2e-summary-v1"
COHORTS = {"all": 475, "practice": 350, "dev": 125, "pilot": 8,
           "remaining_dev": 117, "remaining_all": 467}
NUMERICAL = ("logits", "probs", "audio_embeddings")
EDIT_UNITS = ("tokens", "characters", "words")
VOCABULARY = 59264


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def relative_l2(sum_sq, sum_reference_sq):
    if sum_reference_sq > 0:
        return math.sqrt(sum_sq / sum_reference_sq), True
    return (0.0, True) if sum_sq == 0 else (None, False)


def validate_error_stats(item):
    require(integer(item["elements"]) and item["elements"] > 0, "Invalid numerical element count")
    for key in ("max_abs", "sum_abs", "sum_sq", "sum_reference_sq", "mean_abs", "rms"):
        require(finite(item[key]) and item[key] >= 0, "Invalid numerical sufficient statistic: " + key)
    expected_relative, defined = relative_l2(item["sum_sq"], item["sum_reference_sq"])
    require(item["relative_l2_defined"] is defined, "Relative-L2 definition flag mismatch")
    if defined:
        require(finite(item["relative_l2"]) and item["relative_l2"] >= 0
                and math.isclose(item["relative_l2"], expected_relative, rel_tol=1e-9, abs_tol=1e-18),
                "Relative-L2 sufficient-statistic mismatch")
    else:
        require(item["relative_l2"] is None, "Undefined relative L2 must be null")
    require(math.isclose(item["mean_abs"], item["sum_abs"] / item["elements"], rel_tol=1e-9, abs_tol=1e-18)
            and math.isclose(item["rms"], math.sqrt(item["sum_sq"] / item["elements"]),
                             rel_tol=1e-9, abs_tol=1e-18), "MAE/RMS sufficient-statistic mismatch")


def numerical_summary(records, field):
    values = [record["stats"][field] for record in records]
    for item in values:
        validate_error_stats(item)
    count = len(values)
    elements = sum(item["elements"] for item in values)
    sum_abs = math.fsum(item["sum_abs"] for item in values)
    sum_sq = math.fsum(item["sum_sq"] for item in values)
    sum_reference_sq = math.fsum(item["sum_reference_sq"] for item in values)
    maxima = np.asarray([item["max_abs"] for item in values], dtype=np.float64)
    relatives = [item["relative_l2"] for item in values if item["relative_l2_defined"]]
    pooled_relative, pooled_defined = relative_l2(sum_sq, sum_reference_sq)
    return {"queries": count, "elements": elements, "max_abs": float(np.max(maxima)),
            "mean_query_max_abs": float(np.mean(maxima)), "median_query_max_abs": float(np.median(maxima)),
            "p95_query_max_abs": float(np.percentile(maxima, 95, method="linear")),
            "macro_mean_abs": math.fsum(item["mean_abs"] for item in values) / count,
            "macro_rms": math.fsum(item["rms"] for item in values) / count,
            "macro_relative_l2": math.fsum(relatives) / len(relatives) if relatives else None,
            "relative_l2_defined_queries": len(relatives), "relative_l2_undefined_queries": count - len(relatives),
            "pooled_mean_abs": sum_abs / elements, "pooled_rms": math.sqrt(sum_sq / elements),
            "pooled_relative_l2": pooled_relative, "pooled_relative_l2_defined": pooled_defined,
            "sum_abs": sum_abs, "sum_sq": sum_sq, "sum_reference_sq": sum_reference_sq}


def distance_record(left, right):
    distance = int(Levenshtein.distance(left, right, weights=(1, 1, 1), processor=None))
    maximum = max(len(left), len(right))
    require(0 <= distance <= maximum, "Unit-cost edit distance outside its length bound")
    return {"distance": distance, "max_length": maximum,
            "normalized_distance": distance / maximum if maximum else 0.0}


def query_edits(record):
    reference, cuda = record["reference"], record["cuda"]
    left, right = reference["text_stripped"], cuda["text_stripped"]
    return {"tokens": distance_record(reference["token_ids"], cuda["token_ids"]),
            "characters": distance_record(left, right), "words": distance_record(left.split(), right.split())}


def edit_summary(edits, unit):
    values = [item[unit] for item in edits]
    distances = np.asarray([item["distance"] for item in values], dtype=np.float64)
    sum_edits = sum(item["distance"] for item in values)
    sum_max_length = sum(item["max_length"] for item in values)
    return {"queries": len(values), "sum_edits": sum_edits, "sum_max_length": sum_max_length,
            "mean_edits": float(np.mean(distances)), "median_edits": float(np.median(distances)),
            "p95_edits": float(np.percentile(distances, 95, method="linear")), "max_edits": int(np.max(distances)),
            "macro_normalized_distance": math.fsum(item["normalized_distance"] for item in values) / len(values),
            "pooled_normalized_distance": sum_edits / sum_max_length if sum_max_length else 0.0,
            "zero_queries": sum(item["distance"] == 0 for item in values),
            "both_empty_queries": sum(item["max_length"] == 0 for item in values)}


def agreement_summary(records):
    count = len(records)
    reference_steps = sum(record["reference"]["steps"] for record in records)
    cuda_steps = sum(record["cuda"]["steps"] for record in records)
    max_positions = sum(max(record["reference"]["steps"], record["cuda"]["steps"]) for record in records)
    positional_matches = sum(sum(a == b for a, b in zip(record["reference"]["token_ids"], record["cuda"]["token_ids"]))
                             for record in records)
    covered = sum(record["reference_history"]["coverage_steps"] for record in records)
    argmax_mismatches = sum(record["reference_history"]["argmax_token_mismatch_count"] for record in records)
    comparable = sum(record["greedy"]["comparable_prediction_steps"] for record in records)
    result = {"token_exact_queries": sum(record["greedy"]["token_sequence_equal"] for record in records),
              "text_raw_exact_queries": sum(record["greedy"]["text_raw_equal"] for record in records),
              "text_stripped_exact_queries": sum(record["greedy"]["text_stripped_equal"] for record in records),
              "positional_token_matches": positional_matches, "max_length_token_positions": max_positions,
              "positional_token_identity": positional_matches / max_positions,
              "reference_history_argmax_matches": covered - argmax_mismatches,
              "reference_history_prediction_steps": covered,
              "reference_history_argmax_identity": (covered - argmax_mismatches) / covered,
              "greedy_common_prefix_tokens": sum(record["greedy"]["common_prefix_tokens"] for record in records),
              "greedy_comparable_prediction_steps": comparable,
              "reference_steps": reference_steps, "cuda_steps": cuda_steps,
              "greedy_reference_history_coverage": comparable / reference_steps,
              "reference_history_coverage_steps": covered, "reference_history_coverage_fraction": covered / reference_steps,
              "reference_mean_steps": reference_steps / count, "cuda_mean_steps": cuda_steps / count,
              "reference_max_steps": max(record["reference"]["steps"] for record in records),
              "cuda_max_steps": max(record["cuda"]["steps"] for record in records),
              "replay_queries": sum(record["reference_history"]["tail_rows"] > 0 for record in records),
              "replay_tail_rows": sum(record["reference_history"]["tail_rows"] for record in records),
              "replay_next_token_calls": sum(record["reference_history"]["replay_next_token_calls"] for record in records)}
    for field in ("token", "text_raw", "text_stripped"):
        result[field + "_exact_fraction"] = result[field + "_exact_queries"] / count
    for mode in ("reference", "cuda"):
        result[mode + "_eos_queries"] = sum(record[mode]["eos_reached"] for record in records)
        result[mode + "_at_cap_queries"] = sum(record[mode]["at_cap"] for record in records)
        result[mode + "_cap_without_eos_queries"] = sum(record[mode]["cap_without_eos"] for record in records)
        result[mode + "_empty_text_queries"] = sum(not record[mode]["text_stripped"] for record in records)
    return result


def cohort_summary(records):
    edits = [query_edits(record) for record in records]
    return {"queries": len(records), "numerical": {field: numerical_summary(records, field) for field in NUMERICAL},
            "agreement": agreement_summary(records),
            "edit_distance": {unit: edit_summary(edits, unit) for unit in EDIT_UNITS}}


def inside(base, relative):
    path = (base / relative).resolve()
    path.relative_to(base.resolve())
    return path


def validate_record(record, query, config_sha, config, eos_ids):
    require(record["status"] == "complete" and record["config_sha256"] == config_sha,
            "Incomplete or incompatible query record")
    for key in ("query_id", "split", "audio_file", "audio_sha256", "seconds", "shard"):
        require(record[key] == query[key], "Query record differs from frozen selection: " + key)
    require(set(record["input_fingerprints"]) == {"features", "input_ids", "features_mask"}, "Incomplete input fingerprints")
    for value in record["input_fingerprints"].values():
        require(valid_sha(value["sha256"]) and isinstance(value["dtype"], str)
                and value["shape"] and all(integer(n) and n > 0 for n in value["shape"]), "Invalid input fingerprint")
    cap = config["plan"]["max_new_tokens"]
    for name, mode in (("reference", "none"), ("cuda", "all")):
        item = record[name]
        ids = item["token_ids"]
        require(item["mode"] == mode and isinstance(ids, list) and 0 < len(ids) <= cap
                and all(integer(token) and 0 <= token < VOCABULARY for token in ids), "Invalid greedy token sequence")
        require(item["steps"] == len(ids) and item["terminal_token_id"] == ids[-1], "Greedy length/terminal mismatch")
        eos = ids[-1] in eos_ids
        require(not any(token in eos_ids for token in ids[:-1])
                and item["eos_reached"] is eos and item["at_cap"] is (len(ids) == cap)
                and item["cap_without_eos"] is (len(ids) == cap and not eos)
                and (eos or len(ids) == cap), "EOS/cap accounting mismatch")
        require(isinstance(item["text_raw"], str) and item["text_stripped"] == item["text_raw"].strip(),
                "Decoded output definition changed")
        require(len(item["chosen_probabilities"]) == len(ids)
                and all(finite(value) for value in item["chosen_probabilities"]), "Invalid chosen-probability records")
    left, right = record["reference"]["token_ids"], record["cuda"]["token_ids"]
    prefix = 0
    for one, two in zip(left, right):
        if one != two:
            break
        prefix += 1
    equal = left == right
    comparable = min(len(left), len(right), prefix + 1)
    expected = {"token_sequence_equal": equal,
                "token_mismatch_count": sum(one != two for one, two in zip(left, right)) + abs(len(left) - len(right)),
                "first_token_mismatch_index": None if equal else prefix, "common_prefix_tokens": prefix,
                "comparable_prediction_steps": comparable, "reference_steps": len(left), "cuda_steps": len(right),
                "text_raw_equal": record["reference"]["text_raw"] == record["cuda"]["text_raw"],
                "text_stripped_equal": record["reference"]["text_stripped"] == record["cuda"]["text_stripped"]}
    require(record["greedy"] == expected, "Greedy agreement accounting mismatch")
    history = record["reference_history"]
    control = query["query_id"] in config["plan"]["replay_control_ids"]
    start = 0 if control else comparable if comparable < len(left) else len(left)
    require(history["tail_start"] == start and history["tail_rows"] == len(left) - start
            and history["replay_next_token_calls"] == len(left) - start
            and history["coverage_steps"] == len(left) and history["coverage_complete"] is True
            and history["used_cuda_audio_embeddings"] is True, "Incomplete fixed-reference-history coverage")
    argmax_ids = history["argmax_ids"]
    require(len(argmax_ids) == len(left) and all(integer(token) and 0 <= token < VOCABULARY for token in argmax_ids),
            "Invalid reference-history argmax IDs")
    mismatches = sum(one != two for one, two in zip(left, argmax_ids))
    require(history["argmax_token_mismatch_count"] == mismatches
            and history["argmax_all_equal"] is (mismatches == 0), "Reference-history argmax accounting mismatch")
    if control:
        item = history["replay_control"]
        require(item and item["enabled"] is True and item["compared_steps"] == comparable
                and all(finite(item[key]) and item[key] >= 0 for key in ("logits_max_abs", "probs_max_abs"))
                and all(isinstance(item[key], bool) for key in ("logits_exact_equal", "probs_exact_equal")),
                "Incomplete full-replay control")
    else:
        require(history["replay_control"] is None, "Unexpected additional replay control")
    required = {"reference_logits", "reference_probs", "cuda_greedy_logits", "cuda_greedy_probs",
                "reference_audio_embeddings", "cuda_audio_embeddings", "reference_token_ids", "cuda_token_ids",
                "reference_chosen_probabilities", "cuda_chosen_probabilities"}
    if start < len(left):
        required |= {"cuda_reference_tail_logits", "cuda_reference_tail_probs"}
    shapes = record["trace_shapes"]
    require(set(shapes) == required, "Incomplete trace array inventory")
    for mode, length in (("reference", len(left)), ("cuda_greedy", len(right))):
        require(shapes[mode + "_logits"] == shapes[mode + "_probs"] == [length, VOCABULARY],
                "Incomplete full-vocabulary trace shape")
    for mode, length in (("reference", len(left)), ("cuda", len(right))):
        require(shapes[mode + "_token_ids"] == shapes[mode + "_chosen_probabilities"] == [length],
                "Token trace shape mismatch")
    if start < len(left):
        require(shapes["cuda_reference_tail_logits"] == shapes["cuda_reference_tail_probs"]
                == [len(left) - start, VOCABULARY], "Replay-tail shape mismatch")
    audio_shape = shapes["reference_audio_embeddings"]
    require(audio_shape == shapes["cuda_audio_embeddings"] and len(audio_shape) == 2
            and all(integer(n) and n > 0 for n in audio_shape), "Audio-projection shape mismatch")
    for field in NUMERICAL:
        item = record["stats"][field]
        validate_error_stats(item)
        shape = audio_shape if field == "audio_embeddings" else [len(left), VOCABULARY]
        require(item["shape"] == shape and item["elements"] == math.prod(shape), "Numerical statistic shape mismatch")
    require(record["calls"] == {"model_generate_calls": 2, "audio_encoder_calls": 2,
                                "replay_next_token_calls": len(left) - start}, "Per-query call accounting mismatch")


def load_run(root, run):
    config_path = run / "config.json"
    config, config_sha = read(config_path), sha(config_path)
    require(config["version"] == "asr-full-e2e-v1" and config["expected_queries"] == 475
            and config["shards"] == 4, "Unexpected frozen inference configuration")
    plan = config["plan"]
    require(plan["modes"] == ["none", "all"] and plan["max_new_tokens"] == 100 and plan["vocab_size"] == VOCABULARY
            and plan["precision"] == {"global": "ieee", "matmul": "ieee", "cudnn": "ieee"}
            and plan["bootstrap_replicates"] == 0 and plan["kernel_acceptance_threshold"] is None,
            "Frozen comparison settings changed")
    device = plan["device"]
    require(set(device) == {"name_contains", "major", "minor", "multi_processor_count"}
            and isinstance(device["name_contains"], str) and device["name_contains"]
            and integer(device["major"]) and device["major"] > 0
            and integer(device["minor"]) and 0 <= device["minor"] <= 9
            and integer(device["multi_processor_count"]) and device["multi_processor_count"] > 0
            and isinstance(plan["hostname"], str) and plan["hostname"].endswith(".inf.ed.ac.uk"),
            "Frozen device/hostname context is incomplete")
    queries, cohorts = config["queries"], config["cohorts"]
    require(len(queries) == 475 and len({query["query_id"] for query in queries}) == 475
            and [query["query_id"] for query in queries] == cohorts["all"]
            and set(cohorts) == set(COHORTS), "Frozen full-query coverage mismatch")
    require({name: len(set(ids)) for name, ids in cohorts.items()} == COHORTS
            and all(len(ids) == len(set(ids)) for ids in cohorts.values()), "Frozen cohort counts changed")
    sets = {name: set(ids) for name, ids in cohorts.items()}
    require(not sets["practice"] & sets["dev"] and sets["practice"] | sets["dev"] == sets["all"]
            and not sets["pilot"] & sets["remaining_dev"] and sets["pilot"] | sets["remaining_dev"] == sets["dev"]
            and sets["remaining_all"] == sets["all"] - sets["pilot"], "Frozen cohort membership relationships changed")
    require(plan["replay_control_ids"] == [queries[index]["query_id"] for index in range(4)]
            and all(query["shard"] == index % 4 for index, query in enumerate(queries)), "Frozen shard/control plan changed")
    require(all(query["query_id"] in sets[query["split"]] and query["split"] in ("practice", "dev")
                and finite(query["seconds"]) and query["seconds"] > 0 for query in queries), "Invalid query split/duration")
    for name, expected in config["source_sha256"].items():
        require(valid_sha(expected) and sha(inside(root, name)) == expected, "Frozen inference source changed: " + name)
    require(set(config["input_paths"]) == set(config["input_sha256"]), "Prior input proof inventory mismatch")
    for name, path in config["input_paths"].items():
        require(sha(Path(path)) == config["input_sha256"][name], "Prior audited input changed: " + name)
    library = Path(config["library"]["path"]).resolve()
    require(library == (run / "libasr_kernels.so").resolve()
            and valid_sha(config["library"]["sha256"])
            and sha(library) == config["library"]["sha256"] == plan["library_sha256"],
            "Audited inference library changed")
    build_path = run / "library-build.json"
    require(build_path.resolve() == Path(config["library"]["build_proof_path"]).resolve()
            and sha(build_path) == config["library"]["build_proof_sha256"], "Library build-proof binding changed")
    build = read(build_path)
    expected_sources = {name: expected for name, expected in config["source_sha256"].items()
                        if name == "hw1-asr/Makefile" or name.startswith("hw1-asr/csrc/")}
    architecture = device["major"] * 10 + device["minor"]
    require(expected_sources and "hw1-asr/Makefile" in expected_sources
            and build["status"] == "complete" and build["library_sha256"] == config["library"]["sha256"]
            and build["source_sha256"] == expected_sources and build["hostname"] == plan["hostname"]
            and build["sm"] == architecture == plan["library_build_sm"]
            and str(build["slurm_job_id"]) and isinstance(build["nvcc_version"], str) and build["nvcc_version"]
            and isinstance(build["compiler_version"], str) and build["compiler_version"],
            "Library build source/toolchain/architecture proof mismatch")
    staging = config["staging"]
    proof_path = Path(staging["proof_path"])
    require(sha(proof_path) == staging["proof_sha256"], "Staging proof changed")
    proof = read(proof_path)
    require(proof["status"] == "complete" and proof["hostname"] == plan["hostname"] and proof["audio_files"] == 475
            and proof["model_files_sha256"] == config["model"]["files_sha256"]
            and proof["library_sha256"] == config["library"]["sha256"], "Incomplete staging proof")
    snapshot = Path(staging["snapshot_path"]).resolve()
    require(snapshot.name == config["model"]["snapshot_revision"] == plan["snapshot_revision"], "Pinned model snapshot changed")
    for name, expected in config["model"]["files_sha256"].items():
        require(sha(inside(snapshot, name)) == expected, "Staged model bytes changed: " + name)
    hf_config = read(snapshot / "config.json")
    eos = hf_config["text_config"]["eos_token_id"]
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    require(eos_ids and all(integer(token) and 0 <= token < VOCABULARY for token in eos_ids), "Invalid model EOS configuration")
    input_hashes = {"config.json": config_sha, "staging-proof.json": sha(proof_path),
                    "library-build.json": sha(build_path)}
    shard_jobs = set()
    for shard in range(4):
        path = run / "shards" / (str(shard) + ".json")
        item = read(path)
        expected_ids = [query["query_id"] for query in queries[shard::4]]
        require(item["status"] == "complete" and item["shard"] == shard and item["config_sha256"] == config_sha
                and item["query_ids"] == expected_ids and item["planned_queries"] == item["completed_queries"] == len(expected_ids),
                "Inference shard is incomplete")
        shard_jobs.add(str(item["slurm_job_id"]))
        input_hashes["shards/" + path.name] = sha(path)
    records, record_hashes, trace_hashes = [], {}, {}
    expected_record_names, expected_trace_names = set(), set()
    for query in queries:
        stem = hashlib.sha256(query["query_id"].encode()).hexdigest()[:20]
        record_path = run / "queries" / (stem + ".json")
        record = read(record_path)
        validate_record(record, query, config_sha, config, eos_ids)
        trace_name = "traces/" + stem + ".npz"
        require(record["trace_file"] == trace_name and valid_sha(record["trace_sha256"]), "Trace filename/hash mismatch")
        require(sha(run / trace_name) == record["trace_sha256"], "Saved numerical trace changed")
        audio_path = inside(Path(staging["data_dir"]), query["audio_file"])
        require(sha(audio_path) == query["audio_sha256"] == proof["audio_sha256"][query["audio_file"]], "Frozen query audio changed")
        record_hashes[record_path.name] = sha(record_path)
        trace_hashes[stem + ".npz"] = record["trace_sha256"]
        expected_record_names.add(record_path.name)
        expected_trace_names.add(stem + ".npz")
        records.append(record)
    require({path.name for path in (run / "queries").glob("*.json")} == expected_record_names
            and {path.name for path in (run / "traces").glob("*.npz")} == expected_trace_names,
            "Extra or missing query/trace artifacts")
    jobs = {}
    for path in sorted((run / "jobs").glob("*.json")):
        item = read(path)
        job_id = str(item["slurm_job_id"])
        require(path.stem == job_id and item["config_sha256"] == config_sha and item["source_sha256"] == config["source_sha256"]
                and item["library_sha256"] == config["library"]["sha256"]
                and item["precision"]["after"] == plan["precision"]
                and {key: value["version"] for key, value in item["versions"].items()} == config["versions"],
                "Incompatible inference job metadata")
        require(device["name_contains"] in item["device"]["name"]
                and all(item["device"][key] == device[key] for key in ("major", "minor", "multi_processor_count")),
                "Inference job used another device context")
        for field in ("new_api_calls", "new_external_asr_calls", "new_rag_encoder_calls"):
            require(item[field] == 0, "Unexpected external inference/API activity")
        jobs[job_id] = item
        input_hashes["jobs/" + path.name] = sha(path)
    require(jobs and shard_jobs.issubset(jobs) and all(jobs[job]["status"] == "complete" for job in shard_jobs)
            and all(str(record["slurm_job_id"]) in jobs for record in records), "Missing completed-shard or producing-job metadata")
    for record in records:
        require(jobs[str(record["slurm_job_id"])]["shard"] == record["shard"], "Producing job belongs to another shard")
    # Bind private inventories by digest; do not expose IDs or per-query filenames in public aggregates.
    inventory_sha = lambda values: hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    controls = [record["reference_history"]["replay_control"] for record in records if record["reference_history"]["replay_control"]]
    require(len(controls) == 4, "Expected all four predeclared full-replay controls")
    devices = {json.dumps({key: item["device"][key] for key in
                          ("name", "major", "minor", "multi_processor_count", "total_memory")}, sort_keys=True)
               for item in jobs.values()}
    metadata = {"utc": utc(), "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu_slurm_job_ids": sorted(jobs),
        "inference_versions": config["versions"], "versions": {"numpy": np.__version__, "rapidfuzz": rapidfuzz.__version__},
        "model_snapshot_revision": config["model"]["snapshot_revision"], "library_sha256": config["library"]["sha256"],
        "source_sha256": config["source_sha256"], "input_sha256": input_hashes,
        "private_query_inventory_sha256": inventory_sha(record_hashes), "private_trace_inventory_sha256": inventory_sha(trace_hashes),
        "query_records_checked": 475, "trace_hashes_checked": 475, "audio_hashes_checked": 475,
        "model_file_hashes_checked": len(config["model"]["files_sha256"]), "devices": [json.loads(value) for value in sorted(devices)],
        "completed_query_model_generate_calls": sum(record["calls"]["model_generate_calls"] for record in records),
        "completed_query_audio_encoder_calls": sum(record["calls"]["audio_encoder_calls"] for record in records),
        "recorded_runtime_model_generate_calls": sum(item["model_generate_calls"] for item in jobs.values()),
        "recorded_runtime_audio_encoder_calls": sum(item["audio_encoder_calls"] for item in jobs.values()),
        "runtime_records_by_status": {status: sum(item["status"] == status for item in jobs.values())
                                      for status in sorted({item["status"] for item in jobs.values()})},
        "new_external_asr_calls": 0, "new_api_calls": 0, "new_rag_encoder_calls": 0,
        "summarizer_new_model_calls": 0, "summarizer_new_encoder_calls": 0, "summarizer_new_api_calls": 0,
        "precision": plan["precision"], "max_new_tokens": 100, "bootstrap_replicates": 0,
        "kernel_acceptance_threshold": None, "replay_controls": {"queries": len(controls),
            "compared_steps": sum(item["compared_steps"] for item in controls),
            "logits_max_abs": max(item["logits_max_abs"] for item in controls),
            "probs_max_abs": max(item["probs_max_abs"] for item in controls),
            "logits_numerically_equal_queries": sum(item["logits_exact_equal"] for item in controls),
            "probs_numerically_equal_queries": sum(item["probs_exact_equal"] for item in controls)}}
    return config, records, metadata


def aggregate_run(root, run):
    if (run / "audit.json").exists():
        require(read(run / "audit.json").get("status") != "pass", "Audited aggregates are frozen; use --report-only")
    config, records, metadata = load_run(root, run)
    analysis_config = {"schema": SCHEMA, "inference_config_sha256": metadata["input_sha256"]["config.json"],
        "summarizer_source_sha256": sha(Path(__file__)), "versions": metadata["versions"],
        "cohort_counts": COHORTS, "percentile_method": "linear", "edit_units": list(EDIT_UNITS),
        "edit_weights": [1, 1, 1], "edit_processor": None, "text_definition": config["plan"]["text_definition"],
        "numerical_fields": list(NUMERICAL), "bootstrap_replicates": 0, "kernel_acceptance_threshold": None}
    analysis_path = run / "analysis-config.json"
    if analysis_path.exists():
        require(read(analysis_path) == analysis_config, "Frozen aggregation configuration/source changed")
    else:
        write(analysis_path, analysis_config)
    metadata["input_sha256"]["analysis-config.json"] = sha(analysis_path)
    metadata["summarizer_source_sha256"] = analysis_config["summarizer_source_sha256"]
    by_id = {record["query_id"]: record for record in records}
    data = {"status": "complete", "schema": SCHEMA, "metadata": metadata,
            "cohorts": {name: cohort_summary([by_id[qid] for qid in config["cohorts"][name]]) for name in COHORTS}}
    write(run / "summary.json", data)
    write_csv(data, run / "asr-full-e2e.csv")
    completion = {"status": "complete", "slurm_job_id": os.environ["SLURM_JOB_ID"], "utc": utc(), "queries": 475,
                  "schema": SCHEMA, "config_sha256": sha(run / "config.json"),
                  "artifact_sha256": {name: sha(run / name) for name in ("summary.json", "asr-full-e2e.csv", "analysis-config.json")},
                  "new_model_calls": 0, "new_encoder_calls": 0, "new_api_calls": 0}
    write(run / "completion.json", completion)
    print(json.dumps({"status": "complete", "queries": 475, "cohorts": 6,
                      "summary_sha256": completion["artifact_sha256"]["summary.json"]}), flush=True)


def flattened(value, prefix=""):
    result = {}
    for name, item in value.items():
        key = prefix + name
        if isinstance(item, dict):
            result.update(flattened(item, key + "__"))
        else:
            require(item is None or isinstance(item, (str, int, float, bool)), "Aggregate CSV contains a non-scalar value")
            result[key] = item
    return result


def write_csv(data, path):
    rows = [{"cohort": name, **flattened(data["cohorts"][name])} for name in COHORTS]
    fields = list(rows[0])
    require(all(set(row) == set(fields) for row in rows), "Inconsistent aggregate CSV schema")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def number(value, digits=6, scale=1, scientific=False):
    if value is None:
        return "undefined"
    require(finite(value), "Nonfinite public aggregate")
    return format(value * scale, "." + str(digits) + ("e" if scientific else "f"))


def percent(value):
    return number(value, digits=3, scale=100) + "%"


def report_lines(data, audit, csv_name):
    all_values = data["cohorts"]["all"]
    agreement = all_values["agreement"]
    metadata = data["metadata"]
    lines = ["# Full 475-query ASR kernel numerical and output comparison", "",
             "This completed experiment compares **all custom CUDA tasks** with the same GLM-ASR-Nano model's "
             "**all-PyTorch reference** on all **475** supplied query recordings. Every planned query is retained; "
             "the earlier single-clip fixture is separate and is not a 476th query. "
             "No external ASR/API or RAG-encoder calls are made.", "", "## Observed results", "",
             "- Full independent-greedy token-ID sequences agree for **" + str(agreement["token_exact_queries"])
             + "/475 (" + percent(agreement["token_exact_fraction"]) + ")** queries, including EOS and length differences.",
             "- Decoded transcripts agree for **" + str(agreement["text_stripped_exact_queries"])
             + "/475 (" + percent(agreement["text_stripped_exact_fraction"]) + ")** queries under the frozen decoder-output definition.",
             "- Complete PyTorch-reference-history numerical coverage is **" + str(agreement["reference_history_coverage_steps"])
             + "/" + str(agreement["reference_steps"]) + " positions ("
             + percent(agreement["reference_history_coverage_fraction"]) + ")**; independent greedy shared-history coverage is "
             + percent(agreement["greedy_reference_history_coverage"]) + "."]
    for field, label in (("logits", "logit"), ("probs", "probability")):
        value = all_values["numerical"][field]
        lines.append("- The empirical maximum absolute " + label + " discrepancy is **"
                     + number(value["max_abs"], scientific=True) + "** over " + str(value["elements"])
                     + " full-vocabulary values.")
    lines += ["", "These are exact summaries of the observed finite set, not confidence bounds or a population error guarantee. "
              "No bootstrap intervals, post-hoc numerical acceptance threshold, WER/CER estimate or downstream retrieval claim is introduced.", "",
              "## Numerical discrepancy on the full reference history", "",
              "For every query, the PyTorch greedy sequence defines the text-token history. CUDA retains its own "
              "audio projection. Scores are reused while independent histories match; after a divergence, CUDA is "
              "replayed on the remaining reference text-token history. A candidate EOS does not terminate this "
              "fixed-reference-history scoring early. This covers the full reference trajectory without treating "
              "later scores from two different greedy histories as same-history numerical error.", "",
              "All **59,264** vocabulary entries at every reference prediction position are included for logits and "
              "probabilities. Statistics use float64 calculations on saved float32 values. For candidate `x` and "
              "reference `y`, absolute discrepancy is `abs(x-y)`, RMS is `sqrt(mean((x-y)**2))`, and relative L2 is "
              "`norm(x-y)/norm(y)`. A zero reference norm yields zero relative L2 only when the difference norm is "
              "also zero; otherwise relative L2 is undefined.", "",
              "The empirical maximum is the maximum of the query maxima. Other query-max summaries and macro "
              "MAE/RMS/relative-L2 give each included query equal weight. Macro relative L2 averages only "
              "queries for which it is defined; their count is reported below. The p95 uses NumPy's `method='linear'`.", "",
              "| Output | Empirical max | Mean query max | Median query max | p95 query max | Macro MAE | Macro RMS | Macro relative L2 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for field in NUMERICAL:
        item = all_values["numerical"][field]
        lines.append("| " + field + " | " + " | ".join(number(item[key], scientific=True) for key in
            ("max_abs", "mean_query_max_abs", "median_query_max_abs", "p95_query_max_abs", "macro_mean_abs",
             "macro_rms", "macro_relative_l2")) + " |")
    lines += ["", "Pooled MAE is `sum(abs(diff))/sum(elements)`; pooled RMS is "
              "`sqrt(sum(diff**2)/sum(elements))`; pooled relative L2 is "
              "`sqrt(sum(diff**2)/sum(reference**2))`. These ratios are not averages of per-query ratios.", "",
              "| Output | Values | Pooled MAE | Pooled RMS | Pooled relative L2 | Defined query relative L2 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for field in NUMERICAL:
        item = all_values["numerical"][field]
        lines.append("| " + field + " | " + str(item["elements"]) + " | "
                     + " | ".join(number(item[key], scientific=True) for key in
                                  ("pooled_mean_abs", "pooled_rms", "pooled_relative_l2")) + " | "
                     + str(item["relative_l2_defined_queries"]) + "/475 |")
    lines += ["", "## Output agreement and coverage by cohort", "",
              "Exact token sequence agreement uses complete token IDs, including EOS. Positional token identity "
              "counts equal tokens at the same index and divides by the sum of the larger sequence lengths; "
              "unmatched length positions count as disagreements. It is not edit distance: an insertion can "
              "shift many subsequent positions. Reference-history argmax identity instead compares CUDA's argmax "
              "with the reference token at each fixed-reference-history step.", "",
              "| Cohort | Queries | Exact tokens | Exact decoded text | Positional token identity | Reference-history argmax identity | Greedy shared-history coverage | Full-reference coverage |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in COHORTS:
        item = data["cohorts"][name]
        value = item["agreement"]
        lines.append("| " + name + " | " + str(item["queries"]) + " | "
                     + " | ".join(percent(value[key]) for key in ("token_exact_fraction", "text_stripped_exact_fraction",
                         "positional_token_identity", "reference_history_argmax_identity", "greedy_reference_history_coverage",
                         "reference_history_coverage_fraction")) + " |")
    lines += ["", "Decoded text is `processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()`. "
              "The separate unstripped decoded-string exact-agreement count is "
              + str(agreement["text_raw_exact_queries"]) + "/475. No acronym, case, punctuation or other "
              "transcript-format normalization is applied in the edit-distance comparison.", "",
              "## Token and transcript edit distances", "",
              "Unit-cost Levenshtein insertion, deletion and substitution distances use complete token-ID "
              "sequences (including EOS), Unicode code points of the decoded output, or `text.split()` word "
              "sequences. Word splitting retains case and punctuation but ignores pure whitespace differences. "
              "The mature [RapidFuzz Levenshtein API](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html) "
              "is pinned to [version 3.14.6](https://pypi.org/project/RapidFuzz/3.14.6/), with unit weights and no processor.", "",
              "For each query, length-normalized edit distance is `distance/max(lengths)`; both empty gives zero. "
              "Macro NED averages these per-query ratios. Pooled NED is `sum(edits)/sum(max(lengths))`, a distinct "
              "length-weighted aggregate; its all-empty convention is zero. Here 'normalized' refers only to "
              "length scaling, not text normalization. Empty transcripts are retained.", "",
              "| Cohort | Unit | Mean edits | Macro NED (%) | Pooled NED (%) | Zero-distance queries |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name in COHORTS:
        for unit in EDIT_UNITS:
            item = data["cohorts"][name]["edit_distance"][unit]
            lines.append("| " + name + " | " + unit + " | " + number(item["mean_edits"])
                         + " | " + number(item["macro_normalized_distance"], scale=100)
                         + " | " + number(item["pooled_normalized_distance"], scale=100)
                         + " | " + str(item["zero_queries"]) + "/" + str(item["queries"]) + " |")
    lines += ["", "These compare two implementations of one model without independent reference transcripts. "
              "They measure implementation/output disagreement, not WER/CER, recognition accuracy or semantic "
              "correctness. Exact agreement can preserve a common recognition error.", "",
              "## Termination, replay and finite-set scope", "",
              "The frozen generation cap is 100 new tokens. EOS termination and reaching the cap are recorded "
              "separately; reaching the cap on an EOS token can occur in both categories.", "",
              "| Cohort | Reference EOS | CUDA EOS | Reference at cap | CUDA at cap | Reference cap without EOS | CUDA cap without EOS | Reference empty text | CUDA empty text |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in COHORTS:
        item = data["cohorts"][name]["agreement"]
        keys = ("reference_eos_queries", "cuda_eos_queries", "reference_at_cap_queries", "cuda_at_cap_queries",
                "reference_cap_without_eos_queries", "cuda_cap_without_eos_queries", "reference_empty_text_queries",
                "cuda_empty_text_queries")
        lines.append("| " + name + " | " + " | ".join(str(item[key]) for key in keys) + " |")
    controls = metadata["replay_controls"]
    lines += ["", "Reference-history replay is used for " + str(agreement["replay_queries"])
              + " queries, with " + str(agreement["replay_tail_rows"]) + " stored replay rows and "
              + str(agreement["replay_next_token_calls"]) + " replay next-token calls. These counts include "
              "the four predeclared first-query shard controls; they are not simply the number of divergent queries. "
              "The controls compare full replay with the natural CUDA common-prefix scores: maximum measured "
              "logit/probability discrepancies are " + number(controls["logits_max_abs"], scientific=True) + "/"
              + number(controls["probs_max_abs"], scientific=True) + ".", "",
              "All 475 supplied queries have been examined in earlier work. These are descriptive finite-set "
              "results, with no sampling or skipped-query extrapolation. The six cohorts overlap: all = practice "
              "350 + dev 125; dev = pilot 8 + remaining dev 117; remaining all 467 excludes the pilot. They are "
              "not independent replications or a new confirmatory holdout.", "",
              "## Precision, audit and provenance", "",
              "Global, CUDA-matmul and cuDNN FP32 controls are `ieee`, under the pinned model/kernel/source "
              "configuration. These controls do not guarantee that different implementations yield identical "
              "floating-point values. The [PyTorch 2.10 numerical-accuracy notes](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html) "
              "describe operation-order and platform effects. This instrumented experiment supplies no new "
              "latency, peak-memory or speed comparison and does not replace the historical timing ablation.", "",
              "- Summary Slurm job: `" + str(metadata["slurm_job_id"]) + "`.",
              "- GPU evidence jobs: " + ", ".join("`" + str(value) + "`" for value in metadata["gpu_slurm_job_ids"]) + ".",
              "- Independent saved-array audit job: `" + str(audit["slurm_job_id"]) + "`; status **pass**.",
              "- The audit binds all 475 query records and trace files and all six aggregate cohorts. It recomputes "
              "metrics and distances without rerunning the model, encoder or external APIs.",
              "- Summarizer versions: NumPy `" + metadata["versions"]["numpy"] + "`, RapidFuzz `"
              + metadata["versions"]["rapidfuzz"] + "`.",
              "- Model snapshot: `" + metadata["model_snapshot_revision"] + "`.", "",
              "The completed query evidence contains " + str(metadata["completed_query_model_generate_calls"])
              + " local model-generation calls and " + str(metadata["completed_query_audio_encoder_calls"])
              + " audio-encoder calls, plus the separately counted reference-history replay calls. Recorded runtime "
              "job counters include " + str(metadata["recorded_runtime_model_generate_calls"])
              + " model-generation and " + str(metadata["recorded_runtime_audio_encoder_calls"])
              + " audio-encoder calls. Interrupted jobs can have unpersisted attempts, so runtime counters "
              "are not a claim to reconstruct every attempted call. Summarization and auditing make no model, "
              "encoder or external API calls.", "",
              "No numerical acceptance tolerance was fitted after observing errors. The model is the reference "
              "implementation, not an independent ground-truth recognizer. The original single-clip result and "
              "the four-ASR normalization/retrieval experiments remain separate evidence.", "",
              "| Bound input or source | SHA256 |", "| --- | --- |"]
    for name, value in sorted(metadata["input_sha256"].items()):
        lines.append("| `" + name + "` | `" + value + "` |")
    lines += ["", "| Frozen inference source | SHA256 |", "| --- | --- |"]
    for name, value in sorted(metadata["source_sha256"].items()):
        lines.append("| `" + name + "` | `" + value + "` |")
    lines += ["| Loaded CUDA library | `" + metadata["library_sha256"] + "` |",
              "| Aggregator source | `" + metadata["summarizer_source_sha256"] + "` |"]
    lines += ["", "[All cohort aggregates as CSV](" + csv_name + ") contain unrounded counts and metrics. "
              "This report, the aggregate summary and CSV contain no query IDs, transcripts or token sequences. "
              "Per-query records, traces, audio and model weights remain private. The "
              "[full-set protocol](asr-full-e2e-protocol.md) documents reproduction; the "
              "[single-clip result](asr-e2e-error-results.md) and "
              "[historical kernel note](task34-kernels-results.md) retain their original scope."]
    return lines


def validate_audit(run, summary):
    audit = read(run / "audit.json")
    require(audit.get("status") == "pass" and not audit.get("errors"), "A passing independent audit is required")
    for name in ("summary.json", "asr-full-e2e.csv", "completion.json", "config.json"):
        require(audit["input_sha256"][name] == sha(run / name), "Audit does not bind current " + name)
    require(audit["queries_checked"] == 475 and audit["cohorts_checked"] == 6
            and audit["trace_files_checked"] == 475, "Incomplete independent audit coverage")
    for field in ("audit_new_model_calls", "audit_new_encoder_calls", "audit_new_api_calls"):
        require(audit[field] == 0, "Audit unexpectedly performed inference/API work")
    require(summary["schema"] == SCHEMA and summary["status"] == "complete"
            and set(summary["cohorts"]) == set(COHORTS), "Incomplete summary schema")
    for cohort, count in COHORTS.items():
        require(summary["cohorts"][cohort]["queries"] == count, "Incorrect summary cohort count")
    return audit


def render(run, output):
    summary = read(run / "summary.json")
    audit = validate_audit(run, summary)
    require(output.suffix.lower() == ".md", "Report output must be a Markdown path")
    require(output.resolve() not in {Path(__file__).resolve()}, "Report would overwrite its source")
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = run / "asr-full-e2e.csv"
    if output.parent != run:
        destination = output.parent / csv_path.name
        destination.write_bytes(csv_path.read_bytes())
    else:
        destination = csv_path
    output.write_text("\n".join(report_lines(summary, audit, destination.name)) + "\n", encoding="utf-8")
    manifest = {"status": "rendered", "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "renderer_source_sha256": sha(Path(__file__)),
                "input_sha256": {name: sha(run / name) for name in ("summary.json", "audit.json", "completion.json", "asr-full-e2e.csv")},
                "output_sha256": {path.name: sha(path) for path in (output, destination)},
                "new_model_calls": 0, "new_encoder_calls": 0, "new_api_calls": 0}
    write(output.parent / "render-manifest.json", manifest)
    print(json.dumps(manifest, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Frozen repository root; required for aggregation")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Audited Markdown output; defaults to RUN/report.md")
    parser.add_argument("--report-only", action="store_true", help="Render already audited aggregates without regenerating them")
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"), "Run only inside Edinburgh Slurm")
    require(np.__version__ == "2.4.6" and rapidfuzz.__version__ == "3.14.6", "Unexpected pinned aggregation environment")
    os.umask(0o077)
    run = args.run.resolve()
    if args.report_only:
        render(run, args.output.resolve() if args.output else run / "report.md")
    else:
        require(args.root is not None, "--root is required for source/input validation during aggregation")
        aggregate_run(args.root.resolve(), run)


if __name__ == "__main__":
    main()
