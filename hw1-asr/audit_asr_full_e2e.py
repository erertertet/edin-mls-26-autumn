"""Independent streaming audit of full-set ASR numerical traces; CPU Slurm only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import wave

import numpy as np
from rapidfuzz.distance import Levenshtein_py

VOCAB = 59264
SNAPSHOT = "61ba4e0b3309b6656edea3e93e419f7bd5c61957"
COHORT_SIZES = {"all": 475, "practice": 350, "dev": 125, "pilot": 8, "remaining_dev": 117, "remaining_all": 467}
NUMERICAL = ("logits", "probs", "audio_embeddings")
EDIT_UNITS = ("tokens", "characters", "words")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def numeric(actual, expected, label, relative=1e-9):
    if expected is None:
        require(actual is None, "Undefined value was replaced: " + label)
        return
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual),
            "Invalid finite scalar: " + label)
    if expected == 0:
        require(actual == 0, "Expected exact zero: " + label)
    else:
        require(math.isclose(float(actual), float(expected), rel_tol=relative, abs_tol=1e-300),
                "Independent numerical recount mismatch: " + label)


def relative_l2(sum_sq, sum_reference_sq):
    if sum_reference_sq > 0:
        return math.sqrt(sum_sq / sum_reference_sq), True
    return (0.0, True) if sum_sq == 0 else (None, False)


def error_stats(segments, shape):
    """Accumulate all values in bounded NumPy blocks, with extended-precision sums."""
    absolute = np.longdouble(0)
    squared = np.longdouble(0)
    reference_squared = np.longdouble(0)
    elements, maximum = 0, 0.0
    for reference, candidate in segments:
        require(reference.shape == candidate.shape and reference.dtype == candidate.dtype == np.float32,
                "Comparison segment shape/dtype mismatch")
        for start in range(0, len(reference), 16):
            one, two = reference[start:start + 16].astype(np.float64), candidate[start:start + 16].astype(np.float64)
            require(np.isfinite(one).all() and np.isfinite(two).all(), "Nonfinite numerical segment")
            difference = two - one
            if difference.size:
                maximum = max(maximum, float(np.max(np.abs(difference))))
                absolute += np.sum(np.abs(difference), dtype=np.longdouble)
                squared += np.sum(difference * difference, dtype=np.longdouble)
                reference_squared += np.sum(one * one, dtype=np.longdouble)
                elements += difference.size
    require(elements == math.prod(shape) and elements > 0, "Numerical coverage mismatch")
    sa, ss, sr = float(absolute), float(squared), float(reference_squared)
    relative, defined = relative_l2(ss, sr)
    return {"elements": elements, "shape": list(shape), "max_abs": maximum, "sum_abs": sa, "sum_sq": ss,
            "sum_reference_sq": sr, "mean_abs": sa / elements, "rms": math.sqrt(ss / elements),
            "reference_l2": math.sqrt(sr), "relative_l2": relative, "relative_l2_defined": defined}


def check_stats(actual, expected, label, exact_fields=True):
    if exact_fields:
        require(set(actual) == set(expected), "Numerical statistic fields mismatch: " + label)
    for key, value in actual.items():
        require(key in expected, "Unknown numerical statistic: " + key)
        if key in ("elements", "shape", "relative_l2_defined"):
            require(value == expected[key], "Statistic coverage/definition mismatch: " + label)
        elif key == "max_abs":
            require(value == expected[key], "Maximum absolute discrepancy mismatch: " + label)
        else:
            numeric(value, expected[key], label + "/" + key)


def probability_diagnostics(segments):
    minimum, maximum, mass_min, mass_max, mass_error = 1.0, 0.0, float("inf"), 0.0, 0.0
    rows = 0
    for array in segments:
        if not len(array):
            continue
        require(array.ndim == 2 and array.shape[1] == VOCAB and array.dtype == np.float32
                and np.isfinite(array).all() and np.all((array >= 0) & (array <= 1)), "Invalid probability trace")
        mass = array.astype(np.float64).sum(axis=1)
        minimum, maximum = min(minimum, float(array.min())), max(maximum, float(array.max()))
        mass_min, mass_max = min(mass_min, float(mass.min())), max(mass_max, float(mass.max()))
        mass_error = max(mass_error, float(np.max(np.abs(mass - 1))))
        rows += len(array)
    require(rows > 0, "Missing probability rows")
    return {"minimum": minimum, "maximum": maximum, "row_sum_min": mass_min,
            "row_sum_max": mass_max, "row_sum_max_abs_error": mass_error}


def compare_tree(actual, expected, label):
    require(set(actual) == set(expected), "Aggregate/schema fields mismatch: " + label)
    for key, expected_value in expected.items():
        value = actual[key]
        if isinstance(expected_value, dict):
            compare_tree(value, expected_value, label + "/" + key)
        elif isinstance(expected_value, (bool, int, str)):
            require(value == expected_value, "Exact aggregate mismatch: " + label + "/" + key)
        else:
            numeric(value, expected_value, label + "/" + key)


def distance(left, right):
    edits = int(Levenshtein_py.distance(left, right, weights=(1, 1, 1), processor=None))
    maximum = max(len(left), len(right))
    require(0 <= edits <= maximum, "Invalid unit-cost edit distance")
    return {"distance": edits, "max_length": maximum, "normalized_distance": edits / maximum if maximum else 0.0}


def numerical_aggregate(records, field):
    values = [r["stats"][field] for r in records]
    n = len(values)
    elements = sum(v["elements"] for v in values)
    sa, ss, sr = (math.fsum(v[name] for v in values) for name in ("sum_abs", "sum_sq", "sum_reference_sq"))
    maxima = np.asarray([v["max_abs"] for v in values])
    relatives = [v["relative_l2"] for v in values if v["relative_l2_defined"]]
    pooled, defined = relative_l2(ss, sr)
    return {"queries": n, "elements": elements, "max_abs": float(maxima.max()),
            "mean_query_max_abs": math.fsum(maxima) / n, "median_query_max_abs": float(np.quantile(maxima, .5)),
            "p95_query_max_abs": float(np.quantile(maxima, .95)),
            "macro_mean_abs": math.fsum(v["mean_abs"] for v in values) / n,
            "macro_rms": math.fsum(v["rms"] for v in values) / n,
            "macro_relative_l2": math.fsum(relatives) / len(relatives) if relatives else None,
            "relative_l2_defined_queries": len(relatives), "relative_l2_undefined_queries": n - len(relatives),
            "pooled_mean_abs": sa / elements, "pooled_rms": math.sqrt(ss / elements),
            "pooled_relative_l2": pooled, "pooled_relative_l2_defined": defined,
            "sum_abs": sa, "sum_sq": ss, "sum_reference_sq": sr}


def edit_aggregate(records, unit):
    values = [r["edits"][unit] for r in records]
    changes = np.asarray([v["distance"] for v in values])
    total = int(changes.sum())
    length = sum(v["max_length"] for v in values)
    return {"queries": len(records), "sum_edits": total, "sum_max_length": length,
            "mean_edits": math.fsum(changes) / len(records), "median_edits": float(np.quantile(changes, .5)),
            "p95_edits": float(np.quantile(changes, .95)), "max_edits": int(changes.max()),
            "macro_normalized_distance": math.fsum(v["normalized_distance"] for v in values) / len(records),
            "pooled_normalized_distance": total / length if length else 0.0,
            "zero_queries": sum(v["distance"] == 0 for v in values), "both_empty_queries": sum(v["max_length"] == 0 for v in values)}


def agreement_aggregate(records):
    n = len(records)
    reference_steps = sum(r["reference"]["steps"] for r in records)
    cuda_steps = sum(r["cuda"]["steps"] for r in records)
    positions = sum(max(r["reference"]["steps"], r["cuda"]["steps"]) for r in records)
    matches = sum(sum(a == b for a, b in zip(r["reference"]["token_ids"], r["cuda"]["token_ids"])) for r in records)
    covered = sum(r["reference_history"]["coverage_steps"] for r in records)
    mismatches = sum(r["reference_history"]["argmax_token_mismatch_count"] for r in records)
    comparable = sum(r["greedy"]["comparable_prediction_steps"] for r in records)
    result = {"token_exact_queries": sum(r["greedy"]["token_sequence_equal"] for r in records),
              "text_raw_exact_queries": sum(r["greedy"]["text_raw_equal"] for r in records),
              "text_stripped_exact_queries": sum(r["greedy"]["text_stripped_equal"] for r in records),
              "positional_token_matches": matches, "max_length_token_positions": positions,
              "positional_token_identity": matches / positions,
              "reference_history_argmax_matches": covered - mismatches, "reference_history_prediction_steps": covered,
              "reference_history_argmax_identity": (covered - mismatches) / covered,
              "greedy_common_prefix_tokens": sum(r["greedy"]["common_prefix_tokens"] for r in records),
              "greedy_comparable_prediction_steps": comparable, "reference_steps": reference_steps, "cuda_steps": cuda_steps,
              "greedy_reference_history_coverage": comparable / reference_steps,
              "reference_history_coverage_steps": covered, "reference_history_coverage_fraction": covered / reference_steps,
              "reference_mean_steps": reference_steps / n, "cuda_mean_steps": cuda_steps / n,
              "reference_max_steps": max(r["reference"]["steps"] for r in records),
              "cuda_max_steps": max(r["cuda"]["steps"] for r in records),
              "replay_queries": sum(r["reference_history"]["tail_rows"] > 0 for r in records),
              "replay_tail_rows": sum(r["reference_history"]["tail_rows"] for r in records),
              "replay_next_token_calls": sum(r["reference_history"]["replay_next_token_calls"] for r in records)}
    for name in ("token", "text_raw", "text_stripped"):
        result[name + "_exact_fraction"] = result[name + "_exact_queries"] / n
    for mode in ("reference", "cuda"):
        for field, output in (("eos_reached", "eos_queries"), ("at_cap", "at_cap_queries"),
                              ("cap_without_eos", "cap_without_eos_queries")):
            result[mode + "_" + output] = sum(r[mode][field] for r in records)
        result[mode + "_empty_text_queries"] = sum(not r[mode]["text_stripped"] for r in records)
    return result


def flatten(value, prefix=""):
    result = {}
    for name, item in value.items():
        key = prefix + name
        if isinstance(item, dict):
            result.update(flatten(item, key + "__"))
        else:
            result[key] = item
    return result


def validate_provenance(args, config):
    run = args.run
    plan = config["plan"]
    require(config["version"] == plan["version"] == "asr-full-e2e-v1"
            and config["expected_queries"] == plan["expected_queries"] == 475
            and config["shards"] == plan["shards"] == 4 and plan["max_new_tokens"] == 100
            and plan["modes"] == ["none", "all"] and plan["vocab_size"] == VOCAB
            and plan["precision"] == {"global": "ieee", "matmul": "ieee", "cudnn": "ieee"}
            and plan["kernel_acceptance_threshold"] is None and plan["bootstrap_replicates"] == 0,
            "Frozen full-set estimand changed")
    device = plan["device"]
    require(set(device) == {"name_contains", "major", "minor", "multi_processor_count"}
            and isinstance(device["name_contains"], str) and device["name_contains"]
            and all(type(device[key]) is int and device[key] > 0 for key in ("major", "multi_processor_count"))
            and type(device["minor"]) is int and 0 <= device["minor"] <= 9
            and isinstance(plan["hostname"], str) and plan["hostname"].endswith(".inf.ed.ac.uk"),
            "Frozen hardware context is incomplete")
    library_sha = config["library"]["sha256"]
    require(library_sha == plan["library_sha256"] and len(library_sha) == 64
            and all(char in "0123456789abcdef" for char in library_sha), "Frozen library hash is malformed")
    for name, expected in config["source_sha256"].items():
        require(sha(args.root / name) == expected, "Frozen full-set producer/source changed")
    require(set(config["input_paths"]) == set(config["input_sha256"]), "Original evidence input-path binding missing")
    for name, path in config["input_paths"].items():
        require(sha(Path(path)) == config["input_sha256"][name], "Previously audited input changed")
    old = Path(config["input_paths"]["full/config.json"]).parent
    require(read(old / "audit.json")["status"] == "pass", "Original full-audio audit did not pass")
    selection = read(old / "selection.json")
    require(config["cohorts"] == read(old / "config.json")["cohorts"]
            and config["cohort_order"] == list(config["cohorts"])
            and {name: len(ids) for name, ids in config["cohorts"].items()} == COHORT_SIZES,
            "Original six historical cohorts changed")
    queries = config["queries"]
    require(len(queries) == len({q["query_id"] for q in queries}) == 475
            and [q["query_id"] for q in queries] == selection["query_ids"] == config["cohorts"]["all"]
            and all(q["shard"] == i % 4 for i, q in enumerate(queries)), "Full-query/shard partition mismatch")
    originals = {q["query_id"]: q for q in selection["audio"]}
    for query in queries:
        require({k: query[k] for k in originals[query["query_id"]]} == originals[query["query_id"]], "Original audio selection changed")
    require(plan["replay_control_ids"] == [queries[i]["query_id"] for i in range(4)], "Replay-control assignment changed")
    staging = config["staging"]
    proof_path = run / "staging-proof.json"
    require(proof_path.resolve() == Path(staging["proof_path"]).resolve()
            and sha(proof_path) == staging["proof_sha256"], "Staging proof binding changed")
    proof = read(proof_path)
    require(proof["status"] == "complete" and proof["hostname"] == plan["hostname"]
            and proof["slurm_job_id"] == config["prepared_slurm_job_id"]
            and proof["model_files_sha256"] == config["model"]["files_sha256"]
            and proof["audio_files"] == 475 and proof["audio_sha256"] == {q["audio_file"]: q["audio_sha256"] for q in queries}
            and proof["library_sha256"] == library_sha,
            "Staged model/audio/library proof mismatch")
    library = Path(config["library"]["path"])
    require(library.resolve() == (run / "libasr_kernels.so").resolve() and sha(library) == library_sha,
            "Frozen CUDA library bytes changed")
    build_path = run / "library-build.json"
    require(build_path.resolve() == Path(config["library"]["build_proof_path"]).resolve()
            and sha(build_path) == config["library"]["build_proof_sha256"], "Library build-proof binding changed")
    build = read(build_path)
    expected_build_sources = {name: expected for name, expected in config["source_sha256"].items()
                              if name == "hw1-asr/Makefile" or name.startswith("hw1-asr/csrc/")}
    require(expected_build_sources and "hw1-asr/Makefile" in expected_build_sources
            and build["status"] == "complete" and build["library_sha256"] == library_sha
            and build["source_sha256"] == expected_build_sources and build["hostname"] == plan["hostname"]
            and build["sm"] == device["major"] * 10 + device["minor"]
            and str(build["slurm_job_id"]) and isinstance(build["nvcc_version"], str) and build["nvcc_version"]
            and isinstance(build["compiler_version"], str) and build["compiler_version"], "Library build source/toolchain/architecture proof mismatch")
    model = config["model"]
    require(model["name"] == plan["model_name"] == "zai-org/GLM-ASR-Nano-2512"
            and model["snapshot_revision"] == plan["snapshot_revision"] == SNAPSHOT,
            "Pinned model revision changed")
    snapshot = Path(staging["snapshot_path"])
    staged = snapshot.is_dir()
    if not staged:
        snapshot = Path(model["source_snapshot_path"])
    require(snapshot.is_dir() and snapshot.name == SNAPSHOT, "Pinned model snapshot unavailable")
    for relative, expected in model["files_sha256"].items():
        path = snapshot / relative
        require(sha(path) == expected and path.stat().st_size == proof["model_file_sizes"][relative], "Model checkpoint bytes changed")
        if staged:
            require(path.stat().st_mode & 0o222 == 0, "Staged checkpoint is not read-only")
    model_config = read(snapshot / "config.json")["text_config"]
    require(model_config["vocab_size"] == VOCAB and model_config["hidden_size"] == 2048, "Unexpected decoder dimensions")
    eos = model_config["eos_token_id"]
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    require(eos_ids and all(type(value) is int and 0 <= value < VOCAB for value in eos_ids), "Invalid EOS IDs")
    staged_audio = Path(staging["data_dir"])
    for query in queries:
        path = staged_audio / query["audio_file"] if staged_audio.is_dir() else args.root / "rag/data/rag_corpus" / query["audio_file"]
        require(sha(path) == query["audio_sha256"], "Recorded audio bytes changed")
        with wave.open(str(path), "rb") as wav:
            require((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) == (16000, 1, 2, "NONE")
                    and abs(wav.getnframes() / wav.getframerate() - query["seconds"]) <= 1e-9, "Audio format/duration mismatch")
    numeric(config["audio_seconds"], math.fsum(q["seconds"] for q in queries), "full audio duration")
    return queries, eos_ids, {"checkpoint_files": len(model["files_sha256"]), "source_files": len(config["source_sha256"]),
                             "staged_checkpoint_available": staged, "library_sha256": library_sha,
                             "library_build_proof_sha256": sha(build_path), "library_build_sm": build["sm"]}


def verify_mode(trace, record, mode, eos_ids):
    prefix = "reference" if mode == "none" else "cuda"
    logits_key = "reference_logits" if mode == "none" else "cuda_greedy_logits"
    probs_key = "reference_probs" if mode == "none" else "cuda_greedy_probs"
    logits, probs = trace[logits_key], trace[probs_key]
    tokens, chosen = trace[prefix + "_token_ids"], trace[prefix + "_chosen_probabilities"]
    steps = len(tokens)
    require(record["mode"] == mode and record["steps"] == steps and 1 <= steps <= 100
            and tokens.dtype == np.int64 and tokens.shape == (steps,) and chosen.dtype == np.float64 and chosen.shape == (steps,)
            and logits.dtype == probs.dtype == np.float32 and logits.shape == probs.shape == (steps, VOCAB)
            and np.all((tokens >= 0) & (tokens < VOCAB)), "Greedy mode coverage/dtype mismatch")
    require(tokens.tolist() == record["token_ids"] and chosen.tolist() == record["chosen_probabilities"]
            and np.array_equal(np.argmax(probs, axis=1), tokens)
            and np.array_equal(probs[np.arange(steps), tokens].astype(np.float64), chosen), "Greedy argmax/chosen probability mismatch")
    last_eos = int(tokens[-1]) in eos_ids
    require(not any(int(value) in eos_ids for value in tokens[:-1]) and (last_eos or steps == 100)
            and record["eos_reached"] == last_eos and record["at_cap"] == (steps == 100)
            and record["cap_without_eos"] == (steps == 100 and not last_eos)
            and record["terminal_token_id"] == int(tokens[-1]), "Greedy EOS/cap stopping metadata mismatch")
    require(isinstance(record["text_raw"], str) and isinstance(record["text_stripped"], str)
            and record["text_raw"].strip() == record["text_stripped"], "Decoded text definition mismatch")
    return tokens.tolist()


def recount_query(query, record, trace, eos_ids, controls):
    require(all(np.isfinite(array).all() for array in trace.values()), "Nonfinite saved trace")
    left = verify_mode(trace, record["reference"], "none", eos_ids)
    right = verify_mode(trace, record["cuda"], "all", eos_ids)
    ref_audio, cuda_audio = trace["reference_audio_embeddings"], trace["cuda_audio_embeddings"]
    require(ref_audio.shape == cuda_audio.shape and ref_audio.ndim == 2 and ref_audio.shape[0] > 0
            and ref_audio.shape[1] == 2048 and ref_audio.dtype == cuda_audio.dtype == np.float32, "Audio projection trace mismatch")
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix += 1
    comparable = min(len(left), len(right), prefix + 1)
    greedy = {"token_sequence_equal": left == right, "token_mismatch_count": sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right)),
              "first_token_mismatch_index": None if left == right else prefix, "common_prefix_tokens": prefix,
              "comparable_prediction_steps": comparable, "reference_steps": len(left), "cuda_steps": len(right),
              "text_raw_equal": record["reference"]["text_raw"] == record["cuda"]["text_raw"],
              "text_stripped_equal": record["reference"]["text_stripped"] == record["cuda"]["text_stripped"]}
    require(record["greedy"] == greedy, "Greedy history/sequence agreement accounting mismatch")
    control = query["query_id"] in controls
    start = 0 if control else comparable if comparable < len(left) else len(left)
    tail_rows = len(left) - start
    history = record["reference_history"]
    required = {"reference_logits", "reference_probs", "cuda_greedy_logits", "cuda_greedy_probs",
                "reference_audio_embeddings", "cuda_audio_embeddings", "reference_token_ids", "cuda_token_ids",
                "reference_chosen_probabilities", "cuda_chosen_probabilities"}
    if tail_rows:
        required.update(("cuda_reference_tail_logits", "cuda_reference_tail_probs"))
    require(set(trace) == required, "Reference-history tail trace fields mismatch")
    reference_segments, candidate_segments = {}, {}
    for field in ("logits", "probs"):
        ref = trace["reference_" + field]
        natural = trace["cuda_greedy_" + field]
        segments = [(ref[:start], natural[:start])]
        if tail_rows:
            tail = trace["cuda_reference_tail_" + field]
            require(tail.dtype == np.float32 and tail.shape == (tail_rows, VOCAB), "Forced reference-history tail shape mismatch")
            segments.append((ref[start:], tail))
        reference_segments[field] = segments
        candidate_segments[field] = [candidate for _, candidate in segments]
    argmax_ids = np.concatenate([np.argmax(array, axis=1) for array in candidate_segments["probs"] if len(array)]).tolist()
    expected_history = {"tail_start": start, "tail_rows": tail_rows, "replay_next_token_calls": tail_rows,
                        "coverage_steps": len(left), "coverage_complete": True, "argmax_ids": argmax_ids,
                        "argmax_token_mismatch_count": sum(a != b for a, b in zip(argmax_ids, left)),
                        "argmax_all_equal": argmax_ids == left, "used_cuda_audio_embeddings": True}
    require(all(history[key] == value for key, value in expected_history.items())
            and len(argmax_ids) == len(left), "Full reference-history coverage/accounting mismatch")
    replay_control = None
    if control:
        replay_control = {"enabled": True, "compared_steps": comparable}
        for field in ("logits", "probs"):
            natural = trace["cuda_greedy_" + field][:comparable]
            forced = trace["cuda_reference_tail_" + field][:comparable]
            replay_control[field + "_max_abs"] = float(np.max(np.abs(natural.astype(np.float64) - forced.astype(np.float64))))
            replay_control[field + "_exact_equal"] = bool(np.array_equal(natural, forced))
        require(history["replay_control"] == replay_control, "Full replay-control diagnostic mismatch")
    else:
        require(history["replay_control"] is None, "Undeclared full replay control")
    expected_stats = {field: error_stats(reference_segments[field], (len(left), VOCAB)) for field in ("logits", "probs")}
    expected_stats["audio_embeddings"] = error_stats([(ref_audio, cuda_audio)], ref_audio.shape)
    require(set(record["stats"]) == set(NUMERICAL), "Missing numerical outputs")
    for field in NUMERICAL:
        check_stats(record["stats"][field], expected_stats[field], field)
    diagnostics = {"reference": probability_diagnostics([trace["reference_probs"]]),
                   "cuda_greedy": probability_diagnostics([trace["cuda_greedy_probs"]]),
                   "cuda_reference_history": probability_diagnostics(candidate_segments["probs"])}
    compare_tree(record["probability_diagnostics"], diagnostics, "probability diagnostics")
    require(record["calls"] == {"model_generate_calls": 2, "audio_encoder_calls": 2, "replay_next_token_calls": tail_rows},
            "Per-query model/replay call accounting mismatch")
    require(set(record["input_fingerprints"]) == {"features", "input_ids", "features_mask"}, "Shared prepared-input fingerprints missing")
    for value in record["input_fingerprints"].values():
        require(len(value["sha256"]) == 64 and value["shape"] and isinstance(value["dtype"], str), "Malformed shared-input fingerprint")
    edits = {"tokens": distance(left, right), "characters": distance(record["reference"]["text_stripped"], record["cuda"]["text_stripped"]),
             "words": distance(record["reference"]["text_stripped"].split(), record["cuda"]["text_stripped"].split())}
    logit_argmax_disagreements = {
        "reference": int(np.count_nonzero(np.argmax(trace["reference_logits"], axis=1) != np.argmax(trace["reference_probs"], axis=1))),
        "cuda_greedy": int(np.count_nonzero(np.argmax(trace["cuda_greedy_logits"], axis=1) != np.argmax(trace["cuda_greedy_probs"], axis=1))),
        "cuda_reference_history": sum(int(np.count_nonzero(np.argmax(logits, axis=1) != np.argmax(probs, axis=1)))
                                      for logits, probs in zip(candidate_segments["logits"], candidate_segments["probs"]) if len(logits))}
    return {"query_id": query["query_id"], "stats": expected_stats, "reference": record["reference"], "cuda": record["cuda"],
            "greedy": greedy, "reference_history": {**expected_history, "replay_control": replay_control}, "edits": edits,
            "probability_diagnostics": diagnostics, "logit_probability_argmax_disagreements": logit_argmax_disagreements}


def validate_jobs(run, config, config_sha, records, input_hashes):
    jobs = {}
    for path in sorted((run / "jobs").glob("*.json")):
        value = read(path)
        job_id = str(value["slurm_job_id"])
        require(path.stem == job_id and job_id not in jobs and value["status"] in ("complete", "fail", "running")
                and type(value["shard"]) is int and 0 <= value["shard"] < 4, "Malformed runtime job identity/status")
        require(value["config_sha256"] == config_sha and value["source_sha256"] == config["source_sha256"]
                and value["library_sha256"] == config["library"]["sha256"]
                and value["staging_proof_sha256"] == config["staging"]["proof_sha256"], "Runtime source/library binding mismatch")
        require(value["precision"]["after"] == config["plan"]["precision"]
                and set(value["precision"]["before"]) == {"global", "matmul", "cudnn"}
                and all(isinstance(item, str) for item in value["precision"]["before"].values()), "Runtime precision proof mismatch")
        device = value["device"]
        planned_device = config["plan"]["device"]
        require(planned_device["name_contains"] in device["name"]
                and all(device[key] == planned_device[key] for key in ("major", "minor", "multi_processor_count"))
                and type(device["total_memory"]) is int and device["total_memory"] > 0, "Unexpected GPU execution context")
        require({name: item["version"] for name, item in value["versions"].items()} == config["versions"]
                and all(isinstance(item["path"], str) and item["path"] for item in value["versions"].values()), "Runtime dependency version/path mismatch")
        for field in ("new_api_calls", "new_external_asr_calls", "new_rag_encoder_calls"):
            require(value[field] == 0, "Unexpected external service/RAG encoder call")
        fields = ("model_loads", "model_generate_calls", "audio_encoder_calls", "processor_requests", "replay_next_token_calls",
                  "resumed_queries", "completed_queries")
        require(all(type(value[field]) is int and value[field] >= 0 for field in fields)
                and value["resumed_queries"] <= value["completed_queries"] <= len(config["queries"][value["shard"]::4]),
                "Invalid persisted runtime counters")
        if value["status"] == "complete":
            newly_completed = value["completed_queries"] - value["resumed_queries"]
            require(value["model_generate_calls"] == value["audio_encoder_calls"] == 2 * newly_completed
                    and value["processor_requests"] == newly_completed and value["model_loads"] == int(newly_completed > 0),
                    "Completed-job model/encoder/processor counters mismatch")
        jobs[job_id] = value
        input_hashes["jobs/" + path.name] = sha(path)
    require(jobs, "No runtime job proof")
    produced = {job_id: [] for job_id in jobs}
    for record in records:
        job_id = str(record["slurm_job_id"])
        require(job_id in jobs and jobs[job_id]["shard"] == record["shard"], "Missing producing-job/shard proof")
        produced[job_id].append(record)
    for job_id, value in jobs.items():
        if value["status"] == "complete":
            require(len(produced[job_id]) == value["completed_queries"] - value["resumed_queries"]
                    and value["replay_next_token_calls"] == sum(record["calls"]["replay_next_token_calls"] for record in produced[job_id]),
                    "Completed-job durable query/replay accounting mismatch")
    require({path.name for path in (run / "shards").glob("*.json")} == {str(i) + ".json" for i in range(4)},
            "Extra or missing shard proof")
    for shard in range(4):
        path = run / "shards" / (str(shard) + ".json")
        value = read(path)
        expected_ids = [query["query_id"] for query in config["queries"][shard::4]]
        job_id = str(value["slurm_job_id"])
        require(value["status"] == "complete" and value["shard"] == shard and value["config_sha256"] == config_sha
                and value["query_ids"] == expected_ids and value["planned_queries"] == value["completed_queries"] == len(expected_ids)
                and job_id in jobs and jobs[job_id]["status"] == "complete" and jobs[job_id]["shard"] == shard
                and jobs[job_id]["completed_queries"] == len(expected_ids)
                and value["resumed_queries"] == jobs[job_id]["resumed_queries"], "Incomplete final shard or resume proof")
        input_hashes["shards/" + path.name] = sha(path)
    return jobs


def verify_csv(path, cohorts):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    require(len(rows) == len(cohorts) == 6 and len(fields) == len(set(fields)), "Aggregate CSV row/column coverage mismatch")
    require([row["cohort"] for row in rows] == list(COHORT_SIZES), "Aggregate CSV cohort order mismatch")
    for row in rows:
        expected = {"cohort": row["cohort"], **flatten(cohorts[row["cohort"]])}
        require(set(row) == set(expected), "Aggregate CSV fields mismatch")
        for name, value in expected.items():
            if value is None:
                require(row[name] == "", "Undefined CSV value replaced")
            elif isinstance(value, (bool, str, int)):
                require(row[name] == str(value), "Exact CSV aggregate value mismatch")
            else:
                numeric(float(row[name]), value, "CSV/" + name)
    return len(rows), len(fields)


def audit(args):
    config = read(args.run / "config.json")
    config_sha = sha(args.run / "config.json")
    versions = {"numpy": np.__version__, "rapidfuzz": importlib.metadata.version("rapidfuzz")}
    require(versions == config["analysis_versions"], "Frozen CPU audit dependency versions changed")
    queries, eos_ids, provenance = validate_provenance(args, config)
    controls = set(config["plan"]["replay_control_ids"])
    input_hashes = {"config.json": config_sha, "staging-proof.json": sha(args.run / "staging-proof.json"),
                    "library-build.json": sha(args.run / "library-build.json")}
    record_hashes, trace_hashes = {}, {}
    records, recounted = [], []
    for index, query in enumerate(queries):
        stem = hashlib.sha256(query["query_id"].encode()).hexdigest()[:20]
        record_path = args.run / "queries" / (stem + ".json")
        trace_path = args.run / "traces" / (stem + ".npz")
        record = read(record_path)
        require(record["status"] == "complete" and record["config_sha256"] == config_sha
                and all(record[name] == value for name, value in query.items()), "Query identity/input/config binding mismatch")
        require(record["trace_file"] == "traces/" + stem + ".npz"
                and sha(trace_path) == record["trace_sha256"], "Private full-vocabulary trace hash mismatch")
        with np.load(trace_path, allow_pickle=False) as saved:
            # Stream one query at a time. No arrays from earlier queries are retained.
            trace = {name: saved[name] for name in saved.files}
            require(record["trace_shapes"] == {name: list(value.shape) for name, value in trace.items()}, "Saved trace shape manifest mismatch")
            recounted.append(recount_query(query, record, trace, eos_ids, controls))
            del trace
        records.append(record)
        record_hashes[record_path.name] = sha(record_path)
        trace_hashes[trace_path.name] = record["trace_sha256"]
        if (index + 1) % 25 == 0 or index == len(queries) - 1:
            print(json.dumps({"stage": "independent_saved_array_audit", "queries_checked": index + 1, "expected_queries": 475}), flush=True)
    require({path.name for path in (args.run / "queries").glob("*.json")} == set(record_hashes)
            and {path.name for path in (args.run / "traces").glob("*.npz")} == set(trace_hashes), "Extra or missing completed query/trace files")
    jobs = validate_jobs(args.run, config, config_sha, records, input_hashes)
    summary = read(args.run / "summary.json")
    metadata = summary["metadata"]
    require(summary["status"] == "complete" and summary["schema"] == "asr-full-e2e-summary-v1", "Aggregate summary incomplete")
    analysis = read(args.run / "analysis-config.json")
    summarizer = args.root / "hw1-asr/summarize_asr_full_e2e.py"
    require(analysis == {"schema": summary["schema"], "inference_config_sha256": config_sha,
                        "summarizer_source_sha256": sha(summarizer), "versions": versions, "cohort_counts": COHORT_SIZES,
                        "percentile_method": "linear", "edit_units": list(EDIT_UNITS), "edit_weights": [1, 1, 1],
                        "edit_processor": None, "text_definition": config["plan"]["text_definition"], "numerical_fields": list(NUMERICAL),
                        "bootstrap_replicates": 0, "kernel_acceptance_threshold": None}, "Frozen independent aggregation specification/source changed")
    input_hashes["analysis-config.json"] = sha(args.run / "analysis-config.json")
    require(metadata["input_sha256"] == input_hashes, "Aggregate runtime/input hash inventory mismatch")
    by_id = {record["query_id"]: record for record in recounted}
    cohorts = {}
    for name, ids in config["cohorts"].items():
        selected = [by_id[qid] for qid in ids]
        cohorts[name] = {"queries": len(selected), "numerical": {field: numerical_aggregate(selected, field) for field in NUMERICAL},
                         "agreement": agreement_aggregate(selected), "edit_distance": {unit: edit_aggregate(selected, unit) for unit in EDIT_UNITS}}
    compare_tree(summary["cohorts"], cohorts, "all six independent cohorts")
    controls_recounted = [record["reference_history"]["replay_control"] for record in recounted
                         if record["reference_history"]["replay_control"] is not None]
    require(len(controls_recounted) == 4, "Four full-history replay controls were not verified")
    device_json = {json.dumps({key: value["device"][key] for key in ("name", "major", "minor", "multi_processor_count", "total_memory")}, sort_keys=True)
                   for value in jobs.values()}
    expected_metadata = {"gpu_slurm_job_ids": sorted(jobs), "inference_versions": config["versions"], "versions": versions,
        "model_snapshot_revision": SNAPSHOT, "library_sha256": config["library"]["sha256"], "source_sha256": config["source_sha256"],
        "private_query_inventory_sha256": digest_json(record_hashes), "private_trace_inventory_sha256": digest_json(trace_hashes),
        "query_records_checked": 475, "trace_hashes_checked": 475, "audio_hashes_checked": 475,
        "model_file_hashes_checked": provenance["checkpoint_files"], "devices": [json.loads(value) for value in sorted(device_json)],
        "completed_query_model_generate_calls": 950, "completed_query_audio_encoder_calls": 950,
        "recorded_runtime_model_generate_calls": sum(value["model_generate_calls"] for value in jobs.values()),
        "recorded_runtime_audio_encoder_calls": sum(value["audio_encoder_calls"] for value in jobs.values()),
        "runtime_records_by_status": {status: sum(value["status"] == status for value in jobs.values()) for status in sorted({value["status"] for value in jobs.values()})},
        "new_external_asr_calls": 0, "new_api_calls": 0, "new_rag_encoder_calls": 0,
        "summarizer_new_model_calls": 0, "summarizer_new_encoder_calls": 0, "summarizer_new_api_calls": 0,
        "precision": config["plan"]["precision"], "max_new_tokens": 100, "bootstrap_replicates": 0,
        "kernel_acceptance_threshold": None, "summarizer_source_sha256": analysis["summarizer_source_sha256"],
        "replay_controls": {"queries": 4, "compared_steps": sum(value["compared_steps"] for value in controls_recounted),
            "logits_max_abs": max(value["logits_max_abs"] for value in controls_recounted),
            "probs_max_abs": max(value["probs_max_abs"] for value in controls_recounted),
            "logits_numerically_equal_queries": sum(value["logits_exact_equal"] for value in controls_recounted),
            "probs_numerically_equal_queries": sum(value["probs_exact_equal"] for value in controls_recounted)}}
    require(set(metadata) == set(expected_metadata) | {"utc", "slurm_job_id", "input_sha256"}
            and metadata["utc"] and metadata["slurm_job_id"], "Aggregate metadata schema/job binding mismatch")
    for name, value in expected_metadata.items():
        if isinstance(value, dict) and name in ("replay_controls", "runtime_records_by_status"):
            compare_tree(metadata[name], value, "metadata/" + name)
        else:
            require(metadata[name] == value, "Aggregate metadata recount mismatch: " + name)
    csv_rows, csv_columns = verify_csv(args.run / "asr-full-e2e.csv", cohorts)
    completion = read(args.run / "completion.json")
    require(completion["status"] == "complete" and completion["queries"] == 475 and completion["schema"] == summary["schema"]
            and completion["config_sha256"] == config_sha and str(completion["slurm_job_id"]) == str(metadata["slurm_job_id"])
            and completion["new_model_calls"] == completion["new_encoder_calls"] == completion["new_api_calls"] == 0,
            "Aggregate completion/job/zero-call proof mismatch")
    require(completion["artifact_sha256"] == {name: sha(args.run / name) for name in ("summary.json", "asr-full-e2e.csv", "analysis-config.json")},
            "Aggregate completion artifact binding mismatch")
    result_inputs = {name: sha(args.run / name) for name in ("config.json", "staging-proof.json", "library-build.json", "analysis-config.json", "summary.json", "asr-full-e2e.csv", "completion.json")}
    all_agreement = cohorts["all"]["agreement"]
    return {"status": "pass", "errors": [], "input_sha256": result_inputs,
        "private_query_inventory_sha256": digest_json(record_hashes), "private_trace_inventory_sha256": digest_json(trace_hashes),
        "source_sha256": {"auditor": sha(Path(__file__)), "summarizer": sha(summarizer)},
        "inference_source_sha256": config["source_sha256"], "original_input_sha256": config["input_sha256"],
        "queries_checked": 475, "cohorts_checked": 6, "trace_files_checked": 475, "audio_files_checked": 475,
        "modes_per_query": 2, "numerical_fields_checked": list(NUMERICAL), "edit_distance_comparisons_checked": 1425,
        "csv_rows_checked": csv_rows, "csv_columns_checked": csv_columns, "replay_controls_checked": 4,
        "reference_prediction_steps_checked": all_agreement["reference_steps"],
        "full_vocabulary_values_checked_per_primary_field": all_agreement["reference_steps"] * VOCAB,
        "reference_history_coverage_fraction": all_agreement["reference_history_coverage_fraction"],
        "greedy_reference_history_coverage": all_agreement["greedy_reference_history_coverage"],
        "replay_tail_rows_checked": all_agreement["replay_tail_rows"],
        "numeric_recount": {field: cohorts["all"]["numerical"][field] for field in NUMERICAL},
        "agreement_recount": all_agreement, "edit_distance_recount": cohorts["all"]["edit_distance"],
        "probability_mass_max_abs_error": {mode: max(record["probability_diagnostics"][mode]["row_sum_max_abs_error"] for record in recounted)
                                           for mode in ("reference", "cuda_greedy", "cuda_reference_history")},
        "logit_probability_argmax_disagreements": {mode: sum(record["logit_probability_argmax_disagreements"][mode] for record in recounted)
                                                   for mode in ("reference", "cuda_greedy", "cuda_reference_history")},
        "gpu_runtime_records_checked": len(jobs), "runtime_records_by_status": expected_metadata["runtime_records_by_status"],
        "provenance": provenance, "versions": versions,
        "import_paths": {"numpy": np.__file__, "levenshtein_python": Levenshtein_py.__file__},
        "scalar_recount_relative_tolerance": 1e-9, "max_abs_and_discrete_counts_exact": True,
        "kernel_acceptance_threshold": None, "audit_new_model_calls": 0, "audit_new_encoder_calls": 0, "audit_new_api_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"), "Audit only in an Edinburgh Slurm CPU allocation")
    os.umask(0o077)
    args.root, args.run = args.root.resolve(), args.run.resolve()
    base = {"slurm_job_id": os.environ["SLURM_JOB_ID"], "utc": datetime.now(timezone.utc).isoformat(),
            "schema": "asr-full-e2e-audit-v1"}
    failure = None
    try:
        result = {**base, **audit(args)}
    except Exception as error:
        result = {**base, "status": "fail", "errors": [{"type": type(error).__name__, "message": str(error)}],
                  "audit_new_model_calls": 0, "audit_new_encoder_calls": 0, "audit_new_api_calls": 0}
        failure = error
    path = args.run / "audit.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps({"status": result["status"], "queries_checked": result.get("queries_checked", 0),
                      "audit_sha256": sha(path)}), flush=True)
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
