"""Audit saved transcript disagreements with RapidFuzz's Python implementation in Slurm."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import socket
import sys

import numpy as np
from rapidfuzz.distance import Levenshtein_py

ARMS = ("glm", "whisper", "mai", "voxtral")
CONDITIONS = ("raw", "normalized")
UNITS = ("characters", "words")
METRICS = ("absolute", "normalized_distance")
NORMALIZER_SHA256 = "fff6993b2f1ae85db3a65a6c645a6e2ac3ad0ec415eb82af3f73a763de8e9dd2"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(actual, expected, message, tolerance=1e-12):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isclose(actual, float(expected), rel_tol=0, abs_tol=tolerance), message)


def transcript_path(full_run, qid, arm):
    return full_run / "transcripts" / arm / (text_sha(qid)[:20] + ".json")


def paired_series(edits, normalized):
    # Input axes query,pair,condition,unit. Output query,pair-label,unit,metric,condition.
    base = np.transpose(np.stack((edits.astype(np.float64), normalized), axis=-1), (0, 1, 3, 4, 2))
    overall = base.mean(axis=1, keepdims=True)
    return np.concatenate((overall, base), axis=1)


def exact_means(values):
    shape = values.shape[1:]
    columns = values.reshape(len(values), -1)
    return np.asarray([math.fsum(columns[:, i]) / len(values) for i in range(columns.shape[1])]).reshape(shape)


def audit(args):
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"), "Run only inside Edinburgh Slurm")
    require(importlib.metadata.version("RapidFuzz") == "3.14.6", "Unexpected RapidFuzz version")
    require(Levenshtein_py.distance.__module__ == "rapidfuzz.distance.Levenshtein_py", "Audit is not using Python implementation")
    require(Levenshtein_py.distance("kitten", "sitting", weights=(1, 1, 1), processor=None) == 3
            and Levenshtein_py.distance(("one", "two"), ("one", "three"), weights=(1, 1, 1), processor=None) == 1
            and Levenshtein_py.distance("", "", weights=(1, 1, 1), processor=None) == 0,
            "Python Levenshtein actual-API check failed")
    run, full = args.run, args.full_run
    config, summary, completion = (read(run / name) for name in ("analysis-config.json", "summary.json", "completion.json"))
    original_config, original_audit, selection = (read(full / name) for name in ("config.json", "audit.json", "selection.json"))
    require(original_audit["status"] == "pass" and not original_audit["errors"], "Original experiment audit did not pass")
    require(config["conditions"] == list(CONDITIONS) and config["units"] == list(UNITS)
            and config["metrics"] == list(METRICS) and config["change_tolerance"] == 1e-12
            and config["weights"] == [1, 1, 1] and config["processor"] is None,
            "Disagreement metric configuration changed")
    pairs = [tuple(pair) for pair in config["pairs"]]
    require(pairs == list(itertools.combinations(ARMS, 2)), "Unexpected six unordered model pairs")
    pair_labels = ["overall"] + ["__".join(pair) for pair in pairs]
    require(config["pair_labels"] == pair_labels, "Pair label order mismatch")
    require(config["cohorts"] == original_config["cohorts"]
            and config["cohort_order"] == list(original_config["cohorts"]), "Cohort membership/order changed")
    cohorts, cohort_order = config["cohorts"], config["cohort_order"]
    require(len(cohort_order) == 6 and {name: len(ids) for name, ids in cohorts.items()}
            == {"all": 475, "practice": 350, "dev": 125, "pilot": 8, "remaining_dev": 117, "remaining_all": 467},
            "Unexpected historical cohort sizes")
    boot = config["bootstrap"]
    require(boot["replicates"] == 5000 and boot["seed"] == 20261117 and boot["method"] == "percentile"
            and boot["confidence"] == .95 and boot["batch"] == 50
            and boot["seeds"] == {cohort: boot["seed"] + i for i, cohort in enumerate(cohort_order)},
            "Bootstrap configuration changed")
    normalizer_path = args.root / "rag/extensions/transcript_normalization.py"
    require(sha(normalizer_path) == original_config["normalizer_sha256"] == NORMALIZER_SHA256, "Frozen normalizer changed")
    sys.path.insert(0, str(args.root / "rag/extensions"))
    from transcript_normalization import normalize_transcript
    for relative, expected in config["input_sha256"].items():
        path = (full / relative).resolve()
        path.relative_to(full.resolve())
        require(sha(path) == expected, "Frozen full-experiment input changed")
    for relative, expected in original_audit["artifact_sha256"].items():
        require(config["input_sha256"].get(relative) == expected and sha(full / relative) == expected,
                "Original audited artifacts missing or changed")
    require(config["input_sha256"].get("audit.json") == sha(full / "audit.json"), "Original audit binding absent")
    expected_original_sources = {**original_config["input_sources"], **original_config["sources"]}
    require(config["original_source_sha256"] == expected_original_sources, "Original source declarations changed")
    for relative, expected in expected_original_sources.items():
        require(sha(args.root / relative) == expected, "Original source/input changed")
    source_paths = {"analyze_asr_edit_distance.py": args.root / "rag/extensions/analyze_asr_edit_distance.py",
                    "transcript_normalization.py": normalizer_path,
                    "protocol": args.root / "report/asr-edit-distance-protocol.md",
                    "requirements": args.root / "rag/extensions/asr-edit-distance-requirements.txt"}
    require(set(config["source_sha256"]) == set(source_paths), "Unexpected producer source declarations")
    for name, path in source_paths.items():
        require(sha(path) == config["source_sha256"][name], "Frozen producer source changed")
    for name, expected in completion["artifact_sha256"].items():
        path = (run / name).resolve()
        path.relative_to(run.resolve())
        require(sha(path) == expected, "Completed distance-analysis artifact changed")

    mappings_list = read(full / "transcript-mappings.json")
    mappings = {(item["query_id"], item["arm"]): item for item in mappings_list}
    qids = selection["query_ids"]
    audios = {item["query_id"]: item for item in selection["audio"]}
    require(len(qids) == len(set(qids)) == len(audios) == 475
            and len(mappings) == len(mappings_list) == 1900
            and set(mappings) == {(q, arm) for q in qids for arm in ARMS}, "Original mapping coverage mismatch")
    transcript_files = set()
    for index, ((qid, arm), mapping) in enumerate(mappings.items(), 1):
        source = transcript_path(full, qid, arm)
        relative = str(source.relative_to(full))
        transcript_files.add(relative)
        require(sha(source) == mapping["source_sha256"] == config["input_sha256"].get(relative), "Transcript source SHA mismatch")
        value = read(source)
        require(value["query_id"] == qid and value["model"] == mapping["model"] == original_config["models"][arm]
                and value["status"] == "complete" and value["transcript"] == mapping["raw"]
                and value["audio_sha256"] == mapping["audio_sha256"] == audios[qid]["audio_sha256"]
                and mapping["split"] == audios[qid]["split"], "Raw transcript/mapping provenance mismatch")
        require(isinstance(mapping["raw"], str) and isinstance(mapping["normalized"], str)
                and normalize_transcript(mapping["raw"]) == mapping["normalized"]
                and normalize_transcript(mapping["normalized"]) == mapping["normalized"], "Normalized mapping mismatch")
        for condition in CONDITIONS:
            require(text_sha(mapping[condition]) == mapping[condition + "_text_sha256"], "Mapped text SHA mismatch")
        if index % 400 == 0:
            print(json.dumps({"audit_stage": "transcript_provenance", "checked": index, "total": 1900}), flush=True)
    require({relative for relative in config["input_sha256"] if relative.startswith("transcripts/")} == transcript_files,
            "Unaccounted transcript files in input manifest")

    with np.load(run / "distances.npz", allow_pickle=False) as saved:
        distance_cache = {name: saved[name] for name in saved.files}
    require(set(distance_cache) == {"query_ids", "pairs", "conditions", "units", "edits", "lengths_left", "lengths_right", "normalized"},
            "Unexpected distance-cache fields")
    require(distance_cache["query_ids"].tolist() == qids and distance_cache["pairs"].tolist() == [list(pair) for pair in pairs]
            and distance_cache["conditions"].tolist() == list(CONDITIONS) and distance_cache["units"].tolist() == list(UNITS),
            "Distance-cache labels/order mismatch")
    for field in ("edits", "lengths_left", "lengths_right"):
        require(distance_cache[field].shape == (475, 6, 2, 2) and distance_cache[field].dtype == np.int64
                and np.all(distance_cache[field] >= 0), "Distance-cache integer schema invalid")
    require(distance_cache["normalized"].shape == (475, 6, 2, 2) and distance_cache["normalized"].dtype == np.float64
            and np.isfinite(distance_cache["normalized"]).all()
            and np.all((distance_cache["normalized"] >= 0) & (distance_cache["normalized"] <= 1)), "NED cache schema invalid")
    checks, maximum_ned_error = 0, 0.0
    for qi, qid in enumerate(qids):
        for pi, (left, right) in enumerate(pairs):
            for ci, condition in enumerate(CONDITIONS):
                left_text, right_text = mappings[qid, left][condition], mappings[qid, right][condition]
                for ui, unit in enumerate(UNITS):
                    one, two = (left_text, right_text) if unit == "characters" else (left_text.split(), right_text.split())
                    distance = Levenshtein_py.distance(one, two, weights=(1, 1, 1), processor=None)
                    maximum = max(len(one), len(two))
                    expected_ned = distance / maximum if maximum else 0.0
                    index = qi, pi, ci, ui
                    require(int(distance_cache["edits"][index]) == distance
                            and int(distance_cache["lengths_left"][index]) == len(one)
                            and int(distance_cache["lengths_right"][index]) == len(two), "Python/C++ distance or sequence length mismatch")
                    error = abs(float(distance_cache["normalized"][index]) - expected_ned)
                    require(error <= 1e-15, "Explicit max-length NED formula mismatch")
                    maximum_ned_error = max(maximum_ned_error, error)
                    checks += 1
        if (qi + 1) % 50 == 0 or qi + 1 == len(qids):
            print(json.dumps({"audit_stage": "python_levenshtein", "queries": qi + 1, "unit_comparisons": checks}), flush=True)
    require(checks == 11400, "Incomplete independent edit-distance coverage")
    series = paired_series(distance_cache["edits"], distance_cache["normalized"])
    require(series.shape == (475, 7, 2, 2, 2), "Query-wise pair/overall series shape mismatch")
    with np.load(run / "bootstrap-values.npz", allow_pickle=False) as saved:
        bootstrap_cache = {name: saved[name] for name in saved.files}
    expected_boot_fields = {"pair_labels", "units", "metrics", "conditions", "cohorts"}
    expected_boot_fields.update("values_" + cohort for cohort in cohort_order)
    expected_boot_fields.update("indices_" + cohort for cohort in cohort_order)
    require(set(bootstrap_cache) == expected_boot_fields
            and bootstrap_cache["pair_labels"].tolist() == pair_labels and bootstrap_cache["units"].tolist() == list(UNITS)
            and bootstrap_cache["metrics"].tolist() == list(METRICS) and bootstrap_cache["conditions"].tolist() == list(CONDITIONS)
            and bootstrap_cache["cohorts"].tolist() == cohort_order, "Bootstrap labels/coverage mismatch")
    require(summary["status"] == completion["status"] == "complete"
            and len(summary["rows"]) == 84 and set(summary["cohorts"]) == set(cohort_order), "Summary coverage mismatch")
    metadata = summary["metadata"]
    analysis_job_id = metadata.get("slurm_job_id", metadata.get("analysis_slurm_job_id"))
    require(analysis_job_id == completion["slurm_job_id"], "Summary/completion job IDs disagree")
    require(metadata["source_sha256"] == config["source_sha256"], "Public summary producer-source binding mismatch")
    require(metadata["normalizer_sha256"] == config["normalizer_sha256"] == NORMALIZER_SHA256
            and metadata["bootstrap"] == boot and metadata["versions"] == config["versions"]
            and all(config["versions"][name]["version"] == importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "rapidfuzz")), "Public metric/version/bootstrap declarations mismatch")
    probe = read(run / "api-probe.json")
    require(metadata["api_probe"] == probe and probe["status"] == "pass" and probe["examples_checked"] == 7
            and probe["weights"] == [1, 1, 1] and probe["processor"] is None and "_cpp" in probe["distance_backend"],
            "Producer compiled-backend API probe did not pass")
    for relative, expected in metadata["input_sha256"].items():
        require(config["input_sha256"].get(relative) == expected and sha(full / relative) == expected,
                "Public summary original-input binding mismatch")
    require(set(("config.json", "summary.json", "transcript-mappings.json", "audit.json")) <= set(metadata["input_sha256"]),
            "Public summary original audit/config/mapping binding incomplete")
    for holder in (metadata, completion, config):
        require(all(holder[field] == 0 for field in ("new_asr_calls", "new_encoder_calls", "new_api_calls")), "Unexpected model/API call declaration")
    require(completion["queries"] == 475 and completion["transcript_mappings"] == 1900 and completion["model_pairs"] == 6
            and completion["distances"] == 11400 and completion["summary_rows"] == 84 and completion["bootstrap_replicates"] == 5000,
            "Completion dimension declarations mismatch")
    row_map = {(row["cohort"], row["pair"], row["unit"]): row for row in summary["rows"]}
    require(len(row_map) == 84 and set(row_map) == {(cohort, pair, unit) for cohort in cohort_order for pair in pair_labels for unit in UNITS},
            "Summary row labels duplicated/missing")
    interval_checks, replicate_checks = 0, 0
    largest_bootstrap_error = 0.0
    identity_report, replica_counts = {}, {}
    for cohort in cohort_order:
        ids = cohorts[cohort]
        indices = np.asarray([qids.index(q) for q in ids], dtype=np.intp)
        per_query = series[indices]
        expected_means = exact_means(per_query)
        draws = bootstrap_cache["indices_" + cohort]
        distribution = bootstrap_cache["values_" + cohort]
        require(draws.shape == (5000, len(ids)) and draws.dtype.kind in "iu"
                and np.all((draws >= 0) & (draws < len(ids))), "Stored cohort query indices invalid")
        require(distribution.shape == (7, 2, 2, 2, 5000) and distribution.dtype == np.float64
                and np.isfinite(distribution).all(), "Stored cohort bootstrap distribution invalid")
        require(np.all(distribution[:, :, 0] >= 0)
                and np.all((distribution[:, :, 1] >= -1e-15) & (distribution[:, :, 1] <= 1 + 1e-15)), "Bootstrap metric ranges invalid")
        info = summary["cohorts"][cohort]
        identity = {condition + "_queries": sum(len({mappings[qid, arm][condition] for arm in ARMS}) == 1 for qid in ids)
                    for condition in CONDITIONS}
        require(info["queries"] == len(ids) and info["seed"] == boot["seeds"][cohort]
                and info["character_identity"] == identity, "Cohort sample/seed/character identity mismatch")
        require(identity["raw_queries"] == original_audit["cohorts"][cohort]["raw_identical_queries"]
                and identity["normalized_queries"] == original_audit["cohorts"][cohort]["normalized_identical_queries"],
                "Original exact-character identity counts did not reproduce")
        identity_report[cohort] = {"queries": len(ids), **identity}
        for pi, pair in enumerate(pair_labels):
            for ui, unit in enumerate(UNITS):
                row = row_map[cohort, pair, unit]
                require(row["queries"] == len(ids), "Summary row sample count mismatch")
                absolute_series = per_query[:, pi, ui, 0]
                zero = {condition + "_queries": int(np.count_nonzero(absolute_series[:, ci] == 0)) for ci, condition in enumerate(CONDITIONS)}
                require(row["zero_distance"] == zero, "Zero-distance query count mismatch")
                ned_delta = per_query[:, pi, ui, 1, 1] - per_query[:, pi, ui, 1, 0]
                expected_counts = {"decrease": int(np.count_nonzero(ned_delta < -1e-12)),
                                   "tie": int(np.count_nonzero(np.abs(ned_delta) <= 1e-12)),
                                   "increase": int(np.count_nonzero(ned_delta > 1e-12))}
                require(row["change_counts"] == expected_counts, "NED change-count pairing/tolerance mismatch")
                if pair == "overall" and unit == "characters":
                    require(zero == identity, "Overall character-zero identity mismatch")
                for mi, metric in enumerate(METRICS):
                    values = row[metric]
                    raw, normalized = (float(expected_means[pi, ui, mi, ci]) for ci in range(2))
                    close(values["raw_mean"], raw, "Raw metric mean mismatch")
                    close(values["normalized_mean"], normalized, "Normalized metric mean mismatch")
                    close(values["delta_mean"], normalized - raw, "Mean paired delta mismatch")
                    for label, samples in (("raw_ci", distribution[pi, ui, mi, 0]),
                                           ("normalized_ci", distribution[pi, ui, mi, 1]),
                                           ("delta_ci", distribution[pi, ui, mi, 1] - distribution[pi, ui, mi, 0])):
                        bounds = np.quantile(samples, [.025, .975])
                        require(set(values[label]) == {"low", "high"}, "Unexpected interval fields")
                        close(values[label]["low"], bounds[0], "Percentile interval lower bound mismatch")
                        close(values[label]["high"], bounds[1], "Percentile interval upper bound mismatch")
                        interval_checks += 1
        # Sample stored replicate numbers only; no new query bootstrap samples are generated.
        generator = np.random.default_rng(boot["seeds"][cohort] + 959)
        sampled = [0, 4999, *generator.choice(np.arange(1, 4999), 23, replace=False).tolist()]
        for replicate in sampled:
            rebuilt = exact_means(per_query[draws[replicate]])
            error = float(np.max(np.abs(rebuilt - distribution[..., replicate])))
            require(error <= 1e-12, "Saved query-common bootstrap replicate mismatch")
            largest_bootstrap_error = max(largest_bootstrap_error, error)
            replicate_checks += 1
        replica_counts[cohort] = len(sampled)
        print(json.dumps({"audit_stage": "cohort_summary_and_bootstrap", "cohort": cohort,
                          "summary_rows": 14, "stored_replicates_checked": len(sampled)}), flush=True)
    require(interval_checks == 504 and replicate_checks == 150, "Incomplete summary/CI/bootstrap audit coverage")
    return {"status": "pass", "errors": [], "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "analysis_slurm_job_id": analysis_job_id, "utc": datetime.now(timezone.utc).isoformat(),
            "queries": 475, "transcript_mappings_checked": 1900, "text_hashes_checked": 3800,
            "unordered_model_pairs": 6, "all_unit_comparisons_checked": checks,
            "summary_rows_checked": 84, "paired_percentile_intervals_checked": interval_checks,
            "bootstrap_replicates_per_cohort": 5000, "stored_bootstrap_replicates_recomputed": replica_counts,
            "total_stored_bootstrap_replicates_recomputed": replicate_checks,
            "maximum_normalized_distance_formula_error": maximum_ned_error,
            "maximum_recomputed_bootstrap_value_error": largest_bootstrap_error,
            "character_identity": identity_report,
            "rapidfuzz_version": importlib.metadata.version("RapidFuzz"),
            "independent_distance_implementation": Levenshtein_py.distance.__module__,
            "rapidfuzz_python_source_sha256": sha(Levenshtein_py.__file__),
            "normalizer_sha256": NORMALIZER_SHA256, "audit_script_sha256": sha(__file__),
            "new_asr_calls": 0, "new_encoder_calls": 0, "new_api_calls": 0,
            "input_sha256": {name: sha(run / name) for name in ("analysis-config.json", "summary.json", "distances.npz",
                "bootstrap-values.npz", "completion.json")},
            "full_input_sha256": {name: sha(full / name) for name in ("config.json", "summary.json", "transcript-mappings.json", "audit.json")},
            "scope": "Python/C++ implementation cross-check plus independent macro-distance, identity, paired interval and stored-bootstrap accounting; no ASR ground truth, semantic/retrieval assessment or independent scientific replication."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "full-run", "run"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"), "Run only inside Edinburgh Slurm")
    os.umask(0o077)
    try:
        result = audit(args)
    except Exception as exc:
        result = {"status": "fail", "slurm_job_id": os.environ["SLURM_JOB_ID"], "audit_script_sha256": sha(__file__),
                  "errors": [{"type": type(exc).__name__, "message": str(exc)}]}
        (args.run / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise
    (args.run / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
