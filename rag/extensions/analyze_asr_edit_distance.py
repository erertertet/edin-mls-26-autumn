"""Reference-free pairwise transcript edit distances; saved data, CPU Slurm only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
from itertools import combinations
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.stats import bootstrap

ARMS = ("glm", "whisper", "mai", "voxtral")
PAIRS = tuple(combinations(ARMS, 2))
PAIR_LABELS = ("overall",) + tuple("__".join(pair) for pair in PAIRS)
CONDITIONS = ("raw", "normalized")
UNITS = ("characters", "words")
METRICS = ("absolute", "normalized_distance")
NORMALIZER_SHA256 = "fff6993b2f1ae85db3a65a6c645a6e2ac3ad0ec415eb82af3f73a763de8e9dd2"
REPLICATES, SEED, BATCH, CHANGE_TOLERANCE = 5000, 20261117, 50, 1e-12


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, freeze=False):
    if freeze and path.exists():
        require(read(path) == value, "Frozen analysis configuration changed: " + path.name)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def distance_probe():
    cases = [("kitten", "sitting", 3), ("", "", 0), ("", "abc", 3),
             (["one", "two"], ["one", "three"], 1), ([], ["word"], 1),
             ("é", "e\u0301", 2), ("🐈", "🐕", 1)]
    for left, right, expected in cases:
        value = Levenshtein.distance(left, right, weights=(1, 1, 1), processor=None)
        reverse = Levenshtein.distance(right, left, weights=(1, 1, 1), processor=None)
        require(value == reverse == expected, "Levenshtein API/example or symmetry mismatch")
        denominator = max(len(left), len(right))
        ratio = value / denominator if denominator else 0.0
        library = Levenshtein.normalized_distance(left, right, weights=(1, 1, 1), processor=None)
        require(abs(library - ratio) <= 1e-15, "Normalized Levenshtein denominator mismatch")
    backend = Levenshtein.distance.__module__
    require("_cpp" in backend, "Expected the pinned RapidFuzz compiled implementation")
    return {"status": "pass", "examples_checked": len(cases), "distance_backend": backend,
            "weights": [1, 1, 1], "processor": None}


class QueryMeans:
    """One recorded query draw shared by all 56 paired feature coordinates."""
    def __init__(self, matrix):
        self.matrix = matrix
        self.draws = []

    def __call__(self, indices, axis=-1):
        indices = np.moveaxis(np.asarray(indices, dtype=np.intp), axis, -1)
        batch_shape = indices.shape[:-1]
        selected = indices.reshape(-1, len(self.matrix))
        values = self.matrix[selected].mean(axis=1).T
        if batch_shape:
            self.draws.append(selected.copy())
        return values.reshape((self.matrix.shape[1],) + batch_shape)


def interval(values):
    require(values.shape == (REPLICATES,) and np.isfinite(values).all(), "Invalid paired bootstrap values")
    bounds = np.quantile(values, [0.025, 0.975])
    return {"low": float(bounds[0]), "high": float(bounds[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "full-run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"), "Run in Edinburgh Slurm")
    os.umask(0o077)
    args.root, args.full_run, args.out = (path.resolve() for path in (args.root, args.full_run, args.out))
    require(args.out != args.full_run and args.out not in args.full_run.parents
            and args.full_run not in args.out.parents, "Use a separate output directory; preserve the full run")
    args.out.mkdir(parents=True, exist_ok=True)
    versions = {name: {"version": importlib.metadata.version(name), "path": importlib.util.find_spec(module).origin}
                for name, module in (("numpy", "numpy"), ("scipy", "scipy"), ("rapidfuzz", "rapidfuzz"))}
    require({name: value["version"] for name, value in versions.items()}
            == {"numpy": "2.4.6", "scipy": "1.17.1", "rapidfuzz": "3.14.6"}, "Pinned dependency versions changed")
    audit, full_config, old_summary = (read(args.full_run / name) for name in ("audit.json", "config.json", "summary.json"))
    require(audit["status"] == "pass" and old_summary["status"] == "complete"
            and audit["queries"] == 475 and audit["transcripts_checked"] == 1900, "Original full run is not audited complete")
    inputs = {"audit.json": sha(args.full_run / "audit.json")}
    for relative, expected in audit["artifact_sha256"].items():
        path = (args.full_run / relative).resolve()
        path.relative_to(args.full_run)
        require(sha(path) == expected, "Original audited artifact changed: " + relative)
        inputs[relative] = expected
    original_sources = {**full_config["input_sources"], **full_config["sources"]}
    for relative, expected in original_sources.items():
        require(sha(args.root / relative) == expected, "Original source/input changed: " + relative)
    normalizer = args.root / "rag/extensions/transcript_normalization.py"
    require(sha(normalizer) == full_config["normalizer_sha256"] == audit["normalizer_sha256"] == NORMALIZER_SHA256,
            "Frozen normalization rule changed")
    sys.path.insert(0, str(normalizer.parent))
    from transcript_normalization import POLICY, POLICY_VERSION, normalize_transcript
    require(full_config["policy"] == POLICY and full_config["policy_version"] == POLICY_VERSION,
            "Normalization policy declaration mismatch")
    cohorts = full_config["cohorts"]
    cohort_order = list(cohorts)
    require(set(cohort_order) == {"all", "practice", "dev", "pilot", "remaining_dev", "remaining_all"}, "Changed cohorts")
    ids = cohorts["all"]
    require(len(ids) == len(set(ids)) == 475, "Need every original query exactly once")
    for name, selected in cohorts.items():
        require(len(selected) == len(set(selected)) and set(selected) <= set(ids), "Invalid frozen cohort: " + name)
    mappings = read(args.full_run / "transcript-mappings.json")
    mapped = {(m["query_id"], m["arm"]): m for m in mappings}
    require(len(mapped) == len(mappings) == 1900
            and set(mapped) == {(q, arm) for q in ids for arm in ARMS}, "Incomplete or duplicate transcript mappings")
    texts = {}
    for (qid, arm), item in mapped.items():
        path = Path("transcripts") / arm / (hashlib.sha256(qid.encode()).hexdigest()[:20] + ".json")
        expected = item["source_sha256"]
        require(sha(args.full_run / path) == expected, "Source transcript hash mismatch")
        inputs[path.as_posix()] = expected
        source = read(args.full_run / path)
        require(source["query_id"] == qid and source["model"] == item["model"] == full_config["models"][arm]
                and source["status"] == "complete" and source["audio_sha256"] == item["audio_sha256"]
                and source["transcript"] == item["raw"], "Source transcript/mapping mismatch")
        require(normalize_transcript(item["raw"]) == item["normalized"]
                and normalize_transcript(item["normalized"]) == item["normalized"], "Frozen text normalization mismatch")
        for condition in CONDITIONS:
            text = item[condition]
            require(isinstance(text, str) and hashlib.sha256(text.encode("utf-8")).hexdigest()
                    == item[condition + "_text_sha256"], "Text SHA mismatch")
            texts[qid, arm, condition] = text
    identities = {}
    for name, selected in cohorts.items():
        counts = {c: sum(len({texts[q, arm, c] for arm in ARMS}) == 1 for q in selected) for c in CONDITIONS}
        old = old_summary["cohorts"][name]["text_agreement"]
        require(counts["raw"] == old["raw_identical_queries"] == audit["cohorts"][name]["raw_identical_queries"]
                and counts["normalized"] == old["normalized_identical_queries"]
                == audit["cohorts"][name]["normalized_identical_queries"], "Original exact-text identity count mismatch")
        identities[name] = {"raw_queries": counts["raw"], "normalized_queries": counts["normalized"]}
    require(identities["all"] == {"raw_queries": 195, "normalized_queries": 390}, "All-query identity control mismatch")
    sources = {"analyze_asr_edit_distance.py": sha(__file__), "transcript_normalization.py": sha(normalizer),
               "protocol": sha(args.root / "report/asr-edit-distance-protocol.md"),
               "requirements": sha(args.root / "rag/extensions/asr-edit-distance-requirements.txt")}
    bootstrap_plan = {"replicates": REPLICATES, "seed": SEED, "method": "percentile", "batch": BATCH,
                      "confidence": 0.95, "seeds": {name: SEED + index for index, name in enumerate(cohort_order)}}
    config = {"version": "asr-edit-distance-v1", "cohort_order": cohort_order, "cohorts": cohorts,
              "pairs": PAIRS, "pair_labels": PAIR_LABELS, "conditions": CONDITIONS, "units": UNITS, "metrics": METRICS,
              "weights": [1, 1, 1], "processor": None, "change_tolerance": CHANGE_TOLERANCE,
              "bootstrap": bootstrap_plan, "input_sha256": inputs, "source_sha256": sources,
              "original_source_sha256": original_sources, "versions": versions,
              "normalizer_sha256": NORMALIZER_SHA256, "policy_version": POLICY_VERSION,
              "character_unit": "Python Unicode code points; no extra preprocessing",
              "word_unit": "text.split(); case and punctuation retained; no extra preprocessing",
              "normalization": "unit-cost distance/max(sequence lengths); both empty -> 0",
              "overall": "mean six pair values within each query before averaging/bootstrap",
              "new_asr_calls": 0, "new_encoder_calls": 0, "new_api_calls": 0}
    # Convert tuples once so a resumed run compares the same JSON representation.
    config = json.loads(json.dumps(config))
    write(args.out / "analysis-config.json", config, freeze=True)
    probe = distance_probe()
    write(args.out / "api-probe.json", probe)
    shape = (len(ids), len(PAIRS), len(CONDITIONS), len(UNITS))
    edits, lengths_left, lengths_right = (np.empty(shape, dtype=np.int64) for _ in range(3))
    for qi, qid in enumerate(ids):
        for pi, (left, right) in enumerate(PAIRS):
            for ci, condition in enumerate(CONDITIONS):
                for ui, unit in enumerate(UNITS):
                    one, two = texts[qid, left, condition], texts[qid, right, condition]
                    if unit == "words":
                        one, two = one.split(), two.split()
                    edits[qi, pi, ci, ui] = Levenshtein.distance(one, two, weights=(1, 1, 1), processor=None)
                    lengths_left[qi, pi, ci, ui], lengths_right[qi, pi, ci, ui] = len(one), len(two)
        if (qi + 1) % 100 == 0 or qi + 1 == len(ids):
            print(json.dumps({"stage": "edit_distances", "queries": qi + 1, "total": len(ids)}), flush=True)
    denominator = np.maximum(lengths_left, lengths_right)
    normalized = np.divide(edits, denominator, out=np.zeros(shape, dtype=np.float64), where=denominator > 0)
    require((edits >= 0).all() and (edits <= denominator).all()
            and np.isfinite(normalized).all() and (normalized <= 1).all(), "Invalid edit-distance arrays")
    save_npz(args.out / "distances.npz", query_ids=np.array(ids), pairs=np.array(PAIRS), conditions=np.array(CONDITIONS),
             units=np.array(UNITS), edits=edits, lengths_left=lengths_left, lengths_right=lengths_right, normalized=normalized)
    # N, six pairs, condition, unit, metric -> N, seven pair labels, unit, metric, condition.
    features = np.stack([edits, normalized], axis=-1).transpose(0, 1, 3, 4, 2)
    features = np.concatenate([features.mean(axis=1, keepdims=True), features], axis=1)
    boot_arrays = {"pair_labels": np.array(PAIR_LABELS), "units": np.array(UNITS), "metrics": np.array(METRICS),
                   "conditions": np.array(CONDITIONS), "cohorts": np.array(cohort_order)}
    rows, cohort_metadata = [], {}
    positions = {qid: index for index, qid in enumerate(ids)}
    for name in cohort_order:
        selected = cohorts[name]
        require(len(selected) >= 2, "Bootstrap needs at least two cohort queries")
        values = features[[positions[q] for q in selected]]
        statistic = QueryMeans(values.reshape(len(selected), -1))
        result = bootstrap((np.arange(len(selected)),), statistic, n_resamples=REPLICATES, batch=BATCH,
                           vectorized=True, paired=True, axis=0, confidence_level=0.95, method="percentile",
                           rng=np.random.default_rng(bootstrap_plan["seeds"][name]))
        draws = result.bootstrap_distribution.reshape((7, 2, 2, 2, REPLICATES))
        indices = np.concatenate(statistic.draws).astype(np.int64, copy=False)
        require(indices.shape == (REPLICATES, len(selected)) and np.isfinite(draws).all(), "Bootstrap capture mismatch")
        boot_arrays["values_" + name], boot_arrays["indices_" + name] = draws, indices
        means = values.mean(axis=0)
        for pi, pair in enumerate(PAIR_LABELS):
            for ui, unit in enumerate(UNITS):
                row = {"cohort": name, "pair": pair, "unit": unit, "queries": len(selected)}
                for mi, metric in enumerate(METRICS):
                    raw_draws, norm_draws = draws[pi, ui, mi, 0], draws[pi, ui, mi, 1]
                    row[metric] = {"raw_mean": float(means[pi, ui, mi, 0]),
                        "normalized_mean": float(means[pi, ui, mi, 1]),
                        "delta_mean": float(np.mean(values[:, pi, ui, mi, 1] - values[:, pi, ui, mi, 0])),
                        "raw_ci": interval(raw_draws), "normalized_ci": interval(norm_draws),
                        "delta_ci": interval(norm_draws - raw_draws)}
                delta = values[:, pi, ui, 1, 1] - values[:, pi, ui, 1, 0]
                row["zero_distance"] = {"raw_queries": int(np.count_nonzero(values[:, pi, ui, 0, 0] == 0)),
                    "normalized_queries": int(np.count_nonzero(values[:, pi, ui, 0, 1] == 0))}
                row["change_counts"] = {"decrease": int(np.count_nonzero(delta < -CHANGE_TOLERANCE)),
                    "tie": int(np.count_nonzero(np.abs(delta) <= CHANGE_TOLERANCE)),
                    "increase": int(np.count_nonzero(delta > CHANGE_TOLERANCE))}
                require(sum(row["change_counts"].values()) == len(selected), "Change-count denominator mismatch")
                rows.append(row)
        overall_char = next(r for r in rows if r["cohort"] == name and r["pair"] == "overall" and r["unit"] == "characters")
        require(overall_char["zero_distance"] == identities[name], "Character edit-zero identity control mismatch")
        cohort_metadata[name] = {"queries": len(selected), "seed": bootstrap_plan["seeds"][name], "character_identity": identities[name]}
        print(json.dumps({"stage": "query_bootstrap", "cohort": name, "replicates": REPLICATES}), flush=True)
    save_npz(args.out / "bootstrap-values.npz", **boot_arrays)
    metadata = {"utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.root).decode().strip(),
        "hostname": socket.getfqdn(), "versions": versions, "normalizer_sha256": NORMALIZER_SHA256,
        "policy_version": POLICY_VERSION, "bootstrap": bootstrap_plan, "cohort_order": cohort_order,
        "pair_labels": PAIR_LABELS, "units": UNITS, "conditions": CONDITIONS, "metrics": METRICS,
        "change_tolerance": CHANGE_TOLERANCE,
        "source_sha256": sources, "input_sha256": {key: inputs[key] for key in
            ("audit.json", "config.json", "summary.json", "transcript-mappings.json")}, "api_probe": probe,
        "new_asr_calls": 0, "new_encoder_calls": 0, "new_api_calls": 0}
    require(len(rows) == 84, "Expected 84 cohort/pair/unit rows")
    write(args.out / "summary.json", {"status": "complete", "metadata": metadata, "cohorts": cohort_metadata, "rows": rows})
    artifacts = ("analysis-config.json", "api-probe.json", "distances.npz", "bootstrap-values.npz", "summary.json")
    write(args.out / "completion.json", {"status": "complete", "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "utc": datetime.now(timezone.utc).isoformat(), "queries": len(ids), "transcript_mappings": len(mappings),
        "model_pairs": len(PAIRS), "distances": int(edits.size), "summary_rows": len(rows), "bootstrap_replicates": REPLICATES,
        "new_asr_calls": 0, "new_encoder_calls": 0, "new_api_calls": 0,
        "artifact_sha256": {name: sha(args.out / name) for name in artifacts}})
    print(json.dumps({"status": "complete", "summary_rows": len(rows), "new_api_calls": 0}), flush=True)


if __name__ == "__main__":
    main()
