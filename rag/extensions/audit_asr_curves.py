"""Independent saved-count/bootstrap audit for ASR curves; Slurm, without model/API calls."""
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

METRICS = ("achieved_recall", "macro_precision", "micro_precision", "mean_tokens", "mean_documents")
CURVE_FIELDS = ("macro_recall", "macro_precision", "micro_precision", "mean_tokens", "mean_documents", "mean_hits")
FAMILIES = ("topk", "cutoff")
ARMS = ("glm", "whisper", "mai", "voxtral")
NORMALIZER_SHA256 = "fff6993b2f1ae85db3a65a6c645a6e2ac3ad0ec415eb82af3f73a763de8e9dd2"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tolerance(field):
    if field in ("mean_tokens",):
        return 2e-5
    if field in ("mean_documents", "mean_hits"):
        return 1e-7
    return 1e-10


def compare(actual, expected, field, label):
    if expected is None or not math.isfinite(float(expected)):
        require(actual is None or not math.isfinite(float(actual)), "Expected unsupported numeric value: " + label)
    else:
        require(actual is not None and math.isfinite(float(actual))
                and math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance(field)),
                "Numeric mismatch: " + label)


def pkey(panel, arm, condition, family, target=None):
    return ":".join((panel, "fixed" if target is None else f"{target:.2f}", arm, condition, family))


def average_curves(cache, indices, exact=False):
    """Directly aggregate resampled raw count arrays, not the producer's moments matrices."""
    h = cache["hits"][indices].astype(np.float64)
    d = cache["docs"][indices].astype(np.float64)
    t = cache["tokens"][indices].astype(np.float64)
    g = cache["gold"][indices].astype(np.float64)
    if exact:
        def avg(array):
            return np.asarray([math.fsum(array[:, case, point]) / len(indices)
                for case in range(array.shape[1]) for point in range(array.shape[2])]).reshape(array.shape[1:])
    else:
        def avg(array):
            return np.mean(array, axis=0)
    mean_hits, mean_documents = avg(h), avg(d)
    return {"macro_recall": avg(h / g[:, None, None]), "macro_precision": avg(h / d),
            "mean_tokens": avg(t), "mean_documents": mean_documents,
            "mean_hits": mean_hits, "micro_precision": mean_hits / mean_documents}


def single_curve(means, case, family_indices):
    return {field: means[field][case, family_indices] for field in CURVE_FIELDS}


def unsupported():
    return {"supported": False, "low_index": None, "high_index": None,
            "high_weight": None, "parameter_low": None, "parameter_high": None,
            **{field: None for field in ("achieved_recall", "macro_precision", "micro_precision",
                                       "mean_tokens", "mean_documents", "mean_hits")}}


def apply_selection(curve, selection):
    if not selection["supported"]:
        return unsupported()
    low, high, probability = selection["low_index"], selection["high_index"], selection["high_weight"]
    require(0 <= probability <= 1 and 0 <= low < len(curve["macro_recall"])
            and 0 <= high < len(curve["macro_recall"]), "Invalid selected mixture")
    def value(field):
        return float(curve[field][low] * (1 - probability) + curve[field][high] * probability)
    result = {key: selection[key] for key in ("supported", "low_index", "high_index", "high_weight",
                                               "parameter_low", "parameter_high")}
    result.update(achieved_recall=value("macro_recall"), macro_precision=value("macro_precision"),
                  mean_tokens=value("mean_tokens"), mean_documents=value("mean_documents"), mean_hits=value("mean_hits"))
    result["micro_precision"] = result["mean_hits"] / result["mean_documents"]
    return result


def choose_mixture(curve, parameters, target):
    """Independent implementation of the predeclared adjacent-recall/plateau rule."""
    recalls = curve["macro_recall"]
    if target < float(min(recalls)) or target > float(max(recalls)):
        return unsupported()
    ordered = sorted(range(len(recalls)), key=lambda i: float(recalls[i]))
    representatives = []
    group, anchor = [], None
    def cost_key(i):
        return (float(curve["mean_tokens"][i]), float(curve["mean_documents"][i]), float(parameters[i]), i)
    for i in ordered:
        if anchor is not None and float(recalls[i]) - anchor > 1e-12:
            representatives.append(min(group, key=cost_key))
            group = []
        if not group:
            anchor = float(recalls[i])
        group.append(i)
    representatives.append(min(group, key=cost_key))
    close = [i for i in representatives if abs(float(recalls[i]) - target) <= 1e-12]
    if close:
        low = high = min(close, key=lambda i: (abs(float(recalls[i]) - target), *cost_key(i)))
        weight = 0.0
    else:
        below = [i for i in representatives if float(recalls[i]) < target]
        above = [i for i in representatives if float(recalls[i]) > target]
        if not below or not above:
            return unsupported()
        low = max(below, key=lambda i: float(recalls[i]))
        high = min(above, key=lambda i: float(recalls[i]))
        weight = float((target - recalls[low]) / (recalls[high] - recalls[low]))
    return apply_selection(curve, {"supported": True, "low_index": int(low), "high_index": int(high),
        "high_weight": weight, "parameter_low": float(parameters[low]), "parameter_high": float(parameters[high])})


def compare_selection(actual, expected, label):
    require(set(actual) == set(expected), "Selection fields mismatch: " + label)
    for field in ("supported", "low_index", "high_index"):
        require(actual[field] == expected[field], "Selection identity mismatch: " + label + "/" + field)
    for field in set(expected) - {"supported", "low_index", "high_index"}:
        compare(actual[field], expected[field], field, label + "/" + field)


def reconstruct_points(cache, plan, calibration_means, evaluation_means):
    cases = [tuple(case) for case in cache["cases"].tolist()]
    groups = {family: np.flatnonzero(cache["families"] == family) for family in FAMILIES}
    selections, points = {}, []
    for panel in ("calibrated", "matched", "fixed"):
        for target in ([None] if panel == "fixed" else plan["targets"]):
            for ci, (arm, condition) in enumerate(cases):
                for family in FAMILIES:
                    parameters = cache["parameters"][groups[family]]
                    curve = single_curve(evaluation_means, ci, groups[family])
                    key = pkey(panel, arm, condition, family, target)
                    if panel == "calibrated":
                        selected = choose_mixture(single_curve(calibration_means, ci, groups[family]), parameters, target)
                        selections[key] = selected
                        values = apply_selection(curve, selected)
                        calibration_recall = selected["achieved_recall"]
                    elif panel == "matched":
                        selected = choose_mixture(curve, parameters, target)
                        values, calibration_recall = selected, None
                    else:
                        parameter = plan["fixed_k"] if family == "topk" else plan["fixed_tau"]
                        located = np.flatnonzero(np.isclose(parameters, parameter, rtol=0, atol=1e-12))
                        require(len(located) == 1, "Ambiguous fixed grid control")
                        index = int(located[0])
                        selected = {"supported": True, "low_index": index, "high_index": index, "high_weight": 0.0,
                                    "parameter_low": parameter, "parameter_high": parameter}
                        values, calibration_recall = apply_selection(curve, selected), None
                    points.append({"key": key, "panel": panel, "arm": arm, "condition": condition,
                        "family": family, "target": target, "supported": values["supported"],
                        "selection": selected, "calibration_recall": calibration_recall,
                        "parameters": {"low": selected["parameter_low"], "high": selected["parameter_high"],
                                       "high_weight": selected["high_weight"]},
                        "metrics": {field: values[field] for field in METRICS}})
    for panel in ("calibrated", "matched"):
        for target in plan["targets"]:
            group = [point for point in points if point["panel"] == panel and point["target"] == target]
            common = all(point["supported"] for point in group)
            for point in group:
                point["common_target_supported"] = common
                if not common:
                    point["metrics"] = dict.fromkeys(METRICS)
    return points, selections


def expected_interval(values, point, plan):
    valid = np.isfinite(values)
    count = int(valid.sum())
    total = len(values)
    available = point is not None and math.isfinite(float(point)) and count >= plan["bootstrap"]["minimum_supported_fraction"] * total
    alpha = (1 - plan["bootstrap"]["confidence"]) / 2
    bounds = np.quantile(values[valid], [alpha, 1 - alpha]) if available else [None, None]
    return {"status": "available" if available else "unsupported", "valid_bootstraps": count,
            "total_bootstraps": total, "low": bounds[0], "high": bounds[1]}


def compare_interval(actual, expected, metric, label):
    require(set(actual) == set(expected) and all(actual[field] == expected[field]
        for field in ("status", "valid_bootstraps", "total_bootstraps")), "CI support/count mismatch: " + label)
    for field in ("low", "high"):
        compare(actual[field], expected[field], metric, label + "/" + field)


def expected_contrast_specs(plan):
    result = []
    def add(panel, kind, arm, condition, family, target, left, right):
        result.append({"panel": panel, "kind": kind, "arm": arm, "condition": condition,
                       "family": family, "target": target, "left_key": left, "right_key": right})
    for panel in ("calibrated", "matched", "fixed"):
        for target in ([None] if panel == "fixed" else plan["targets"]):
            for family in FAMILIES:
                for arm in ARMS:
                    add(panel, "normalization", arm, None, family, target,
                        pkey(panel, arm, "normalized", family, target), pkey(panel, arm, "raw", family, target))
                for condition in ("raw", "normalized"):
                    for arm in ARMS[1:]:
                        add(panel, "asr_vs_glm", arm, condition, family, target,
                            pkey(panel, arm, condition, family, target), pkey(panel, "glm", condition, family, target))
            for arm in ARMS:
                for condition in ("raw", "normalized"):
                    add(panel, "cutoff_vs_topk", arm, condition, None, target,
                        pkey(panel, arm, condition, "cutoff", target), pkey(panel, arm, condition, "topk", target))
    return result


def audit(args):
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"), "Run only inside Edinburgh Slurm")
    run, plan = args.run, read(args.plan)
    require(plan["bootstrap"]["replicates"] == 5000 and plan["bootstrap"]["confidence"] == .95
            and plan["bootstrap"]["minimum_supported_fraction"] == .95 and plan["bootstrap"]["method"] == "percentile",
            "Unexpected bootstrap plan")
    require(plan["targets"] == [.3, .4, .5] and plan["fixed_k"] == 5 and plan["fixed_tau"] == .8925,
            "Unexpected recall targets or controls")
    metadata, config, partition = (read(run / name) for name in ("metadata.json", "config.json", "partition.json"))
    require(metadata["status"] == "complete" and metadata["plan_sha256"] == config["plan_sha256"] == sha(args.plan)
            and config["plan"] == plan, "Curve preparation plan mismatch")
    require(metadata["source_sha256"] == config["source_sha256"] and metadata["input_sha256"] == config["input_sha256"],
            "Curve preparation provenance mismatch")
    require(sha(args.root / "rag/extensions/transcript_normalization.py")
            == metadata["normalizer_sha256"] == config["normalizer_sha256"] == NORMALIZER_SHA256,
            "Frozen normalization rule changed")
    for relative, expected in metadata["source_sha256"].items():
        path = args.root / relative if "/" in relative else args.root / "rag/extensions" / relative
        require(sha(path) == expected, "Curve preparation source changed: " + relative)
    for name in ("curve-cache.npz", "partition.json", "config.json", "doc-token-cache.npz"):
        require(sha(run / name) == metadata["artifact_sha256"][name], "Prepared cache artifact changed: " + name)
    fixed = metadata["fixed_policy_reproduction"]
    require(fixed["status"] == "pass" and fixed["cells"] == 7600 and fixed["exact_ids_recall_tokens_docs_hits_and_fallback"]
            and fixed["max_abs_selected_score_drift"] <= 1e-6, "Original fixed-policy preparation gate failed")
    require(metadata["asr_calls"] == metadata["encoder_calls"] == metadata["api_calls"] == 0, "Unexpected model/API declaration")
    full = args.root / "rag/results/asr-four-model-full-20261117"
    require(read(full / "audit.json")["status"] == "pass", "Original full experiment audit did not pass")
    for name in ("audit.json", "config.json", "selection.json", "query-vectors.npz", "retrieval.jsonl"):
        require(sha(full / name) == metadata["input_sha256"]["full-run/" + name], "Original audited input changed: " + name)
    with np.load(run / "curve-cache.npz", allow_pickle=False) as saved:
        cache = {name: saved[name] for name in saved.files}
    require(set(cache) == {"query_ids", "splits", "cases", "families", "parameters", "hits", "docs", "tokens", "gold"},
            "Unexpected prepared count-cache fields")
    qids, splits = cache["query_ids"].tolist(), cache["splits"].tolist()
    cases = [tuple(case) for case in cache["cases"].tolist()]
    require(len(qids) == len(set(qids)) == 475 and cache["splits"].shape == (475,)
            and cache["cases"].shape == (8, 2) and cases == [tuple(c) for c in plan["cases"]], "Cache query/case coverage mismatch")
    expected_families = ["topk"] * len(plan["topk_grid"]) + ["cutoff"] * len(plan["tau_grid"])
    expected_parameters = np.asarray(plan["topk_grid"] + plan["tau_grid"], dtype=np.float64)
    require(cache["families"].tolist() == expected_families and np.array_equal(cache["parameters"], expected_parameters)
            and len(expected_families) == 307, "Prepared policy grid mismatch")
    for name, dtype in (("hits", np.int32), ("docs", np.int32), ("tokens", np.int64)):
        require(cache[name].shape == (475, 8, 307) and cache[name].dtype == dtype, "Count-array schema mismatch: " + name)
    require(cache["gold"].shape == (475,) and cache["gold"].dtype == np.int32 and np.all(cache["gold"] > 0)
            and np.all(cache["hits"] >= 0) and np.all(cache["hits"] <= cache["gold"][:, None, None])
            and np.all(cache["hits"] <= cache["docs"]) and np.all(cache["docs"] >= 1)
            and np.all(cache["docs"] <= 24000) and np.all(cache["tokens"] >= 0), "Invalid curve-cache counts")
    family_indices = {family: np.flatnonzero(cache["families"] == family) for family in FAMILIES}
    for family in FAMILIES:
        for name in ("hits", "docs", "tokens"):
            differences = np.diff(cache[name][:, :, family_indices[family]], axis=2)
            require(np.all(differences >= 0) if family == "topk" else np.all(differences <= 0), "Prefix monotonicity mismatch")
    with np.load(run / "doc-token-cache.npz", allow_pickle=False) as saved:
        record_ids, document_tokens = saved["record_ids"], saved["tokens"]
    require(record_ids.shape == document_tokens.shape == (24000,) and len(set(record_ids.tolist())) == 24000
            and document_tokens.dtype == np.int32 and np.all(document_tokens >= 0), "Document token-cache schema mismatch")
    full_index = int(np.flatnonzero((cache["families"] == "topk") & (cache["parameters"] == 24000))[0])
    require(np.all(cache["docs"][:, :, full_index] == 24000)
            and np.array_equal(cache["hits"][:, :, full_index], np.broadcast_to(cache["gold"][:, None], (475, 8)))
            and np.all(cache["tokens"][:, :, full_index] == document_tokens.sum(dtype=np.int64)), "Full-corpus endpoint mismatch")

    cal_ids, eval_ids, pilot_ids = (partition[field] for field in ("calibration_ids", "evaluation_ids", "pilot_ids"))
    require(len(cal_ids) == len(set(cal_ids)) == 350 and len(eval_ids) == len(set(eval_ids)) == 117
            and len(pilot_ids) == len(set(pilot_ids)) == 8
            and set(cal_ids).isdisjoint(eval_ids) and set(cal_ids).isdisjoint(pilot_ids) and set(eval_ids).isdisjoint(pilot_ids)
            and set(cal_ids + eval_ids + pilot_ids) == set(qids), "Calibration/evaluation/pilot partition mismatch")
    split_for = dict(zip(qids, splits))
    require(all(split_for[q] == "practice" for q in cal_ids) and all(split_for[q] == "dev" for q in eval_ids + pilot_ids),
            "Partition split labels mismatch")
    require(partition == config["partition"], "Frozen partition changed")
    cal_indices = np.asarray([qids.index(q) for q in cal_ids])
    eval_indices = np.asarray([qids.index(q) for q in eval_ids])
    original_rows = [json.loads(line) for line in (full / "retrieval.jsonl").read_text().splitlines() if line.strip()]
    original = {(r["query_id"], r["arm"], r["condition"], r["rule"]): r for r in original_rows}
    require(len(original) == len(original_rows) == 7600, "Original fixed-policy cells missing/duplicated")
    fixed_controls_checked = 0
    for qi, qid in enumerate(qids):
        for ci, (arm, condition) in enumerate(cases):
            for rule, family, parameter in (("top5", "topk", 5), ("cutoff", "cutoff", .8925)):
                index = int(np.flatnonzero((cache["families"] == family)
                            & np.isclose(cache["parameters"], parameter, rtol=0, atol=1e-12))[0])
                row = original[qid, arm, condition, rule]
                require(row["split"] == splits[qi] and row["relevant_hits"] == int(cache["hits"][qi, ci, index])
                        and row["documents"] == int(cache["docs"][qi, ci, index])
                        and row["context_tokens"] == int(cache["tokens"][qi, ci, index])
                        and row["total_relevant"] == int(cache["gold"][qi])
                        and row["recall"] == int(cache["hits"][qi, ci, index]) / int(cache["gold"][qi]),
                        "Cached original fixed-policy count mismatch")
                fixed_controls_checked += 1

    analysis, completion, analysis_config = (read(run / name) for name in ("analysis.json", "analysis-completion.json", "analysis-config.json"))
    require(analysis["status"] == completion["status"] == "complete" and analysis["plan"] == analysis_config["plan"] == plan,
            "Analysis completion/plan mismatch")
    require(analysis_config["cache_metadata_sha256"] == sha(run / "metadata.json")
            and analysis_config["calibration_ids"] == cal_ids and analysis_config["evaluation_ids"] == eval_ids,
            "Analysis frozen input mismatch")
    for name in ("analysis.json", "curves.csv", "calibration-policies.json", "bootstrap-distribution.npz"):
        require(sha(run / name) == completion["artifact_sha256"][name], "Analysis artifact changed: " + name)
    require(analysis["source_sha256"] == analysis_config["source_sha256"], "Analysis source declarations disagree")
    for name, expected in analysis["source_sha256"].items():
        path = args.plan if name == "plan" else args.root / "report/asr-matched-curves-protocol.md" if name == "protocol" else args.root / "rag/extensions" / name
        require(sha(path) == expected, "Frozen analysis source changed: " + name)
    for owner in (analysis["metadata"], completion, analysis_config):
        require(all(owner[name] == 0 for name in ("new_asr_calls", "new_encoder_calls", "new_api_calls")), "Unexpected ASR/encoder/API call declaration")
    require(analysis["metadata"]["calibration_queries"] == 350 and analysis["metadata"]["evaluation_queries"] == 117
            and analysis["metadata"]["bootstrap_replicates"] == completion["bootstrap_replicates"] == 5000
            and analysis["metadata"]["bootstrap_seed"] == analysis["metadata"]["seed"] == plan["bootstrap"]["seed"],
            "Analysis sample/replicate/seed declaration mismatch")
    with np.load(run / "bootstrap-distribution.npz", allow_pickle=False) as saved:
        values, cal_draws, eval_draws = saved["values"], saved["calibration_indices"], saved["evaluation_indices"]
        saved_keys, saved_metrics = saved["point_keys"].tolist(), saved["metrics"].tolist()
    require(values.shape == (112, 5, 5000) and not np.isinf(values).any() and saved_metrics == list(METRICS), "Bootstrap distribution schema mismatch")
    require(cal_draws.shape == (5000, 350) and eval_draws.shape == (5000, 117)
            and cal_draws.dtype.kind in "iu" and eval_draws.dtype.kind in "iu"
            and np.all((cal_draws >= 0) & (cal_draws < 350)) and np.all((eval_draws >= 0) & (eval_draws < 117)),
            "Stored bootstrap query indices invalid")
    calibration_means = average_curves(cache, cal_indices, exact=True)
    evaluation_means = average_curves(cache, eval_indices, exact=True)
    expected_points, expected_choices = reconstruct_points(cache, plan, calibration_means, evaluation_means)
    points = analysis["calibrated_points"] + analysis["matched_points"] + analysis["fixed_controls"]
    require(len(points) == len(expected_points) == len(set(saved_keys)) == 112
            and [p["key"] for p in points] == saved_keys == [p["key"] for p in expected_points], "Point keys/order mismatch")
    policies = read(run / "calibration-policies.json")
    require(policies["calibration_ids"] == cal_ids and policies["calibration_source"] == "practice only"
            and set(policies["policies"]) == set(expected_choices), "Practice-only policy coverage mismatch")
    for key, expected in expected_choices.items():
        compare_selection(policies["policies"][key], expected, "practice policy/" + key)
    interval_count = 0
    for pi, (actual, expected) in enumerate(zip(points, expected_points)):
        for name in ("key", "panel", "arm", "condition", "family", "target", "supported"):
            require(actual[name] == expected[name], "Point identity/support mismatch: " + expected["key"])
        if expected["panel"] != "fixed":
            require(actual["common_target_supported"] == expected["common_target_supported"], "Common target point support mismatch")
        else:
            require(np.isfinite(values[pi]).all(), "Fixed controls have unsupported bootstrap values")
        compare(actual["calibration_recall"], expected["calibration_recall"], "achieved_recall", "calibration recall")
        compare_selection(actual["selection"], expected["selection"], expected["key"])
        for field in ("low", "high", "high_weight"):
            compare(actual["parameters"][field], expected["parameters"][field], field, "point mixture parameter")
        require(set(actual["metrics"]) == set(actual["intervals"]) == set(METRICS), "Point metric coverage mismatch")
        for mi, metric in enumerate(METRICS):
            compare(actual["metrics"][metric], expected["metrics"][metric], metric, expected["key"] + "/" + metric)
            compare_interval(actual["intervals"][metric], expected_interval(values[pi, mi], expected["metrics"][metric], plan),
                             metric, expected["key"] + "/" + metric)
            interval_count += 1

    key_index = {key: i for i, key in enumerate(saved_keys)}
    support_rows = {(row["panel"], row["target"]): row for row in analysis["support"]}
    require(len(support_rows) == len(analysis["support"]) == 6, "Common target support panel coverage mismatch")
    common_support = []
    for panel in ("calibrated", "matched"):
        for target in plan["targets"]:
            indices = [i for i, p in enumerate(points) if p["panel"] == panel and p["target"] == target]
            group = values[indices]
            require(len(indices) == 16, "Common target group has unexpected size")
            finite = np.isfinite(group)
            valid = finite.all(axis=(0, 1))
            empty = np.isnan(group).all(axis=(0, 1))
            require(np.all(valid | empty), "Partial-NaN common target panel detected")
            row = support_rows[panel, target]
            require(row["valid_bootstraps"] == int(valid.sum()) and row["total_bootstraps"] == 5000
                    and row["intervals_available"] == (int(valid.sum()) >= 4750), "Common target support count mismatch")
            common_support.append({"panel": panel, "target": target, "valid_bootstraps": int(valid.sum()), "total_bootstraps": 5000})
    specs = expected_contrast_specs(plan)
    contrasts = analysis["contrasts"]
    require(len(specs) == len(contrasts) == completion["paired_contrasts"] == 196
            and completion["points"] == 112, "Contrast/point count mismatch")
    for actual, spec in zip(contrasts, specs):
        require(all(actual[name] == expected for name, expected in spec.items()), "Paired contrast identity mismatch")
        left_index, right_index = key_index[spec["left_key"]], key_index[spec["right_key"]]
        require(set(actual["metrics"]) == set(actual["intervals"]) == set(METRICS), "Contrast metrics missing")
        for mi, metric in enumerate(METRICS):
            left, right = expected_points[left_index]["metrics"][metric], expected_points[right_index]["metrics"][metric]
            difference = None if left is None or right is None else left - right
            compare(actual["metrics"][metric], difference, metric, "paired contrast/" + metric)
            distribution = values[left_index, mi] - values[right_index, mi]
            compare_interval(actual["intervals"][metric], expected_interval(distribution, difference, plan), metric, "paired contrast CI")
            interval_count += 1

    # Audit a fixed, value-independent sample of stored bootstrap replicates. No new query draws are generated.
    rng = np.random.default_rng(plan["bootstrap"]["seed"] + 1031)
    replicate_checks = sorted([0, 4999, *rng.choice(np.arange(1, 4999), 23, replace=False).tolist()])
    maximum_replicate_difference = {metric: 0.0 for metric in METRICS}
    for bi in replicate_checks:
        cm = average_curves(cache, cal_indices[cal_draws[bi]])
        em = average_curves(cache, eval_indices[eval_draws[bi]])
        rebuilt, _ = reconstruct_points(cache, plan, cm, em)
        for pi, point in enumerate(rebuilt):
            for mi, metric in enumerate(METRICS):
                expected = point["metrics"][metric]
                observed = values[pi, mi, bi]
                compare(observed, expected, metric, "stored bootstrap replicate " + str(bi) + "/" + point["key"] + "/" + metric)
                if expected is not None:
                    maximum_replicate_difference[metric] = max(maximum_replicate_difference[metric], abs(float(observed) - expected))
        print(json.dumps({"audit_stage": "stored_bootstrap_replicate", "replicate": bi}), flush=True)

    cohort_ids = {"all": qids, "practice": cal_ids, "remaining_dev": eval_ids, "pilot": pilot_ids,
                  "dev": [q for q, split in zip(qids, splits) if split == "dev"],
                  "remaining_all": [q for q in qids if q not in pilot_ids]}
    cohort_means = {cohort: average_curves(cache, np.asarray([qids.index(q) for q in selected]), exact=True)
                    for cohort, selected in cohort_ids.items()}
    curve_rows = list(csv.DictReader((run / "curves.csv").open(encoding="utf-8", newline="")))
    require(len(curve_rows) == 6 * 8 * 307, "CSV curve coverage mismatch")
    seen = set()
    for row in curve_rows:
        ci = cases.index((row["arm"], row["condition"]))
        pi = np.flatnonzero((cache["families"] == row["family"])
                           & np.isclose(cache["parameters"], float(row["parameter"]), rtol=0, atol=1e-12))
        require(len(pi) == 1 and row["cohort"] in cohort_ids, "CSV unknown curve point")
        key = row["cohort"], ci, int(pi[0])
        require(key not in seen, "Duplicate CSV curve point")
        seen.add(key)
        for field in CURVE_FIELDS:
            compare(float(row[field]), cohort_means[row["cohort"]][field][ci, int(pi[0])], field, "CSV curve/" + field)
    require(len(seen) == 6 * 8 * 307, "Incomplete CSV curve points")
    return {"status": "pass", "errors": [], "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "utc": datetime.now(timezone.utc).isoformat(), "analysis_slurm_job_id": completion["slurm_job_id"],
            "calibration_queries": 350, "evaluation_queries": 117, "pilot_queries": 8,
            "points_independently_recomputed": 112, "practice_policy_selections_checked": 48,
            "paired_contrasts_checked": 196, "point_and_contrast_intervals_checked": interval_count,
            "bootstrap_replicates_per_interval": 5000, "stored_bootstrap_replicates_recomputed": replicate_checks,
            "max_abs_replicate_metric_differences": maximum_replicate_difference,
            "common_target_support": common_support, "fixed_original_count_controls_checked": fixed_controls_checked,
            "csv_curve_points_checked": len(curve_rows), "metric_units": "recall/precision fractions; tokens/documents as counts",
            "new_asr_calls": 0, "new_encoder_calls": 0, "new_api_calls": 0,
            "audit_script_sha256": sha(__file__), "plan_sha256": sha(args.plan),
            "input_sha256": {name: sha(run / name) for name in ("metadata.json", "partition.json", "curve-cache.npz",
                "analysis-config.json", "analysis.json", "calibration-policies.json", "bootstrap-distribution.npz", "analysis-completion.json")},
            "scope": "Independent saved-count aggregates and mixture selections; all saved-distribution percentile CIs/paired differences; 25 stored bootstrap replicates rebuilt from counts. No ASR, encoder, API, or new query bootstrap draws."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run", "plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"), "Run only inside Edinburgh Slurm")
    os.umask(0o077)
    try:
        result = audit(args)
    except Exception as exc:
        result = {"status": "fail", "slurm_job_id": os.environ["SLURM_JOB_ID"], "audit_script_sha256": sha(__file__),
                  "errors": [{"type": type(exc).__name__, "message": str(exc)}]}
        (args.run / "curve-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise
    (args.run / "curve-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
