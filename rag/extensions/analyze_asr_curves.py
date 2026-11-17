"""Separated-calibration curve inference using saved counts; CPU Slurm only."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import warnings

import numpy as np
from scipy.stats import bootstrap
from asr_curve_statistics import aggregate, mix_at_recall, mix_evaluate

METRICS = ("achieved_recall", "macro_precision", "micro_precision", "mean_tokens", "mean_documents")
FAMILIES = ("topk", "cutoff")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write(path, value, freeze=False):
    value = clean(value)
    if freeze and path.exists():
        require(read(path) == value, "Frozen analysis configuration changed: " + path.name)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pkey(panel, arm, condition, family, target=None):
    return ":".join([panel, "fixed" if target is None else f"{target:.2f}", arm, condition, family])


class CurveInference:
    """Shared query multiplicities across all cases, parameters and contrasts."""
    def __init__(self, cache, partition, plan):
        self.plan = plan
        self.qids = cache["query_ids"].tolist()
        self.cases = [tuple(x) for x in cache["cases"].tolist()]
        self.params = cache["parameters"]
        self.families = cache["families"]
        self.family_indices = {f: np.flatnonzero(self.families == f) for f in FAMILIES}
        hits, docs, tokens = (cache[name].astype(np.float64) for name in ("hits", "docs", "tokens"))
        self.features = np.stack([hits / cache["gold"][:, None, None], hits / docs,
                                  tokens, docs, hits], axis=-1)
        self.cal_idx = np.asarray([self.qids.index(q) for q in partition["calibration_ids"]])
        self.eval_idx = np.asarray([self.qids.index(q) for q in partition["evaluation_ids"]])
        self.cal_matrix = self.features[self.cal_idx].reshape(len(self.cal_idx), -1)
        self.eval_matrix = self.features[self.eval_idx].reshape(len(self.eval_idx), -1)
        self.mean_shape = self.features.shape[1:]
        self.specs = []
        for panel in ("calibrated", "matched"):
            for target in plan["targets"]:
                for arm, condition in self.cases:
                    for family in FAMILIES:
                        self.specs.append({"key": pkey(panel, arm, condition, family, target), "panel": panel,
                            "arm": arm, "condition": condition, "family": family, "target": target})
        for arm, condition in self.cases:
            for family in FAMILIES:
                self.specs.append({"key": pkey("fixed", arm, condition, family), "panel": "fixed",
                                   "arm": arm, "condition": condition, "family": family, "target": None})
        self.case_indices = {case: i for i, case in enumerate(self.cases)}
        self.cal_draws, self.eval_draws = [], []
        self.finished = 0

    def curve(self, means, case, family):
        values = means[case][self.family_indices[family]]
        return {"macro_recall": values[:, 0], "macro_precision": values[:, 1],
                "mean_tokens": values[:, 2], "mean_documents": values[:, 3], "mean_hits": values[:, 4],
                "micro_precision": values[:, 4] / values[:, 3]}

    def calibration_choices(self, means):
        result = {}
        for target in self.plan["targets"]:
            for ci, (arm, condition) in enumerate(self.cases):
                for family in FAMILIES:
                    curve = self.curve(means, ci, family)
                    result[pkey("calibrated", arm, condition, family, target)] = mix_at_recall(
                        curve, self.params[self.family_indices[family]], target)
        return result

    def evaluate(self, calibration_means, evaluation_means, choices=None, details=False):
        choices = choices if choices is not None else self.calibration_choices(calibration_means)
        rows = []
        for spec in self.specs:
            ci = self.case_indices[spec["arm"], spec["condition"]]
            family = spec["family"]
            curve = self.curve(evaluation_means, ci, family)
            parameters = self.params[self.family_indices[family]]
            if spec["panel"] == "calibrated":
                selection = choices[spec["key"]]
                value = mix_evaluate(curve, selection)
                cal_recall = selection["achieved_recall"]
            elif spec["panel"] == "matched":
                selection = mix_at_recall(curve, parameters, spec["target"])
                value, cal_recall = selection, None
            else:
                parameter = self.plan["fixed_k"] if family == "topk" else self.plan["fixed_tau"]
                position = np.flatnonzero(np.isclose(parameters, parameter, atol=1e-12, rtol=0))
                require(len(position) == 1, "Fixed control is not a unique grid point")
                i = int(position[0])
                selection = {"supported": True, "low_index": i, "high_index": i, "high_weight": 0.0,
                             "parameter_low": parameter, "parameter_high": parameter}
                value = mix_evaluate(curve, selection)
                cal_recall = None
            rows.append(dict(spec, supported=value["supported"], calibration_recall=cal_recall,
                parameters={"low": selection["parameter_low"], "high": selection["parameter_high"],
                            "high_weight": selection["high_weight"]},
                selection=selection, metrics={name: value[name] for name in METRICS}))
        # A common feasible replicate set at each panel/target keeps every paired contrast aligned.
        for panel in ("calibrated", "matched"):
            for target in self.plan["targets"]:
                group = [r for r in rows if r["panel"] == panel and r["target"] == target]
                common = all(r["supported"] for r in group)
                for row in group:
                    row["common_target_supported"] = common
                    if not common:
                        row["metrics"] = {name: float("nan") for name in METRICS}
        array = np.asarray([[row["metrics"][name] for name in METRICS] for row in rows]).reshape(-1)
        return (array, rows) if details else array

    def statistic(self, calibration_indices, evaluation_indices, axis=-1):
        cal = np.moveaxis(np.asarray(calibration_indices, dtype=np.intp), axis, -1)
        ev = np.moveaxis(np.asarray(evaluation_indices, dtype=np.intp), axis, -1)
        batch_shape = cal.shape[:-1]
        require(ev.shape[:-1] == batch_shape, "Bootstrap group batches are not aligned")
        cal, ev = cal.reshape(-1, len(self.cal_idx)), ev.reshape(-1, len(self.eval_idx))
        wc = np.stack([np.bincount(x, minlength=len(self.cal_idx)) for x in cal]).astype(np.float64)
        we = np.stack([np.bincount(x, minlength=len(self.eval_idx)) for x in ev]).astype(np.float64)
        cm = (wc @ self.cal_matrix / len(self.cal_idx)).reshape((len(cal),) + self.mean_shape)
        em = (we @ self.eval_matrix / len(self.eval_idx)).reshape((len(ev),) + self.mean_shape)
        values = np.stack([self.evaluate(c, e) for c, e in zip(cm, em)], axis=-1)
        if batch_shape:
            self.cal_draws.append(cal.copy())
            self.eval_draws.append(ev.copy())
            self.finished += len(cal)
            if self.finished % 250 == 0:
                print(json.dumps({"stage": "paired_bootstrap", "completed": self.finished,
                                  "total": self.plan["bootstrap"]["replicates"]}), flush=True)
        return values.reshape((values.shape[0],) + batch_shape)


def interval(values, point, plan):
    valid = np.isfinite(values)
    count = int(valid.sum())
    total = len(values)
    available = np.isfinite(point) and count >= plan["bootstrap"]["minimum_supported_fraction"] * total
    bounds = np.quantile(values[valid], [.025, .975]) if available else [None, None]
    return {"low": bounds[0], "high": bounds[1], "valid_bootstraps": count,
            "total_bootstraps": total, "status": "available" if available else "unsupported"}


def build_contrasts(points, draws, plan):
    by_key = {row["key"]: (i, row) for i, row in enumerate(points)}
    result = []
    def add(panel, kind, arm, condition, family, target, left, right):
        li, a = by_key[left]
        ri, b = by_key[right]
        metrics = {m: a["metrics"][m] - b["metrics"][m] for m in METRICS}
        result.append({"panel": panel, "kind": kind, "arm": arm, "condition": condition,
            "family": family, "target": target, "left_key": left, "right_key": right,
            "metrics": metrics, "intervals": {m: interval(draws[li, mi] - draws[ri, mi], metrics[m], plan)
                                              for mi, m in enumerate(METRICS)}})
    arms = ("glm", "whisper", "mai", "voxtral")
    for panel in ("calibrated", "matched", "fixed"):
        for target in ([None] if panel == "fixed" else plan["targets"]):
            for family in FAMILIES:
                for arm in arms:
                    add(panel, "normalization", arm, None, family, target,
                        pkey(panel, arm, "normalized", family, target), pkey(panel, arm, "raw", family, target))
                for condition in ("raw", "normalized"):
                    for arm in arms[1:]:
                        add(panel, "asr_vs_glm", arm, condition, family, target,
                            pkey(panel, arm, condition, family, target), pkey(panel, "glm", condition, family, target))
            for arm in arms:
                for condition in ("raw", "normalized"):
                    add(panel, "cutoff_vs_topk", arm, condition, None, target,
                        pkey(panel, arm, condition, "cutoff", target), pkey(panel, arm, condition, "topk", target))
    return result


def export_curves(cache, partition, out):
    qids = cache["query_ids"].tolist()
    splits = cache["splits"].tolist()
    cases = cache["cases"].tolist()
    cohorts = {"all": qids, "practice": partition["calibration_ids"], "remaining_dev": partition["evaluation_ids"],
               "pilot": partition["pilot_ids"], "dev": [q for q, s in zip(qids, splits) if s == "dev"],
               "remaining_all": [q for q in qids if q not in partition["pilot_ids"]]}
    with (out / "curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["cohort", "arm", "condition", "family", "parameter",
                "macro_recall", "macro_precision", "micro_precision", "mean_tokens", "mean_documents", "mean_hits"])
        writer.writeheader()
        for cohort, ids in cohorts.items():
            ix = np.asarray([qids.index(q) for q in ids])
            for ci, (arm, condition) in enumerate(cases):
                curve = aggregate(cache["hits"][ix, ci], cache["docs"][ix, ci], cache["tokens"][ix, ci], cache["gold"][ix])
                for pi, (family, parameter) in enumerate(zip(cache["families"], cache["parameters"])):
                    writer.writerow(dict(cohort=cohort, arm=arm, condition=condition, family=str(family), parameter=float(parameter),
                                         **{name: float(values[pi]) for name, values in curve.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"), "Run only in Edinburgh Slurm")
    os.umask(0o077)
    run = args.run.resolve()
    plan, metadata, partition = read(args.plan), read(run / "metadata.json"), read(run / "partition.json")
    require(metadata["status"] == "complete" and metadata["plan_sha256"] == sha(args.plan), "Cache/plan mismatch")
    for name in ("curve-cache.npz", "partition.json", "config.json", "doc-token-cache.npz"):
        require(sha(run / name) == metadata["artifact_sha256"][name], "Curve cache changed after preparation")
    sources = {Path(__file__).name: sha(__file__), "asr_curve_statistics.py": sha(Path(__file__).with_name("asr_curve_statistics.py")),
               "plan": sha(args.plan), "protocol": sha(Path(__file__).resolve().parents[2] / "report/asr-matched-curves-protocol.md")}
    write(run / "analysis-config.json", {"plan": plan, "source_sha256": sources,
          "cache_metadata_sha256": sha(run / "metadata.json"), "calibration_ids": partition["calibration_ids"],
          "evaluation_ids": partition["evaluation_ids"], "new_asr_calls": 0, "new_encoder_calls": 0, "new_api_calls": 0}, freeze=True)
    with np.load(run / "curve-cache.npz", allow_pickle=False) as loaded:
        cache = {name: loaded[name] for name in loaded.files}
    inference = CurveInference(cache, partition, plan)
    cm = inference.cal_matrix.mean(axis=0).reshape(inference.mean_shape)
    choices = inference.calibration_choices(cm)
    # Freeze practice-only choices before computing any evaluation mean curve.
    write(run / "calibration-policies.json", {"calibration_ids": partition["calibration_ids"],
          "calibration_source": "practice only", "policies": choices}, freeze=True)
    em = inference.eval_matrix.mean(axis=0).reshape(inference.mean_shape)
    point_values, points = inference.evaluate(cm, em, choices, details=True)
    export_curves(cache, partition, run)
    b = plan["bootstrap"]
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", RuntimeWarning)
        result = bootstrap((np.arange(len(inference.cal_idx)), np.arange(len(inference.eval_idx))), inference.statistic,
             n_resamples=b["replicates"], batch=b["batch"], paired=False, vectorized=True, axis=-1,
             confidence_level=b["confidence"], method=b["method"], rng=np.random.default_rng(b["seed"]))
    draws = result.bootstrap_distribution.reshape(len(points), len(METRICS), b["replicates"])
    cal_draws, eval_draws = np.concatenate(inference.cal_draws), np.concatenate(inference.eval_draws)
    require(len(cal_draws) == len(eval_draws) == b["replicates"], "Unexpected bootstrap draw coverage")
    for pi, point in enumerate(points):
        point["intervals"] = {m: interval(draws[pi, mi], point["metrics"][m], plan) for mi, m in enumerate(METRICS)}
    contrasts = build_contrasts(points, draws, plan)
    support = []
    for panel in ("calibrated", "matched"):
        for target in plan["targets"]:
            pi = [i for i, p in enumerate(points) if p["panel"] == panel and p["target"] == target]
            valid = np.isfinite(draws[pi]).all(axis=(0, 1))
            support.append({"panel": panel, "target": target, "valid_bootstraps": int(valid.sum()),
                            "total_bootstraps": b["replicates"],
                            "intervals_available": int(valid.sum()) >= b["minimum_supported_fraction"] * b["replicates"]})
    np.savez_compressed(run / "bootstrap-distribution.npz", values=draws, calibration_indices=cal_draws,
                        evaluation_indices=eval_draws, point_keys=np.asarray([p["key"] for p in points]), metrics=np.asarray(METRICS))
    environment = {"slurm_job_id": os.environ["SLURM_JOB_ID"], "utc": datetime.now(timezone.utc).isoformat(),
                   "hostname": socket.getfqdn(), "numpy_version": np.__version__, "scipy_version": importlib.metadata.version("scipy"),
                   "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]).decode().strip()}
    analysis = {"status": "complete", "metadata": dict(environment, calibration_queries=len(inference.cal_idx),
                evaluation_queries=len(inference.eval_idx), bootstrap_replicates=b["replicates"], bootstrap_seed=b["seed"], seed=b["seed"],
                method="two-independent-group query bootstrap, common indices across all cases; percentile paired differences",
                intervals="pointwise, unadjusted, conditional on explicitly reported common target support",
                new_asr_calls=0, new_encoder_calls=0, new_api_calls=0, bootstrap_warnings=[str(w.message) for w in recorded]),
                "plan": plan, "fixed_controls": [p for p in points if p["panel"] == "fixed"],
                "calibrated_points": [p for p in points if p["panel"] == "calibrated"],
                "matched_points": [p for p in points if p["panel"] == "matched"],
                "contrasts": contrasts, "support": support, "source_sha256": sources,
                "estimands": {"calibrated": "practice-selected policies frozen and applied to remaining dev; actual achieved recall may differ",
                              "matched": "descriptive evaluation-curve standardization at common macro recall; not deployable tuning",
                              "fixed": "unretuned top-5 and tau=0.8925 controls"}}
    write(run / "analysis.json", analysis)
    write(run / "analysis-completion.json", {"status": "complete", **environment,
          "bootstrap_replicates": b["replicates"], "points": len(points), "paired_contrasts": len(contrasts),
          "artifact_sha256": {name: sha(run / name) for name in ("analysis.json", "curves.csv", "calibration-policies.json", "bootstrap-distribution.npz")},
          "new_asr_calls": 0, "new_encoder_calls": 0, "new_api_calls": 0})
    print(json.dumps({"status": "complete", "points": len(points), "contrasts": len(contrasts), "support": support}), flush=True)


if __name__ == "__main__":
    main()
