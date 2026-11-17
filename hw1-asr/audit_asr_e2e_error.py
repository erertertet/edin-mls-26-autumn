"""Recount saved end-to-end ASR numerical traces on CPU Slurm; no model/API calls."""
from __future__ import annotations

import argparse
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

RUNS = (("none_1", "none"), ("task12", "task1,task2"), ("task123", "task1,task2,task3"),
        ("all_1", "all"), ("none_2", "none"), ("all_2", "all"))
COMPARISONS = (("none_1", "task12"), ("none_1", "task123"), ("none_1", "all_1"))
SELF_DRIFT = (("none_1", "none_2"), ("all_1", "all_2"))
TRACE_FIELDS = ("logits", "probs", "audio_embeddings", "token_ids", "chosen_probabilities")
SNAPSHOT = "61ba4e0b3309b6656edea3e93e419f7bd5c61957"


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


def nonnegative_close(actual, expected, label):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and actual >= 0, "Invalid finite nonnegative metric: " + label)
    if expected == 0:
        require(actual == 0, "Expected exact zero from identical arrays: " + label)
    else:
        require(actual > 0 and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-300),
                "Saved-array metric mismatch: " + label)


def recount(reference, candidate):
    require(reference.shape == candidate.shape and reference.size > 0, "Comparison shape/coverage mismatch")
    left, right = reference.astype(np.float64), candidate.astype(np.float64)
    require(np.isfinite(left).all() and np.isfinite(right).all(), "Nonfinite comparison trace")
    difference = (right - left).reshape(-1)
    ref_flat = left.reshape(-1)
    # Use correctly rounded scalar summation, independent of producer's NumPy mean/BLAS norms.
    squared_difference = math.fsum(difference * difference)
    squared_reference = math.fsum(ref_flat * ref_flat)
    difference_norm, reference_norm = math.sqrt(squared_difference), math.sqrt(squared_reference)
    relative = difference_norm / reference_norm if reference_norm else (0.0 if not difference_norm else None)
    return {"elements": int(reference.size), "shape": list(reference.shape),
            "max_abs": float(np.max(np.abs(difference))),
            "mean_abs": math.fsum(np.abs(difference)) / reference.size,
            "rms": math.sqrt(squared_difference / reference.size), "reference_l2": reference_norm,
            "relative_l2": relative, "relative_l2_defined": relative is not None}


def check_metrics(actual, expected, label):
    require(set(actual) == set(expected) and actual["elements"] == expected["elements"]
            and actual["shape"] == expected["shape"]
            and actual["relative_l2_defined"] == expected["relative_l2_defined"], "Metric schema mismatch: " + label)
    require(actual["max_abs"] == expected["max_abs"], "Exact maximum-absolute discrepancy mismatch: " + label)
    for name in ("max_abs", "mean_abs", "rms", "reference_l2"):
        nonnegative_close(actual[name], expected[name], label + "/" + name)
    if expected["relative_l2"] is None:
        require(actual["relative_l2"] is None, "Undefined zero-reference relative L2 was replaced: " + label)
    else:
        nonnegative_close(actual["relative_l2"], expected["relative_l2"], label + "/relative_l2")


def groups_for(spec):
    if spec == "none":
        return []
    if spec == "all":
        return sorted(("rope", "reduce", "gemm", "attention", "provided"))
    aliases = {"task1": "rope", "task2": "reduce", "task3": "gemm", "task4": "attention"}
    return sorted({"provided", *(aliases[part] for part in spec.split(","))})


def audit(args):
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"), "Run only in Edinburgh Slurm")
    run = args.run
    config, runs_file, summary, completion = (read(run / name) for name in ("config.json", "runs.json", "summary.json", "completion.json"))
    require(config["version"] == "asr-e2e-numerical-v1" and config["run_plan"]
            == [{"run_id": name, "cuda_spec": spec} for name, spec in RUNS], "Frozen run plan changed")
    require(runs_file["status"] == summary["status"] == completion["status"] == "complete"
            and summary["run_ids"] == [name for name, _ in RUNS], "Incomplete six-run numerical measurement")
    require(config["max_new_tokens"] == 100 and config["offline"] is True
            and config["numerical_tolerance"] is None and config["timing_claim"] is False,
            "Numerical measurement context changed")
    require(config["precision"]["after"] == {"global": "ieee", "matmul": "ieee", "cudnn": "ieee"},
            "PyTorch IEEE float32 settings were not recorded")
    for relative, expected in config["source_sha256"].items():
        require(sha(args.root / relative) == expected, "Frozen numerical-measurement source changed: " + relative)
    require("hw1-asr/measure_asr_e2e_error.py" in config["source_sha256"]
            and "hw1-asr/glm_asr_cuda/model.py" in config["source_sha256"]
            and "hw1-asr/glm_asr_cuda/ops.py" in config["source_sha256"], "Original model/operation source binding missing")
    library = run / config["library"]["saved_file"]
    require(library.name == "libasr_kernels.so" and sha(library) == config["library"]["sha256"], "Saved CUDA library changed")
    for relative, expected in completion["artifact_sha256"].items():
        path = (run / relative).resolve()
        path.relative_to(run.resolve())
        require(sha(path) == expected, "Completed measurement artifact changed: " + relative)
    model = config["model"]
    require(model["name"] == "zai-org/GLM-ASR-Nano-2512" and model["snapshot_revision"] == SNAPSHOT
            and model["weight_dtype"] == "torch.float32" and model["weight_tensors"] > 0 and model["weight_elements"] > 0,
            "Unexpected model checkpoint/dtype declaration")
    snapshot = Path(model.get("source_snapshot_path", model["snapshot_path"]))
    if not snapshot.is_dir():
        snapshot = Path.home() / ".cache/huggingface/hub" / ("models--" + model["name"].replace("/", "--")) / "snapshots" / SNAPSHOT
    require(snapshot.is_dir() and snapshot.name == SNAPSHOT and "model.safetensors" in model["files_sha256"],
            "Persistent checkpoint snapshot unavailable")
    for relative, expected in model["files_sha256"].items():
        path = snapshot / relative
        require(sha(path) == expected, "Loaded checkpoint source file changed")
    audio_path = Path(config["audio"]["path"])
    require(audio_path.resolve() == (args.root / "hw1-asr/test_audio.wav").resolve()
            and sha(audio_path) == config["audio"]["sha256"], "Single-clip audio fixture changed")
    with wave.open(str(audio_path), "rb") as wav:
        require(wav.getsampwidth() == 2 and wav.getframerate() == config["audio"]["sample_rate"] == 16000
                and wav.getnframes() == config["audio"]["samples"], "Audio fixture properties changed")
        audio_seconds = wav.getnframes() / wav.getframerate()
    require(audio_seconds > 0, "Empty single-clip fixture")
    require(set(config["input_fingerprints"]) == {"features", "input_ids", "features_mask"}, "Prepared-input fingerprint coverage missing")
    for fingerprint in config["input_fingerprints"].values():
        require(fingerprint["shape"] and all(type(n) is int and n > 0 for n in fingerprint["shape"])
                and isinstance(fingerprint["dtype"], str) and len(fingerprint["sha256"]) == 64,
                "Invalid prepared-input fingerprint declaration")
    metadata = summary["metadata"]
    require(metadata["config_sha256"] == sha(run / "config.json") and metadata["device"] == config["device"]
            and metadata["versions"] == config["versions"] and metadata["precision"] == config["precision"],
            "Summary configuration/environment mismatch")
    require(metadata["slurm_job_id"] == completion["slurm_job_id"] and metadata["model_loads"] == completion["model_loads"] == 1
            and metadata["processor_requests"] == 1 and metadata["model_generate_calls"] == metadata["audio_encoder_calls"] == 6
            and metadata["new_api_calls"] == metadata["new_external_asr_calls"] == metadata["new_rag_encoder_calls"] == 0
            and metadata["numerical_tolerance"] is None and metadata["timing_claim"] is False,
            "Inference/precision/API scope accounting mismatch")
    require(completion["runs"] == 6 and completion["comparisons"] == 3 and completion["self_drift_comparisons"] == 2
            and completion["new_api_calls"] == 0, "Completion run accounting mismatch")
    require(config["versions"]["numpy"]["version"] == np.__version__ == "2.4.6"
            and config["versions"]["torch"]["version"] == "2.10.0+cu128", "Recorded numerical environment changed")

    records = {item["run_id"]: item for item in runs_file["runs"]}
    require(len(records) == len(runs_file["runs"]) == 6 and list(records) == [name for name, _ in RUNS], "Duplicate or missing run records")
    traces, probability_checks = {}, []
    total_rows = 0
    for run_id, spec in RUNS:
        item = records[run_id]
        require(item["cuda_spec"] == spec and item["cuda_groups"] == groups_for(spec), "Kernel-group declaration mismatch")
        description = "  ".join(group + "=" + ("cuda" if group in groups_for(spec) else "torch")
                                 for group in ("rope", "reduce", "gemm", "attention", "provided"))
        require(item["backend_description"] == description, "Kernel backend description mismatch")
        path = run / item["trace_file"]
        require(path.resolve().parent == (run / "traces").resolve() and sha(path) == item["trace_sha256"]
                == completion["artifact_sha256"][item["trace_file"]], "Saved trace hash/path mismatch")
        with np.load(path, allow_pickle=False) as archive:
            trace = {name: archive[name] for name in archive.files}
        require(set(trace) == set(TRACE_FIELDS) and {name: list(value.shape) for name, value in trace.items()} == item["trace_shapes"],
                "Trace keys or recorded shapes mismatch")
        steps, vocab = item["steps"], item["vocab_size"]
        require(type(steps) is int and 1 <= steps <= config["max_new_tokens"] and vocab == 59264
                and trace["logits"].shape == trace["probs"].shape == (steps, vocab)
                and trace["logits"].dtype == trace["probs"].dtype == np.float32
                and trace["audio_embeddings"].ndim == 2 and trace["audio_embeddings"].shape[0] > 0
                and trace["audio_embeddings"].shape[1] == 2048 and trace["audio_embeddings"].dtype == np.float32
                and trace["token_ids"].shape == trace["chosen_probabilities"].shape == (steps,)
                and trace["token_ids"].dtype == np.int64 and trace["chosen_probabilities"].dtype == np.float64,
                "Full-vocabulary/audio/token trace schema mismatch")
        require(all(np.isfinite(value).all() for value in trace.values())
                and np.all((trace["token_ids"] >= 0) & (trace["token_ids"] < vocab))
                and np.all((trace["probs"] >= 0) & (trace["probs"] <= 1)), "Invalid finite token/probability trace")
        require(trace["token_ids"].tolist() == item["token_ids"] and isinstance(item["text"], str), "Recorded token/text metadata mismatch")
        argmax = np.argmax(trace["probs"], axis=1)
        require(np.array_equal(argmax, trace["token_ids"])
                and np.array_equal(trace["probs"][np.arange(steps), trace["token_ids"]].astype(np.float64), trace["chosen_probabilities"]),
                "Greedy token or chosen-probability capture mismatch")
        mass = trace["probs"].astype(np.float64).sum(axis=1)
        max_mass_error = float(np.max(np.abs(mass - 1)))
        diagnostics = {"probability_mass_max_abs_error": max_mass_error, "probability_mass_min": float(mass.min()),
                       "probability_mass_max": float(mass.max()), "probability_min": float(trace["probs"].min()),
                       "probability_max": float(trace["probs"].max())}
        for name, expected in diagnostics.items():
            nonnegative_close(item[name], expected, run_id + "/" + name)
        disagreements = int(np.count_nonzero(np.argmax(trace["logits"], axis=1) != trace["token_ids"]))
        require(item["logit_probability_argmax_disagreements"] == disagreements, "Logit/probability argmax diagnostic mismatch")
        probability_checks.append({"run_id": run_id, "steps": steps, "vocab_size": vocab,
                                  "probability_mass_max_abs_error": max_mass_error,
                                  "logit_probability_argmax_disagreements": disagreements})
        traces[run_id] = trace
        total_rows += steps

    require(len(summary["comparisons"]) == 3 and len(summary["self_drift"]) == 2, "Comparison coverage mismatch")
    comparison_reports, metric_blocks, per_steps = [], 0, 0
    for saved, (reference_id, candidate_id) in zip(summary["comparisons"] + summary["self_drift"], COMPARISONS + SELF_DRIFT):
        reference, candidate = traces[reference_id], traces[candidate_id]
        left, right = reference["token_ids"].tolist(), candidate["token_ids"].tolist()
        prefix = 0
        for one, two in zip(left, right):
            if one != two:
                break
            prefix += 1
        equal = left == right
        comparable = min(len(left), len(right), prefix + 1)
        expected = {"reference_run": reference_id, "candidate_run": candidate_id,
                    "token_agreement": equal, "token_mismatch_count": sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right)),
                    "first_token_mismatch_index": None if equal else prefix, "common_prefix_tokens": prefix,
                    "comparable_prediction_steps": comparable, "full_trajectory_comparable": equal,
                    "reference_steps": len(left), "candidate_steps": len(right), "compared_logit_values": comparable * 59264,
                    "transcript_exact_match": records[reference_id]["text"] == records[candidate_id]["text"],
                    "argmax_mismatch_count_common_steps": int(np.count_nonzero(
                        np.argmax(reference["probs"][:comparable], axis=1) != np.argmax(candidate["probs"][:comparable], axis=1))),
                    "scope": "full shared-token-history trajectory" if equal else "common-input-token-prefix only; later branch undetermined"}
        require(all(saved[name] == value for name, value in expected.items()), "Token-history alignment/accounting mismatch")
        for field in ("audio_embeddings", "logits", "probs"):
            one, two = reference[field], candidate[field]
            if field != "audio_embeddings":
                one, two = one[:comparable], two[:comparable]
            check_metrics(saved[field], recount(one, two), reference_id + "/" + candidate_id + "/" + field)
            metric_blocks += 1
        require(len(saved["per_step"]) == comparable, "Aligned per-step metric coverage mismatch")
        for step, actual in enumerate(saved["per_step"]):
            require(actual["step"] == step and actual["reference_token_id"] == left[step]
                    and actual["candidate_token_id"] == right[step] and actual["token_match"] == (left[step] == right[step]),
                    "Per-step token metadata mismatch")
            for field in ("logits", "probs"):
                check_metrics(actual[field], recount(reference[field][step], candidate[field][step]), "per-step/" + field)
                metric_blocks += 1
            per_steps += 1
        comparison_reports.append({name: expected[name] for name in ("reference_run", "candidate_run", "token_agreement",
            "token_mismatch_count", "common_prefix_tokens", "comparable_prediction_steps", "full_trajectory_comparable", "compared_logit_values")})
    return {"status": "pass", "errors": [], "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "analysis_slurm_job_id": completion["slurm_job_id"], "utc": datetime.now(timezone.utc).isoformat(),
            "audio_clips": 1, "audio_seconds": audio_seconds, "runs_checked": 6,
            "comparisons_checked": 3, "self_drift_comparisons_checked": 2, "per_step_comparisons_checked": per_steps,
            "metric_blocks_independently_recomputed": metric_blocks, "full_vocabulary_size": 59264,
            "captured_prediction_rows_checked": total_rows, "captured_logit_values_checked": total_rows * 59264,
            "captured_probability_values_checked": total_rows * 59264,
            "comparison_coverage": comparison_reports, "probability_diagnostics": probability_checks,
            "source_files_sha256_checked": len(config["source_sha256"]), "model_snapshot_files_sha256_checked": len(model["files_sha256"]),
            "library_sha256": config["library"]["sha256"], "model_snapshot_revision": SNAPSHOT,
            "metric_recomputation_relative_tolerance": 1e-10, "kernel_numerical_acceptance_tolerance": None,
            "numpy_version": np.__version__, "audit_script_sha256": sha(__file__),
            "audit_new_model_calls": 0, "audit_new_encoder_calls": 0, "audit_new_api_calls": 0,
            "production_model_generate_calls": metadata["model_generate_calls"], "production_audio_encoder_calls": metadata["audio_encoder_calls"],
            "input_sha256": {name: sha(run / name) for name in ("config.json", "runs.json", "summary.json", "completion.json", library.name)},
            "scope": "Saved-array numerical/accounting audit on one supplied clip; Python float64 sums/norm formulas, greedy capture and common-history alignment. Does not rerun the model, decode tokens independently, infer WER, or impose a kernel acceptance threshold."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"), "Run only in Edinburgh Slurm")
    os.umask(0o077)
    try:
        result = audit(args)
    except Exception as error:
        result = {"status": "fail", "slurm_job_id": os.environ["SLURM_JOB_ID"], "audit_script_sha256": sha(__file__),
                  "errors": [{"type": type(error).__name__, "message": str(error)}]}
        (args.run / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise
    (args.run / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
