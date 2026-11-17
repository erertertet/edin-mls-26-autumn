"""Instrument the supplied GLM generation path for numerical parity in GPU Slurm."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import numpy as np
import torch

SNAPSHOT = "61ba4e0b3309b6656edea3e93e419f7bd5c61957"
RUNS = (("none_1", "none"), ("task12", "task1,task2"),
        ("task123", "task1,task2,task3"), ("all_1", "all"),
        ("none_2", "none"), ("all_2", "all"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_trace(path, trace):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **trace)
    temporary.replace(path)


def cpu_array(tensor):
    require(tensor.is_cuda and tensor.dtype == torch.float32, "Expected CUDA float32 model activation")
    return tensor.detach().cpu().contiguous().numpy().copy()


def fingerprint(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def precision_state():
    return {"global": torch.backends.fp32_precision,
            "matmul": torch.backends.cuda.matmul.fp32_precision,
            "cudnn": torch.backends.cudnn.fp32_precision}


def metrics(reference, candidate):
    require(reference.shape == candidate.shape and reference.size > 0, "Comparable array shape mismatch")
    one, two = reference.astype(np.float64), candidate.astype(np.float64)
    require(np.isfinite(one).all() and np.isfinite(two).all(), "Nonfinite trace")
    difference = two - one
    reference_l2 = float(np.linalg.norm(one.reshape(-1)))
    difference_l2 = float(np.linalg.norm(difference.reshape(-1)))
    defined = reference_l2 > 0 or difference_l2 == 0
    relative = difference_l2 / reference_l2 if reference_l2 > 0 else (0.0 if difference_l2 == 0 else None)
    return {"elements": int(reference.size), "shape": list(reference.shape),
            "max_abs": float(np.max(np.abs(difference))), "mean_abs": float(np.mean(np.abs(difference))),
            "rms": float(np.sqrt(np.mean(difference * difference))), "relative_l2": relative,
            "reference_l2": reference_l2, "relative_l2_defined": defined}


def compare(reference_meta, candidate_meta, reference, candidate):
    left, right = reference_meta["token_ids"], candidate_meta["token_ids"]
    common = 0
    for one, two in zip(left, right):
        if one != two:
            break
        common += 1
    equal = left == right
    first_mismatch = None if equal else common
    # The first differing prediction still conditions on identical prior token IDs.
    comparable = min(len(left), len(right), common + 1)
    result = {"reference_run": reference_meta["run_id"], "candidate_run": candidate_meta["run_id"],
        "token_agreement": equal, "token_mismatch_count": sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right)),
        "first_token_mismatch_index": first_mismatch, "common_prefix_tokens": common,
        "comparable_prediction_steps": comparable, "full_trajectory_comparable": equal,
        "reference_steps": len(left), "candidate_steps": len(right),
        "compared_logit_values": comparable * reference_meta["vocab_size"],
        "transcript_exact_match": reference_meta["text"] == candidate_meta["text"],
        "argmax_mismatch_count_common_steps": int(np.count_nonzero(
            np.argmax(reference["probs"][:comparable], axis=1) != np.argmax(candidate["probs"][:comparable], axis=1))),
        "scope": "full shared-token-history trajectory" if equal else "common-input-token-prefix only; later branch undetermined",
        "audio_embeddings": metrics(reference["audio_embeddings"], candidate["audio_embeddings"]),
        "logits": metrics(reference["logits"][:comparable], candidate["logits"][:comparable]),
        "probs": metrics(reference["probs"][:comparable], candidate["probs"][:comparable]), "per_step": []}
    for step in range(comparable):
        result["per_step"].append({"step": step, "reference_token_id": left[step], "candidate_token_id": right[step],
            "token_match": left[step] == right[step],
            "logits": metrics(reference["logits"][step], candidate["logits"][step]),
            "probs": metrics(reference["probs"][step], candidate["probs"][step])})
    return result


def measure(args):
    require(torch.cuda.is_available(), "A Slurm GPU allocation is required")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    hw = args.root / "hw1-asr"
    sys.path.insert(0, str(hw))
    import benchmark
    from glm_asr_cuda import ops
    from glm_asr_cuda.model import MODEL_NAME, load_model
    from huggingface_hub import snapshot_download
    require(Path(benchmark.__file__).resolve() == (hw / "benchmark.py").resolve(), "Unexpected benchmark import")
    versions = {name: {"version": importlib.metadata.version(name), "path": importlib.util.find_spec(module).origin}
                for name, module in (("torch", "torch"), ("numpy", "numpy"), ("transformers", "transformers"),
                                     ("safetensors", "safetensors"), ("huggingface-hub", "huggingface_hub"))}
    require(versions["torch"]["version"] == "2.10.0+cu128" and versions["numpy"]["version"] == "2.4.6",
            "Numerical measurement environment changed")
    precision_before = precision_state()
    # Only the PyTorch 2.10 fp32_precision API is used; do not mix legacy switches.
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    precision_after = precision_state()
    require(set(precision_after.values()) == {"ieee"}, "IEEE float32 precision setting did not take effect")
    tiny = torch.tensor([[1., 2.], [3., 4.]], device="cuda", dtype=torch.float32)
    require(np.array_equal((tiny @ tiny).cpu().numpy(), np.array([[7., 10.], [15., 22.]], dtype=np.float32)),
            "Installed CUDA float32 matrix API probe failed")
    snapshot = Path(snapshot_download(MODEL_NAME, local_files_only=True))
    require(snapshot.name == SNAPSHOT, "Cached GLM snapshot does not match the pinned revision")
    library = ops.library_path().resolve()
    require(library.is_file(), "Compiled CUDA library is unavailable")
    saved_library = args.out / "libasr_kernels.so"
    shutil.copyfile(library, saved_library)
    require(sha(saved_library) == sha(library), "Saved/runtime CUDA library mismatch")
    source_paths = [Path(__file__).resolve(), hw / "benchmark.py", hw / "glm_asr_cuda/model.py",
                    hw / "glm_asr_cuda/ops.py", args.root / "report/asr-e2e-error-protocol.md",
                    hw / "run_asr_e2e_error.sbatch"]
    source_paths += sorted(p for p in (hw / "csrc").iterdir() if p.suffix in (".cu", ".cuh"))
    sources = {str(path.relative_to(args.root)): sha(path) for path in source_paths}
    model_hashes = {path.relative_to(snapshot).as_posix(): sha(path)
                    for path in sorted(snapshot.rglob("*")) if path.is_file()}
    require("model.safetensors" in model_hashes, "Pinned checkpoint has no model.safetensors")
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    device = {"index": device_index, "name": props.name, "major": props.major, "minor": props.minor,
              "total_memory": props.total_memory, "multi_processor_count": props.multi_processor_count,
              "uuid": str(getattr(props, "uuid", "unavailable")), "torch_cuda_version": torch.version.cuda}
    ops.set_cuda_groups("none")
    model, processor = load_model(device="cuda")
    require(model.c.vocab == 59264 and model.w
            and all(t.is_cuda and t.dtype == torch.float32 and t.is_contiguous() for t in model.w.values()),
            "Expected one loaded model with contiguous float32 CUDA weights")
    audio_path = hw / "test_audio.wav"
    audio = benchmark.read_wav(audio_path)
    inputs = processor.apply_transcription_request(audio)
    features = inputs["input_features"].to("cuda", torch.float32)
    input_ids = inputs["input_ids"].to("cuda")
    features_mask = inputs["input_features_mask"].to("cuda")
    tensors = {"features": features, "input_ids": input_ids, "features_mask": features_mask}
    input_fingerprints = {name: fingerprint(tensor) for name, tensor in tensors.items()}
    config = {"version": "asr-e2e-numerical-v1", "run_plan": [{"run_id": name, "cuda_spec": spec} for name, spec in RUNS],
        "reference": "PyTorch ops.set_cuda_groups('none'); not a ground-truth transcript", "max_new_tokens": 100,
        "source_sha256": sources, "library": {"runtime_path": str(library), "saved_file": saved_library.name, "sha256": sha(library)},
        "model": {"name": MODEL_NAME, "snapshot_revision": SNAPSHOT, "snapshot_path": str(snapshot),
                  "source_snapshot_path": str(Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL_NAME.replace("/", "--")) / "snapshots" / SNAPSHOT),
                  "files_sha256": model_hashes,
                  "weight_dtype": "torch.float32", "weight_tensors": len(model.w), "weight_elements": sum(t.numel() for t in model.w.values())},
        "audio": {"path": str(audio_path), "sha256": sha(audio_path), "sample_rate": 16000, "samples": len(audio)},
        "input_fingerprints": input_fingerprints, "precision": {"before": precision_before, "after": precision_after},
        "precision_context": "New IEEE float32 numerical-validation context, separate from previous timing measurements",
        "versions": versions, "device": device, "offline": True,
        "metric_definition": "float64 CPU candidate-reference; max/mean absolute, RMS, L2(diff)/L2(reference), no epsilon",
        "zero_reference_convention": "relative L2=0 if both reference and difference have zero norm; otherwise undefined/null",
        "trajectory_alignment": "Compare through first differing prediction inclusive; later token-conditioned branches are undetermined",
        "numerical_tolerance": None, "timing_claim": False}
    config_path = args.out / "config.json"
    if config_path.exists():
        require(json.loads(config_path.read_text()) == config, "Frozen measurement configuration changed")
    else:
        write(config_path, config)
    traces, run_records = {}, []
    for run_id, spec in RUNS:
        ops.set_cuda_groups(spec)
        logits, probabilities, embeddings = [], [], []
        original_softmax, original_encode = ops.softmax, model.encode_audio
        instance_encode = model.__dict__.get("encode_audio")
        had_instance_encode = "encode_audio" in model.__dict__

        def traced_softmax(value):
            capture = value.ndim == 2 and tuple(value.shape) == (1, model.c.vocab)
            if capture:
                logits.append(cpu_array(value)[0])
            result = original_softmax(value)
            if capture:
                probabilities.append(cpu_array(result)[0])
            return result

        def traced_encode(*values, **keywords):
            result = original_encode(*values, **keywords)
            embeddings.append(cpu_array(result))
            return result

        try:
            ops.softmax, model.encode_audio = traced_softmax, traced_encode
            tokens, chosen = model.generate(features, input_ids, features_mask, max_new_tokens=100)
            torch.cuda.synchronize()
        finally:
            ops.softmax = original_softmax
            if had_instance_encode:
                model.encode_audio = instance_encode
            else:
                del model.encode_audio
        require(len(tokens) == len(logits) == len(probabilities) > 0 and len(embeddings) == 1,
                "Generation/hook coverage mismatch")
        trace = {"logits": np.stack(logits), "probs": np.stack(probabilities), "audio_embeddings": embeddings[0],
                 "token_ids": np.asarray(tokens, dtype=np.int64), "chosen_probabilities": np.asarray(chosen, dtype=np.float64)}
        require(all(np.isfinite(value).all() for value in trace.values()), "Nonfinite captured trace")
        require(trace["logits"].shape == trace["probs"].shape == (len(tokens), model.c.vocab)
                and np.array_equal(np.argmax(trace["probs"], axis=1), trace["token_ids"])
                and np.array_equal(trace["probs"][np.arange(len(tokens)), trace["token_ids"]].astype(np.float64), trace["chosen_probabilities"]),
                "Greedy token/probability capture mismatch")
        require({name: fingerprint(tensor) for name, tensor in tensors.items()} == input_fingerprints,
                "The reused generation inputs changed")
        text = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        trace_file = Path("traces") / (run_id + ".npz")
        save_trace(args.out / trace_file, trace)
        mass = trace["probs"].astype(np.float64).sum(axis=1)
        record = {"run_id": run_id, "cuda_spec": spec, "cuda_groups": sorted(ops.USE_CUDA), "backend_description": ops.describe(),
                  "token_ids": [int(t) for t in tokens], "text": text, "steps": len(tokens), "vocab_size": model.c.vocab,
                  "trace_file": trace_file.as_posix(), "trace_sha256": sha(args.out / trace_file),
                  "trace_shapes": {name: list(value.shape) for name, value in trace.items()},
                  "probability_mass_max_abs_error": float(np.max(np.abs(mass - 1))),
                  "probability_mass_min": float(mass.min()), "probability_mass_max": float(mass.max()),
                  "probability_min": float(trace["probs"].min()), "probability_max": float(trace["probs"].max()),
                  "logit_probability_argmax_disagreements": int(np.count_nonzero(np.argmax(trace["logits"], axis=1) != trace["token_ids"]))}
        traces[run_id] = trace
        run_records.append(record)
        write(args.out / "runs.json", {"status": "partial", "runs": run_records})
        print(json.dumps({"stage": "captured", "run_id": run_id, "steps": len(tokens), "vocab_size": model.c.vocab}), flush=True)
    require(sha(library) == config["library"]["sha256"]
            and all(sha(args.root / name) == expected for name, expected in sources.items()), "Library or source changed during measurement")
    by_id = {value["run_id"]: value for value in run_records}
    def difference(reference, candidate):
        return compare(by_id[reference], by_id[candidate], traces[reference], traces[candidate])
    metadata = {"utc": utc(), "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.getfqdn(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.root).decode().strip(),
        "config_sha256": sha(config_path), "device": device, "versions": versions, "precision": config["precision"],
        "model_loads": 1, "processor_requests": 1, "numerical_tolerance": None, "timing_claim": False,
        "new_api_calls": 0, "new_external_asr_calls": 0, "new_rag_encoder_calls": 0,
        "model_generate_calls": len(run_records), "audio_encoder_calls": len(run_records)}
    write(args.out / "runs.json", {"status": "complete", "runs": run_records})
    write(args.out / "summary.json", {"status": "complete", "metadata": metadata, "run_ids": [name for name, _ in RUNS],
        "comparisons": [difference("none_1", candidate) for candidate in ("task12", "task123", "all_1")],
        "self_drift": [difference("none_1", "none_2"), difference("all_1", "all_2")]})
    artifacts = ["config.json", "runs.json", "summary.json", saved_library.name] + [value["trace_file"] for value in run_records]
    write(args.out / "completion.json", {"status": "complete", "slurm_job_id": os.environ["SLURM_JOB_ID"], "utc": utc(),
        "runs": len(run_records), "comparisons": 3, "self_drift_comparisons": 2, "model_loads": 1, "new_api_calls": 0,
        "artifact_sha256": {name: sha(args.out / name) for name in artifacts}})
    print(json.dumps({"status": "complete", "runs": len(run_records), "numerical_tolerance": None}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"), "Run only inside Edinburgh Slurm")
    os.umask(0o077)
    args.root, args.out = args.root.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    require(not (args.out / "completion.json").exists(), "Use a fresh directory to preserve an existing measurement")
    try:
        measure(args)
    except Exception as error:
        write(args.out / "completion.json", {"status": "fail", "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "utc": utc(), "errors": [{"type": type(error).__name__, "message": str(error)}]})
        raise


if __name__ == "__main__":
    main()
