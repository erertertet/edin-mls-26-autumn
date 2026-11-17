"""All-audio CUDA/PyTorch numerical parity with complete reference-token histories."""
from __future__ import annotations
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

from measure_asr_e2e_error import SNAPSHOT, cpu_array, fingerprint, metrics, precision_state, require, sha, write, save_trace, utc

LIBRARY_SHA256 = "76817bd7fef1878069515d12eb4156f76368c21e12076a00ec450c1a7c78da8a"


def error_stats(reference, candidate):
    result = metrics(reference, candidate)
    one, two = reference.astype(np.float64), candidate.astype(np.float64)
    difference = two - one
    result.update(sum_abs=float(np.abs(difference).sum(dtype=np.float64)),
                  sum_sq=float((difference * difference).sum(dtype=np.float64)),
                  sum_reference_sq=float((one * one).sum(dtype=np.float64)))
    return result


def probability_diagnostics(values):
    mass = values.astype(np.float64).sum(axis=1)
    return {"minimum": float(values.min()), "maximum": float(values.max()),
            "row_sum_min": float(mass.min()), "row_sum_max": float(mass.max()),
            "row_sum_max_abs_error": float(np.max(np.abs(mass - 1)))}


def verify_trace(trace, vocab):
    logits, probs, ids = trace["logits"], trace["probs"], trace["token_ids"]
    require(logits.dtype == probs.dtype == np.float32 and logits.shape == probs.shape == (len(ids), vocab)
            and len(ids) > 0 and ids.dtype == np.int64, "Invalid generated trace shape/dtype")
    require(all(np.isfinite(value).all() for value in trace.values()), "Nonfinite generated trace")
    require(np.array_equal(np.argmax(probs, axis=1), ids)
            and np.array_equal(probs[np.arange(len(ids)), ids].astype(np.float64), trace["chosen_probabilities"]),
            "Generated tokens/chosen probabilities do not match captured probabilities")


def capture_generate(model, processor, ops, tensors, mode, cap, counters, retain_audio=False):
    ops.set_cuda_groups(mode)
    original_softmax, original_encode = ops.softmax, model.encode_audio
    had_instance = "encode_audio" in model.__dict__
    instance_encode = model.__dict__.get("encode_audio")
    logits, probs, audio = [], [], []
    gpu_audio = []

    def traced_softmax(value):
        capture = value.ndim == 2 and tuple(value.shape) == (1, model.c.vocab)
        if capture:
            logits.append(cpu_array(value)[0])
        result = original_softmax(value)
        if capture:
            probs.append(cpu_array(result)[0])
        return result

    def traced_encode(*values, **keywords):
        counters["audio_encoder_calls"] += 1
        result = original_encode(*values, **keywords)
        audio.append(cpu_array(result))
        if retain_audio:
            gpu_audio.append(result.detach())
        return result

    try:
        ops.softmax, model.encode_audio = traced_softmax, traced_encode
        counters["model_generate_calls"] += 1
        tokens, chosen = model.generate(tensors["features"], tensors["input_ids"], tensors["features_mask"], max_new_tokens=cap)
        torch.cuda.synchronize()
    finally:
        ops.softmax = original_softmax
        if had_instance:
            model.encode_audio = instance_encode
        else:
            del model.encode_audio
    require(len(logits) == len(probs) == len(tokens) > 0 and len(audio) == 1, "Generation hook coverage mismatch")
    trace = {"logits": np.stack(logits), "probs": np.stack(probs), "audio_embeddings": audio[0],
             "token_ids": np.asarray(tokens, dtype=np.int64), "chosen_probabilities": np.asarray(chosen, dtype=np.float64)}
    verify_trace(trace, model.c.vocab)
    text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
    eos = int(tokens[-1]) in model.c.eos_token_ids
    require(eos or len(tokens) == cap, "Unexpected non-EOS termination before the configured cap")
    record = {"mode": mode, "token_ids": [int(value) for value in tokens], "chosen_probabilities": [float(value) for value in chosen],
              "text_raw": text, "text_stripped": text.strip(), "steps": len(tokens), "eos_reached": eos,
              "at_cap": len(tokens) == cap, "cap_without_eos": len(tokens) == cap and not eos,
              "terminal_token_id": int(tokens[-1])}
    return trace, record, gpu_audio[0] if retain_audio else None


@torch.no_grad()
def replay_tail(model, ops, input_ids, cuda_audio, reference_ids, start, counters):
    """Use the original decoder on CUDA audio and only the reference token history."""
    ops.set_cuda_groups("all")
    table = model.w["language_model.model.embed_tokens.weight"]
    embeds = ops.embedding(input_ids, table)
    positions = (input_ids[0] == model.c.audio_token_id).nonzero().flatten()
    require(len(positions) > 0, "No audio placeholders in the supplied processor prompt")
    first, last = int(positions[0]), int(positions[-1])
    embeds = torch.cat([embeds[:, :first], cuda_audio[None], embeds[:, last + 1:]], dim=1)
    original_softmax = ops.softmax
    logits, probs = [], []

    def traced_softmax(value):
        capture = value.ndim == 2 and tuple(value.shape) == (1, model.c.vocab)
        if capture:
            logits.append(cpu_array(value)[0])
        result = original_softmax(value)
        if capture:
            probs.append(cpu_array(result)[0])
        return result

    try:
        ops.softmax = traced_softmax
        for step, reference_id in enumerate(reference_ids):
            if step >= start:
                counters["replay_next_token_calls"] += 1
                model.next_token_probs(embeds)
            if step + 1 < len(reference_ids):
                # No greedy CUDA choice enters this history, including after divergence.
                token = torch.tensor([[reference_id]], device=input_ids.device, dtype=input_ids.dtype)
                embeds = torch.cat([embeds, ops.embedding(token, table)], dim=1)
        torch.cuda.synchronize()
    finally:
        ops.softmax = original_softmax
    require(len(logits) == len(probs) == len(reference_ids) - start > 0, "Reference-history replay coverage mismatch")
    result = {"logits": np.stack(logits), "probs": np.stack(probs)}
    require(all(value.dtype == np.float32 and np.isfinite(value).all() for value in result.values()), "Invalid replay trace")
    return result


def process_query(query, args, config, config_sha, model, processor, ops, read_audio, counters):
    audio_path = Path(config["staging"]["data_dir"]) / query["audio_file"]
    require(sha(audio_path) == query["audio_sha256"], "Staged audio hash mismatch")
    audio = read_audio(audio_path)
    require(abs(len(audio) / 16000 - query["seconds"]) <= 1e-9, "Staged audio duration mismatch")
    counters["processor_requests"] += 1
    prepared = processor.apply_transcription_request(audio)
    tensors = {"features": prepared["input_features"].to("cuda", torch.float32),
               "input_ids": prepared["input_ids"].to("cuda"),
               "features_mask": prepared["input_features_mask"].to("cuda")}
    fingerprints = {name: fingerprint(value) for name, value in tensors.items()}
    reference, ref_record, _ = capture_generate(model, processor, ops, tensors, "none", config["plan"]["max_new_tokens"], counters)
    cuda, cuda_record, cuda_audio = capture_generate(model, processor, ops, tensors, "all", config["plan"]["max_new_tokens"], counters, True)
    left, right = ref_record["token_ids"], cuda_record["token_ids"]
    prefix = 0
    for one, two in zip(left, right):
        if one != two:
            break
        prefix += 1
    equal = left == right
    common_steps = min(len(left), len(right), prefix + 1)
    is_control = query["query_id"] in config["plan"]["replay_control_ids"]
    start = 0 if is_control else (common_steps if common_steps < len(left) else len(left))
    tail = replay_tail(model, ops, tensors["input_ids"], cuda_audio, left, start, counters) if start < len(left) else None
    full_logits = np.concatenate([cuda["logits"][:start], tail["logits"]], axis=0) if tail is not None else cuda["logits"][:len(left)]
    full_probs = np.concatenate([cuda["probs"][:start], tail["probs"]], axis=0) if tail is not None else cuda["probs"][:len(left)]
    require(full_logits.shape == reference["logits"].shape and full_probs.shape == reference["probs"].shape,
            "Incomplete full reference-history trace")
    require({name: fingerprint(value) for name, value in tensors.items()} == fingerprints, "Shared input tensors changed")
    control = None
    if is_control:
        logits_equal = np.array_equal(full_logits[:common_steps], cuda["logits"][:common_steps])
        probs_equal = np.array_equal(full_probs[:common_steps], cuda["probs"][:common_steps])
        control = {"enabled": True, "compared_steps": common_steps,
                   "logits_max_abs": float(np.max(np.abs(full_logits[:common_steps].astype(np.float64) - cuda["logits"][:common_steps]))),
                   "probs_max_abs": float(np.max(np.abs(full_probs[:common_steps].astype(np.float64) - cuda["probs"][:common_steps]))),
                   "logits_exact_equal": logits_equal, "probs_exact_equal": probs_equal}
    arrays = {"reference_logits": reference["logits"], "reference_probs": reference["probs"],
              "cuda_greedy_logits": cuda["logits"], "cuda_greedy_probs": cuda["probs"],
              "reference_audio_embeddings": reference["audio_embeddings"], "cuda_audio_embeddings": cuda["audio_embeddings"],
              "reference_token_ids": reference["token_ids"], "cuda_token_ids": cuda["token_ids"],
              "reference_chosen_probabilities": reference["chosen_probabilities"], "cuda_chosen_probabilities": cuda["chosen_probabilities"]}
    if tail is not None:
        arrays.update(cuda_reference_tail_logits=tail["logits"], cuda_reference_tail_probs=tail["probs"])
    import hashlib
    stem = hashlib.sha256(query["query_id"].encode()).hexdigest()[:20]
    trace_file = Path("traces") / (stem + ".npz")
    save_trace(args.run / trace_file, arrays)
    argmax = np.argmax(full_probs, axis=1)
    record = {"status": "complete", **query, "shard": args.shard, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "config_sha256": config_sha, "trace_file": trace_file.as_posix(), "trace_sha256": sha(args.run / trace_file),
        "input_fingerprints": fingerprints, "trace_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "reference": ref_record, "cuda": cuda_record,
        "greedy": {"token_sequence_equal": equal, "token_mismatch_count": sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right)),
                   "first_token_mismatch_index": None if equal else prefix, "common_prefix_tokens": prefix,
                   "comparable_prediction_steps": common_steps, "reference_steps": len(left), "cuda_steps": len(right),
                   "text_raw_equal": ref_record["text_raw"] == cuda_record["text_raw"],
                   "text_stripped_equal": ref_record["text_stripped"] == cuda_record["text_stripped"]},
        "reference_history": {"tail_start": start, "tail_rows": 0 if tail is None else len(tail["logits"]),
            "replay_next_token_calls": 0 if tail is None else len(tail["logits"]), "coverage_steps": len(full_probs),
            "coverage_complete": len(full_probs) == len(left), "argmax_ids": argmax.tolist(),
            "argmax_token_mismatch_count": int(np.count_nonzero(argmax != reference["token_ids"])),
            "argmax_all_equal": bool(np.array_equal(argmax, reference["token_ids"])),
            "used_cuda_audio_embeddings": True, "replay_control": control},
        "stats": {"logits": error_stats(reference["logits"], full_logits), "probs": error_stats(reference["probs"], full_probs),
                  "audio_embeddings": error_stats(reference["audio_embeddings"], cuda["audio_embeddings"])},
        "probability_diagnostics": {"reference": probability_diagnostics(reference["probs"]),
            "cuda_greedy": probability_diagnostics(cuda["probs"]), "cuda_reference_history": probability_diagnostics(full_probs)},
        "calls": {"model_generate_calls": 2, "audio_encoder_calls": 2,
                  "replay_next_token_calls": 0 if tail is None else len(tail["logits"])}, "utc": utc()}
    query_path = args.run / "queries" / (stem + ".json")
    query_path.parent.mkdir(parents=True, exist_ok=True)
    write(query_path, record)
    return record


def load_completed(query, args, config_sha):
    import hashlib
    path = args.run / "queries" / (hashlib.sha256(query["query_id"].encode()).hexdigest()[:20] + ".json")
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    require(value["status"] == "complete" and value["config_sha256"] == config_sha
            and all(value.get(name) == expected for name, expected in query.items())
            and value["shard"] == args.shard and value["reference_history"]["coverage_complete"]
            and value["reference_history"]["coverage_steps"] == value["reference"]["steps"], "Invalid completed-query resume record")
    trace = (args.run / value["trace_file"]).resolve()
    trace.relative_to((args.run / "traces").resolve())
    require(sha(trace) == value["trace_sha256"], "Completed-query trace changed")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk") and torch.cuda.is_available(),
            "Run only in an Edinburgh Slurm GPU allocation")
    os.umask(0o077)
    args.root, args.run = args.root.resolve(), args.run.resolve()
    config_path = args.run / "config.json"
    config, config_sha = json.loads(config_path.read_text()), sha(config_path)
    require(config["version"] == "asr-full-e2e-v1" and config["shards"] == 4 and 0 <= args.shard < 4,
            "Invalid frozen shard plan")
    require(config["plan"]["modes"] == ["none", "all"] and config["plan"]["max_new_tokens"] == 100
            and config["plan"]["vocab_size"] == 59264, "Frozen inference settings changed")
    queries = config["queries"]
    require([q["query_id"] for q in queries] == config["cohorts"]["all"]
            and len(queries) == len(set(q["query_id"] for q in queries)) == 475
            and all(q["shard"] == index % 4 for index, q in enumerate(queries)), "Invalid full-query/shard coverage")
    selected = queries[args.shard::4]
    require(config["plan"]["replay_control_ids"] == [queries[index]["query_id"] for index in range(4)], "Replay-control IDs changed")
    if os.getenv("SLURM_ARRAY_TASK_ID") is not None:
        require(int(os.environ["SLURM_ARRAY_TASK_ID"]) == args.shard, "Array task and shard disagree")
    for name, expected in config["source_sha256"].items():
        require(sha(args.root / name) == expected, "Frozen source changed: " + name)
    if "input_paths" in config:
        require(set(config["input_paths"]) == set(config["input_sha256"]), "Input proof path/hash coverage mismatch")
        for name, path in config["input_paths"].items():
            require(sha(Path(path)) == config["input_sha256"][name], "Frozen input changed: " + name)
    os.environ["HF_HUB_CACHE"] = config["staging"]["hf_cache"]
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["ASR_LIB"] = config["library"]["path"]
    sys.path.insert(0, str(args.root / "hw1-asr"))
    sys.path.insert(0, str(args.root / "rag/src"))
    from asr_worker import read_audio
    from glm_asr_cuda import ops
    from glm_asr_cuda.model import MODEL_NAME, load_model
    from huggingface_hub import snapshot_download
    require(config["library"]["sha256"] == LIBRARY_SHA256 == sha(ops.library_path()), "Frozen CUDA library changed")
    require(MODEL_NAME == config["model"]["name"] and config["model"]["snapshot_revision"] == SNAPSHOT, "Model changed")
    snapshot = Path(snapshot_download(MODEL_NAME, local_files_only=True)).resolve()
    require(snapshot.name == SNAPSHOT and snapshot == Path(config["staging"]["snapshot_path"]).resolve(), "Staged snapshot changed")
    staging = config["staging"]
    proof_sha = None
    if staging.get("proof_path"):
        proof_sha = sha(Path(staging["proof_path"]))
        require(proof_sha == staging.get("proof_sha256", staging.get("proof_sha")), "Staging proof changed")
        proof = json.loads(Path(staging["proof_path"]).read_text())
        require(proof["status"] == "complete" and proof["audio_files"] == 475
                and proof["library_sha256"] == LIBRARY_SHA256
                and proof["model_files_sha256"] == config["model"]["files_sha256"]
                and Path(proof["snapshot_path"]).resolve() == snapshot, "Staging snapshot proof mismatch")
        for name in config["model"]["files_sha256"]:
            path = snapshot / name
            require(path.is_file() and path.stat().st_size == proof["model_file_sizes"][name]
                    and path.stat().st_mode & 0o222 == 0, "Immutable staged checkpoint file changed: " + name)
    else:
        for name, expected in config["model"]["files_sha256"].items():
            require(sha(snapshot / name) == expected, "Staged checkpoint bytes changed: " + name)
    versions = {name: {"version": importlib.metadata.version(name), "path": importlib.util.find_spec(module).origin}
                for name, module in (("torch", "torch"), ("numpy", "numpy"), ("transformers", "transformers"),
                                     ("safetensors", "safetensors"), ("huggingface-hub", "huggingface_hub"))}
    require({name: value["version"] for name, value in versions.items()} == config["versions"], "Runtime versions changed")
    before = precision_state()
    torch.backends.fp32_precision = torch.backends.cuda.matmul.fp32_precision = torch.backends.cudnn.fp32_precision = "ieee"
    precision = precision_state()
    require(precision == config["plan"]["precision"] == {"global": "ieee", "matmul": "ieee", "cudnn": "ieee"}, "Precision controls changed")
    tiny = torch.tensor([[1., 2.], [3., 4.]], device="cuda", dtype=torch.float32)
    require(np.array_equal((tiny @ tiny).cpu().numpy(), np.array([[7., 10.], [15., 22.]], dtype=np.float32)), "CUDA API probe failed")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    require("H200" in properties.name and (properties.major, properties.minor, properties.multi_processor_count) == (9, 0, 16),
            "Use the frozen H200 MIG 1g SM90 device context")
    counters = {"model_loads": 0, "model_generate_calls": 0, "audio_encoder_calls": 0, "processor_requests": 0,
                "replay_next_token_calls": 0, "resumed_queries": 0, "completed_queries": 0}
    job = {"status": "running", "shard": args.shard, "slurm_job_id": os.environ["SLURM_JOB_ID"],
           "config_sha256": config_sha, "started_utc": utc(), "precision": {"before": before, "after": precision},
           "device": {"name": properties.name, "major": properties.major, "minor": properties.minor,
                      "multi_processor_count": properties.multi_processor_count, "total_memory": properties.total_memory,
                      "uuid": str(getattr(properties, "uuid", "unavailable"))}, "versions": versions,
           "source_sha256": config["source_sha256"], "library_sha256": LIBRARY_SHA256, "staging_proof_sha256": proof_sha,
           "new_api_calls": 0, "new_external_asr_calls": 0, "new_rag_encoder_calls": 0, **counters}
    job_path = args.run / "jobs" / (os.environ["SLURM_JOB_ID"] + ".json")
    job_path.parent.mkdir(parents=True, exist_ok=True)
    write(job_path, job)
    try:
        model, processor = None, None
        for query in selected:
            require(sha(Path(staging["data_dir"]) / query["audio_file"]) == query["audio_sha256"], "Resume/input audio hash mismatch")
            if load_completed(query, args, config_sha) is not None:
                counters["resumed_queries"] += 1
            else:
                if model is None:
                    ops.set_cuda_groups("none")
                    model, processor = load_model(device="cuda")
                    counters["model_loads"] += 1
                    require(model.c.vocab == config["plan"]["vocab_size"] and model.w
                            and all(t.is_cuda and t.dtype == torch.float32 and t.is_contiguous() for t in model.w.values()),
                            "Expected one float32 CUDA model per active shard")
                process_query(query, args, config, config_sha, model, processor, ops, read_audio, counters)
            counters["completed_queries"] += 1
            write(job_path, {**job, **counters, "last_query_id": query["query_id"]})
            print(json.dumps({"stage": "query_complete", "shard": args.shard, "query_id": query["query_id"],
                              "completed": counters["completed_queries"], "planned": len(selected)}), flush=True)
        require(sha(config_path) == config_sha and sha(ops.library_path()) == LIBRARY_SHA256
                and precision_state() == precision, "Frozen runtime context changed during the shard")
        for name, expected in config["source_sha256"].items():
            require(sha(args.root / name) == expected, "Source changed during the shard: " + name)
        shard_path = args.run / "shards" / (str(args.shard) + ".json")
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        write(shard_path, {"status": "complete", "shard": args.shard, "config_sha256": config_sha,
              "query_ids": [q["query_id"] for q in selected], "planned_queries": len(selected),
              "completed_queries": counters["completed_queries"], "resumed_queries": counters["resumed_queries"],
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "utc": utc()})
        write(job_path, {**job, **counters, "status": "complete", "finished_utc": utc()})
    except Exception as error:
        write(job_path, {**job, **counters, "status": "fail", "finished_utc": utc(),
              "errors": [{"type": type(error).__name__, "message": str(error)}]})
        raise


if __name__ == "__main__":
    main()
