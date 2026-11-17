"""Freeze and stage the full ASR numerical-validation inputs inside CPU Slurm."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import wave


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def copy_verified(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha(destination) == expected:
        destination.chmod(0o444)
        return
    temporary = destination.with_name(destination.name + ".preparing-" + os.environ["SLURM_JOB_ID"])
    shutil.copyfile(source, temporary)
    require(sha(temporary) == expected, "Staged content mismatch: " + destination.name)
    temporary.replace(destination)
    destination.chmod(0o444)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run", "stage-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID"), "Stage only inside Slurm")
    os.umask(0o077)
    root, run, stage = args.root.resolve(), args.run.resolve(), args.stage_dir.resolve()
    require(stage.parent == Path("/disk/scratch").resolve()
            and stage.name.startswith("mls-asr-full-e2e-s2487016-"), "Unexpected staging directory: " + str(stage))
    run.mkdir(parents=True, exist_ok=True)
    require(not (run / "config.json").exists(), "Global configuration already frozen; resume workers without restaging")
    stage.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(stage).free
    require(free >= 15 * 1024**3, "Need 15 GiB free for the shared checkpoint/audio staging cache")
    plan_path = root / "hw1-asr/asr-full-e2e-plan.json"
    plan = read(plan_path)
    require(socket.getfqdn() == plan["hostname"], "Wrong staging host")
    require(plan["expected_queries"] == 475 and plan["shards"] == 4 and plan["max_new_tokens"] == 100,
            "Unexpected fixed full-audio plan")
    old = root / "rag/results/asr-four-model-full-20261117"
    single = root / "rag/results/asr-e2e-error-20261117"
    old_audit, old_config, selection = (read(old / name) for name in ("audit.json", "config.json", "selection.json"))
    single_audit, single_config = (read(single / name) for name in ("audit.json", "config.json"))
    require(old_audit["status"] == single_audit["status"] == "pass", "Prior input/library audits must pass")
    input_paths = {"full/" + name: str(old / name) for name in ("audit.json", "config.json", "selection.json")}
    input_paths.update({"single/" + name: str(single / name) for name in ("audit.json", "config.json", "completion.json")})
    for name in ("config.json", "selection.json"):
        require(sha(old / name) == old_audit["artifact_sha256"][name], "Original frozen selection/config changed")
    require(sha(single / "config.json") == single_audit["input_sha256"]["config.json"], "Audited single-clip config changed")
    for relative, expected in single_config["source_sha256"].items():
        require(sha(root / relative) == expected, "Previously validated model/kernel source changed: " + relative)
    require(sha(single / "libasr_kernels.so") == single_audit["library_sha256"], "Historical CUDA library changed")
    build_proof_path = run / "library-build.json"
    build_proof = read(build_proof_path)
    expected_build_sources = ["hw1-asr/Makefile"] + [str(path.relative_to(root)) for path in sorted((root / "hw1-asr/csrc").iterdir()) if path.suffix in (".cu", ".cuh", ".h")]
    require(build_proof["status"] == "complete" and build_proof["sm"] == plan["library_build_sm"]
            and build_proof["hostname"] == plan["hostname"]
            and set(build_proof["source_sha256"]) == set(expected_build_sources), "Wrong architecture build proof")
    for relative in expected_build_sources:
        require(sha(root / relative) == build_proof["source_sha256"][relative], "Build source changed: " + relative)
    require(plan["library_sha256"] is None, "Build digest must be frozen from proof before inference")
    plan = dict(plan, library_sha256=build_proof["library_sha256"])
    library = run / "build/sm_86/libasr_kernels.so"
    require(sha(library) == plan["library_sha256"], "New SM86 CUDA library changed")
    saved_library = run / "libasr_kernels.so"
    copy_verified(library, saved_library, plan["library_sha256"])
    snapshot = Path(single_config["model"]["source_snapshot_path"])
    require(snapshot.name == plan["snapshot_revision"] and snapshot.is_dir(), "Pinned source snapshot missing")
    expected_model = single_config["model"]["files_sha256"]
    model_folder = "models--" + plan["model_name"].replace("/", "--")
    hf_cache = stage / "hf"
    staged_snapshot = hf_cache / model_folder / "snapshots" / plan["snapshot_revision"]
    model_sizes = {}
    for relative, expected in expected_model.items():
        destination = staged_snapshot / relative
        destination.resolve().relative_to(staged_snapshot.resolve())
        copy_verified(snapshot / relative, destination, expected)
        model_sizes[relative] = destination.stat().st_size
    refs = hf_cache / model_folder / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    if (refs / "main").exists():
        require((refs / "main").read_text(encoding="ascii").strip() == plan["snapshot_revision"], "Staged model ref changed")
    else:
        (refs / "main").write_text(plan["snapshot_revision"], encoding="ascii")
    (refs / "main").chmod(0o444)
    cohorts = old_config["cohorts"]
    require({key: len(value) for key, value in cohorts.items()} == {
        "all": 475, "practice": 350, "dev": 125, "pilot": 8, "remaining_dev": 117, "remaining_all": 467}, "Historical cohorts changed")
    require(selection["query_ids"] == cohorts["all"] and len(set(selection["query_ids"])) == 475,
            "Original query order/coverage changed")
    audio_by_id = {item["query_id"]: item for item in selection["audio"]}
    require(len(audio_by_id) == len(selection["audio"]) == 475 and set(audio_by_id) == set(cohorts["all"]), "Audio selection incomplete")
    source_data = (root / "rag/data/rag_corpus").resolve()
    data_dir = stage / "data"
    queries = []
    for index, qid in enumerate(cohorts["all"]):
        item = audio_by_id[qid]
        source = (source_data / item["audio_file"]).resolve()
        source.relative_to(source_data)
        destination = data_dir / item["audio_file"]
        destination.resolve().relative_to(data_dir.resolve())
        copy_verified(source, destination, item["audio_sha256"])
        with wave.open(str(destination), "rb") as wav:
            require((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) == (16000, 1, 2, "NONE"), "Unexpected audio format")
            seconds = wav.getnframes() / wav.getframerate()
        require(abs(seconds - item["seconds"]) <= 1e-9, "Audio duration changed")
        queries.append({key: item[key] for key in ("query_id", "split", "audio_file", "audio_sha256", "seconds")})
        queries[-1]["shard"] = index % 4
    require(len({hashlib.sha256(q["query_id"].encode()).hexdigest()[:20] for q in queries}) == 475, "Query filename collision")
    proof = {"status": "complete", "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.getfqdn(),
        "snapshot_path": str(staged_snapshot), "model_files_sha256": expected_model, "model_file_sizes": model_sizes,
        "audio_files": 475, "audio_sha256": {q["audio_file"]: q["audio_sha256"] for q in queries},
        "library_sha256": plan["library_sha256"], "staging_free_bytes_before": free}
    proof_path = run / "staging-proof.json"
    write(proof_path, proof)
    paths = ["hw1-asr/measure_asr_full_e2e.py", "hw1-asr/measure_asr_e2e_error.py",
        "hw1-asr/prepare_asr_full_e2e.py", "hw1-asr/asr-full-e2e-plan.json", "hw1-asr/run_asr_full_e2e.sbatch",
        "hw1-asr/run_prepare_asr_full_e2e.sbatch", "hw1-asr/run_build_asr_full_e2e.sbatch",
        "hw1-asr/build_asr_full_e2e_library.py", "report/asr-full-e2e-protocol.md", "hw1-asr/benchmark.py",
        "hw1-asr/glm_asr_cuda/model.py", "hw1-asr/glm_asr_cuda/ops.py", "hw1-asr/glm_asr_cuda/__init__.py",
        "hw1-asr/Makefile", "rag/src/asr_worker.py"]
    paths += [str(path.relative_to(root)) for path in sorted((root / "hw1-asr/csrc").iterdir()) if path.suffix in (".cu", ".cuh", ".h")]
    require(all((root / path).is_file() for path in paths), "Missing frozen producer source")
    versions = {name: single_config["versions"][name]["version"] for name in
        ("torch", "numpy", "transformers", "safetensors", "huggingface-hub")}
    plan = dict(plan, replay_control_ids=[queries[shard]["query_id"] for shard in range(4)])
    config = {"version": "asr-full-e2e-v1", "expected_queries": 475, "shards": 4, "plan": plan,
        "queries": queries, "cohorts": cohorts, "cohort_order": list(cohorts),
        "staging": {"root": str(stage), "hf_cache": str(hf_cache), "data_dir": str(data_dir),
            "snapshot_path": str(staged_snapshot), "proof_path": str(proof_path), "proof_sha256": sha(proof_path)},
        "model": {"name": plan["model_name"], "snapshot_revision": plan["snapshot_revision"],
            "source_snapshot_path": str(snapshot), "files_sha256": expected_model},
        "library": {"path": str(saved_library), "sha256": plan["library_sha256"],
            "build_proof_path": str(build_proof_path), "build_proof_sha256": sha(build_proof_path)}, "versions": versions,
        "source_sha256": {path: sha(root / path) for path in paths}, "input_paths": input_paths,
        "input_sha256": {label: sha(path) for label, path in input_paths.items()},
        "prepared_slurm_job_id": os.environ["SLURM_JOB_ID"], "audio_seconds": math.fsum(q["seconds"] for q in queries),
        "analysis_versions": {name: importlib.metadata.version(name) for name in ("numpy", "rapidfuzz")},
        "preparation_python": sys.version, "preparation_new_model_calls": 0, "new_external_asr_calls": 0, "new_api_calls": 0}
    write(run / "config.json", config)
    print(json.dumps({"status": "prepared", "queries": len(queries), "shards": 4, "config_sha256": sha(run / "config.json"),
        "audio_seconds": config["audio_seconds"], "staging": str(stage)}), flush=True)


if __name__ == "__main__":
    main()
