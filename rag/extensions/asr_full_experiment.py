"""Complete the frozen four-ASR raw/normalized experiment, exclusively in Slurm."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time

import httpx2
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from openai import OpenAI
from tokenizers import Tokenizer

from asr_retrieval_pilot import MODELS, RULES, TAU, freeze, sha, transcript_path, utc, write_json, safe_error
from transcript_normalization import POLICY, POLICY_VERSION, normalize_transcript
from asr_full_summary import build_summary

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.asr_adapter import ASRAdapter
from src.data import CourseData
from src.encoder import LocalEncoder
from src.metrics import recall_at_k
from src.retriever import ExactDenseRetriever

NORMALIZER_SHA256 = "fff6993b2f1ae85db3a65a6c645a6e2ac3ad0ec415eb82af3f73a763de8e9dd2"
GLM_SNAPSHOT = "61ba4e0b3309b6656edea3e93e419f7bd5c61957"
MAI_OPTIONS = {"options": {"azure": {"enhancedMode": {"enabled": True, "model": "MAI-Transcribe-2",
               "modelOptions": {"transcribeStyle": "verbatim"}}, "diarization": {"enabled": False}}}}


def options(arm):
    value = {"model": MODELS[arm], "language": "en", "temperature": 0, "response_format": "json"}
    if arm == "mai":
        value["provider"] = MAI_OPTIONS
    return value


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_transcript(value, arm, audio, imported=False):
    if (value["query_id"] != audio["query_id"] or value["model"] != MODELS[arm]
            or value["audio_sha256"] != audio["audio_sha256"] or value["status"] != "complete"
            or not value.get("transcript", "").strip()):
        raise RuntimeError("Transcript provenance/status mismatch")
    if arm == "glm":
        if value["asr_cuda"] != "all" or value["max_new_tokens"] != 100:
            raise RuntimeError("GLM decoding options changed")
        if not imported and value["glm_snapshot"] != GLM_SNAPSHOT:
            raise RuntimeError("GLM checkpoint changed")
    elif value["requested_options"] != options(arm):
        raise RuntimeError("External ASR options changed")


def seed_and_load(args, selection):
    original = json.loads((args.pilot_run / "config.json").read_text())
    if (original["models"] != MODELS or original["glm_snapshot"] != GLM_SNAPSHOT
            or original["asr_cuda"] != "all" or original["glm_max_new_tokens"] != 100
            or original["mai_provider_options"] != MAI_OPTIONS
            or original["api_options"] != {"language": "en", "temperature": 0, "response_format": "json", "max_retries": 0}):
        raise RuntimeError("Pilot provenance does not match full experiment")
    pilot_ids = selection["pilot_ids"]
    origins, transcripts = [], {}
    for audio in selection["audio"]:
        qid = audio["query_id"]
        for arm in MODELS:
            target = transcript_path(args.out, arm, qid)
            imported = qid in pilot_ids
            if imported:
                source = transcript_path(args.pilot_run, arm, qid)
                result = json.loads(source.read_text())
                validate_transcript(result, arm, audio, imported=True)
                if target.exists() and sha(target) != sha(source):
                    raise RuntimeError("Imported pilot transcript changed")
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                origins.append({"query_id": qid, "arm": arm, "source_path": str(source),
                                "source_sha256": sha(source), "pilot_config_sha256": sha(args.pilot_run / "config.json")})
            if target.exists():
                result = json.loads(target.read_text())
                validate_transcript(result, arm, audio, imported=imported)
                transcripts[qid, arm] = result
    freeze(args.out / "cache-origins.json", origins)
    return transcripts


def attempt_files(out):
    return sorted((out / "api-attempts").glob("*/*.json"))


def transcribe(args, selection, transcripts):
    load_dotenv(Path.home() / ".config/edin-mls/openrouter.env", override=True)
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Private OpenRouter credential is absent")
    with OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0, timeout=90) as client:
        catalogue = client.get("/models?output_modalities=transcription", cast_to=dict)
        account = client.get("/key", cast_to=dict)["data"]
    write_json(args.out / ("catalogue-" + os.environ["SLURM_JOB_ID"] + ".json"), catalogue)
    catalogue_models = {m["id"]: m for m in catalogue["data"]}
    rates = {a: float(catalogue_models[m]["pricing"]["prompt"]) / (3600 if a == "mai" else 1)
             for a, m in MODELS.items() if a != "glm"}
    pending = [(audio, arm) for audio in selection["audio"] for arm in rates
               if (audio["query_id"], arm) not in transcripts]
    estimates = {(audio["query_id"], arm): max(audio["seconds"], 1) * rates[arm]
                 for audio in selection["audio"] for arm in rates}
    previous_attempts = [json.loads(p.read_text()) for p in attempt_files(args.out)]
    reserved = 2 * (sum(estimates[a["query_id"], arm] for a, arm in pending)
                    + sum(max(a.get("response_cost_usd") or 0, a["estimated_cost_usd"]) for a in previous_attempts)) + 0.02
    if reserved > 1 or (account.get("limit_remaining") is not None and account["limit_remaining"] < reserved):
        raise RuntimeError("Conservative USD 1 spend guard or key allowance exceeded")
    if not (args.out / "billing-start.json").exists():
        write_json(args.out / "billing-start.json", {"utc": utc(), "usage_usd": account.get("usage"),
                   "estimated_new_usd": sum(estimates[a["query_id"], arm] for a, arm in pending),
                   "conservative_reserve_usd": reserved, "guard_usd": 1, "rate_units": {
                   "whisper": "USD/second", "mai": "USD/hour divided by 3600", "voxtral": "USD/second"}})
    write_json(args.out / ("spend-preflight-" + os.environ["SLURM_JOB_ID"] + ".json"),
               {"utc": utc(), "pending_requests": len(pending), "reserved_usd": reserved, "rates_usd_per_second": rates})
    if args.stage != "api":
        with ASRAdapter(cuda_groups="all", max_new_tokens=100) as asr:
            for audio in selection["audio"]:
                qid = audio["query_id"]
                if (qid, "glm") in transcripts:
                    continue
                started = time.perf_counter()
                text = asr.transcribe(args.data_dir / audio["audio_file"])
                value = {"status": "complete", "query_id": qid, "model": MODELS["glm"],
                         "audio_sha256": audio["audio_sha256"], "transcript": text, "utc": utc(),
                         "seconds": time.perf_counter() - started, "asr_cuda": "all", "max_new_tokens": 100,
                         "glm_snapshot": GLM_SNAPSHOT, "slurm_job_id": os.environ["SLURM_JOB_ID"]}
                write_json(transcript_path(args.out, "glm", qid), value)
                transcripts[qid, "glm"] = value
                print(json.dumps({"stage": "glm", "completed": sum(a == "glm" for q, a in transcripts),
                                  "total": len(selection["audio"]), "query_id": qid}), flush=True)
    if any((audio["query_id"], "glm") not in transcripts for audio in selection["audio"]):
        raise RuntimeError("Complete GLM baseline before paid requests")

    def request(audio, arm):
        qid = audio["query_id"]
        stem = hashlib.sha256(qid.encode()).hexdigest()[:20]
        history = sorted((args.out / "api-attempts" / arm).glob(stem + "-*.json"))
        if history and not args.retry_failed:
            return {"query_id": qid, "arm": arm, "error": "Prior failed/ambiguous request; explicit retry flag required"}
        if len(history) >= 3:
            return {"query_id": qid, "arm": arm, "error": "Maximum three explicitly logged attempts reached"}
        attempt = args.out / "api-attempts" / arm / (stem + "-" + str(len(history) + 1) + ".json")
        metadata = {"query_id": qid, "arm": arm, "model": MODELS[arm], "status": "started", "utc": utc(),
                    "estimated_cost_usd": estimates[qid, arm], "slurm_job_id": os.environ["SLURM_JOB_ID"]}
        write_json(attempt, metadata)
        started = time.perf_counter()
        try:
            payload = dict(options(arm), input_audio={"data": base64.b64encode(
                (args.data_dir / audio["audio_file"]).read_bytes()).decode("ascii"), "format": "wav"})
            with OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0, timeout=90) as client:
                response = client.post("/audio/transcriptions", cast_to=httpx2.Response, body=payload)
                raw = response.json()
                headers = {h: response.headers.get(h) for h in ["x-request-id", "x-openrouter-provider", "openrouter-provider", "x-generation-id", "date"]}
            metadata.update(response_cost_usd=(raw.get("usage") or {}).get("cost"), raw_response=raw, response_headers=headers)
            text = raw.get("text")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("Provider returned an empty transcript")
            text = text.strip()
            result = {"status": "complete", "query_id": qid, "audio_sha256": audio["audio_sha256"],
                      "model": MODELS[arm], "transcript": text, "seconds": time.perf_counter() - started,
                      "utc": utc(), "raw_response": raw, "response_headers": headers,
                      "requested_options": options(arm), "estimated_cost_usd": estimates[qid, arm],
                      "request_attempt": str(attempt.relative_to(args.out)), "slurm_job_id": os.environ["SLURM_JOB_ID"]}
            write_json(transcript_path(args.out, arm, qid), result)
            metadata.update(status="complete", finished_utc=utc())
            write_json(attempt, metadata)
            return {"query_id": qid, "arm": arm, "result": result,
                    "accounted_cost_usd": float(metadata["response_cost_usd"]) if metadata.get("response_cost_usd") is not None else estimates[qid, arm]}
        except Exception as exc:
            metadata.update(status="failed", finished_utc=utc(), error=safe_error(exc, key),
                            error_type=type(exc).__name__, request_id=getattr(exc, "request_id", None))
            write_json(attempt, metadata)
            return {"query_id": qid, "arm": arm, "error": metadata["error"],
                    "accounted_cost_usd": float(metadata["response_cost_usd"]) if metadata.get("response_cost_usd") is not None else estimates[qid, arm]}

    random.Random(26).shuffle(pending)
    failures = []
    accounted_spend = sum(max(a.get("response_cost_usd") or 0, a["estimated_cost_usd"]) for a in previous_attempts)
    budget_stop = False
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(request, audio, arm) for audio, arm in pending]
        for index, future in enumerate(as_completed(futures), 1):
            if future.cancelled():
                continue
            value = future.result()
            accounted_spend += value.get("accounted_cost_usd", 0)
            if "result" in value:
                transcripts[value["query_id"], value["arm"]] = value["result"]
            else:
                failures.append(value)
            print(json.dumps({"stage": "api", "finished_new_tasks": index, "planned": len(pending),
                              "query_id": value["query_id"], "arm": value["arm"], "success": "result" in value}), flush=True)
            if accounted_spend >= 1 and not budget_stop:
                budget_stop = True
                for queued in futures:
                    queued.cancel()
                failures.append({"error": "USD 1 accounted-spend guard stopped queued tasks; in-flight requests may still be billed"})
    write_json(args.out / "transcription-failures.json", failures)
    with OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0, timeout=90) as client:
        account = client.get("/key", cast_to=dict)["data"]
    attempts = [json.loads(p.read_text()) for p in attempt_files(args.out)]
    all_costs = [(v.get("raw_response", {}).get("usage") or {}).get("cost")
                 for (q, a), v in transcripts.items() if a != "glm"]
    write_json(args.out / "billing.json", {"utc": utc(), "key_usage_after_usd": account.get("usage"),
               "new_attempts": len(attempts), "failed_or_ambiguous_attempts": sum(a["status"] != "complete" for a in attempts),
               "new_response_cost_sum_usd": math.fsum(a["response_cost_usd"] for a in attempts if a.get("response_cost_usd") is not None),
               "all_successful_response_cost_sum_usd": math.fsum(c for c in all_costs if c is not None),
               "successful_responses_with_cost": sum(c is not None for c in all_costs),
               "failed_or_ambiguous_cost_is_unknown": any(a["status"] != "complete" and a.get("response_cost_usd") is None for a in attempts)})
    return transcripts


def retrieve(args, selection, config, transcripts):
    expected = {(r["query_id"], a) for r in selection["audio"] for a in MODELS}
    if set(transcripts) != expected:
        write_json(args.out / "completion.json", {"status": "partial", "transcripts": len(transcripts),
                   "expected_transcripts": len(expected), "missing": [list(k) for k in sorted(expected - set(transcripts))]})
        raise RuntimeError("Full transcription coverage is required; missing cells retained in completion manifest")
    data = CourseData(args.data_dir)
    qrels = {split: data.load_qrels(split) for split in selection["splits"]}
    tokenizer = Tokenizer.from_file(str(args.tokenizer_file))
    encoder = LocalEncoder(args.encoder_dir, device="cpu")
    retriever = ExactDenseRetriever(args.data_dir / "embeddings/corpus_enc_a.npy", args.data_dir / "embeddings/corpus_enc_a_ids.json")
    if len(retriever.record_ids) != len(set(retriever.record_ids)) or len(retriever.record_ids) != 24000:
        raise RuntimeError("Unexpected corpus")
    mappings, rows, vector_keys, vectors = [], [], [], []
    token_cache = {}
    for index, audio in enumerate(selection["audio"], 1):
        qid, split = audio["query_id"], audio["split"]
        for arm in MODELS:
            raw = transcripts[qid, arm]["transcript"]
            normalized = normalize_transcript(raw)
            if not normalized or normalize_transcript(normalized) != normalized:
                write_json(args.out / "normalization-error.json", {"query_id": qid, "arm": arm, "raw": raw, "normalized": normalized})
                raise RuntimeError("Frozen normalizer produced empty/non-idempotent output; rule is not changed")
            mappings.append({"query_id": qid, "split": split, "arm": arm, "model": MODELS[arm],
                             "audio_sha256": audio["audio_sha256"], "source_sha256": sha(transcript_path(args.out, arm, qid)),
                             "raw": raw, "normalized": normalized, "raw_text_sha256": text_sha(raw),
                             "normalized_text_sha256": text_sha(normalized)})
            for condition, text in [("raw", raw), ("normalized", normalized)]:
                vector = np.asarray(encoder.encode(text), dtype=np.float32)
                norm = np.linalg.norm(vector)
                if not np.isfinite(vector).all() or not norm > 0:
                    raise RuntimeError("Invalid query vector")
                vector = vector / norm
                vector_keys.append([qid, arm, condition])
                vectors.append(vector)
                similarities = retriever.embeddings @ vector
                order = np.argsort(-similarities, kind="stable")
                scores = similarities[order].astype(np.float64)
                if not np.isfinite(scores).all() or not np.all(np.diff(scores) <= 0):
                    raise RuntimeError("Invalid ranking")
                matching = int(np.count_nonzero(scores >= TAU))
                for rule, count in [("top5", 5), ("cutoff", max(1, matching))]:
                    ids = [retriever.record_ids[int(i)] for i in order[:count]]
                    for rid in ids:
                        if rid not in token_cache:
                            token_cache[rid] = len(tokenizer.encode(data.record_text(rid), add_special_tokens=False).ids)
                    gold = qrels[split][qid]
                    hits = len(set(ids) & set(gold))
                    recall = recall_at_k(ids, gold, count)
                    if recall != hits / len(gold):
                        raise RuntimeError("Recall mismatch")
                    rows.append({"query_id": qid, "split": split, "arm": arm, "model": MODELS[arm],
                        "condition": condition, "rule": rule, "recall": recall, "relevant_hits": hits,
                        "total_relevant": len(gold), "context_tokens": sum(token_cache[r] for r in ids),
                        "documents": count, "record_ids": ids, "selected_scores": scores[:count].tolist(),
                        "top_similarity": float(scores[0]), "fallback": rule == "cutoff" and matching == 0,
                        "transcript_tokens": len(tokenizer.encode(text, add_special_tokens=False).ids)})
        if index % 10 == 0 or index == len(selection["audio"]):
            print(json.dumps({"stage": "retrieval", "queries": index, "total": len(selection["audio"])}), flush=True)
    np.savez_compressed(args.out / "query-vectors.npz", vectors=np.stack(vectors), keys=np.array(vector_keys))
    write_json(args.out / "transcript-mappings.json", mappings)
    (args.out / "retrieval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(args.out / "document-token-cache.json", token_cache)
    cells = {(r["query_id"], r["arm"], r["condition"], r["rule"]): r for r in rows}
    max_drift = 0.0
    checked = 0
    for condition, source in [("raw", args.pilot_run / "retrieval.jsonl"),
                              ("normalized", args.normalized_pilot / "normalized/retrieval.jsonl")]:
        for previous in map(json.loads, source.read_text().splitlines()):
            row = cells[previous["query_id"], previous["arm"], condition, previous["rule"]]
            for field in ["record_ids", "recall", "context_tokens", "documents", "fallback", "transcript_tokens"]:
                if row[field] != previous[field]:
                    raise RuntimeError("Pilot reproduction mismatch: " + field)
            max_drift = max(max_drift, float(np.max(np.abs(np.array(row["selected_scores"]) - previous["selected_scores"]))))
            checked += 1
    if max_drift > 1e-6:
        raise RuntimeError("Pilot score drift exceeds 1e-6")
    write_json(args.out / "pilot-control-check.json", {"status": "pass", "cells": checked, "max_abs_score_drift": max_drift})
    summary = build_summary(rows, config["cohorts"], mappings)
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "completion.json", {"status": summary["status"], "slurm_job_id": os.environ["SLURM_JOB_ID"],
               "transcripts": len(transcripts), "retrieval_cells": len(rows), "queries": len(selection["audio"]),
               "expected_retrieval_cells": 16 * len(selection["audio"]), "utc": utc()})
    print(json.dumps({"status": summary["status"], "retrieval_cells": len(rows)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-file", type=Path, required=True)
    parser.add_argument("--pilot-run", type=Path, default=ROOT / "results/asr-four-model-pilot-20261116")
    parser.add_argument("--normalized-pilot", type=Path, default=ROOT / "results/asr-four-model-normalized-20261116")
    parser.add_argument("--stage", choices=["all", "api", "retrieve"], default="all")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not socket.getfqdn().endswith(".inf.ed.ac.uk"):
        raise RuntimeError("Run only in Edinburgh Slurm")
    os.umask(0o077)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if sha(Path(__file__).with_name("transcript_normalization.py")) != NORMALIZER_SHA256:
        raise RuntimeError("Frozen normalization rule changed")
    selection = json.loads((Path.home() / ".cache/mls-asr-full/selection.json").read_text())
    freeze(args.out / "selection.json", selection)
    original = json.loads((args.pilot_run / "config.json").read_text())
    for relative, expected in original["sources"].items():
        path = ROOT.parent / relative
        if relative.startswith("rag/data/rag_corpus/"):
            path = args.data_dir / relative.removeprefix("rag/data/rag_corpus/")
        elif relative.startswith("rag/models/enc_a/"):
            path = args.encoder_dir / relative.removeprefix("rag/models/enc_a/")
        if sha(path) != expected:
            raise RuntimeError("Original data/source changed: " + relative)
    for audio in selection["audio"]:
        if sha(args.data_dir / audio["audio_file"]) != audio["audio_sha256"]:
            raise RuntimeError("Audio changed after preflight")
    if sha(args.tokenizer_file) != original["tokenizer"]["sha256"]:
        raise RuntimeError("Pinned tokenizer changed")
    cohorts = {"all": selection["query_ids"],
               "practice": [r["query_id"] for r in selection["audio"] if r["split"] == "practice"],
               "dev": [r["query_id"] for r in selection["audio"] if r["split"] == "dev"],
               "pilot": selection["pilot_ids"],
               "remaining_dev": [r["query_id"] for r in selection["audio"] if r["split"] == "dev" and r["query_id"] not in selection["pilot_ids"]],
               "remaining_all": [q for q in selection["query_ids"] if q not in selection["pilot_ids"]]}
    sources = {str(path.relative_to(ROOT.parent)): sha(path) for path in [Path(__file__),
               Path(__file__).with_name("asr_full_summary.py"), Path(__file__).with_name("transcript_normalization.py"),
               Path(__file__).with_name("run_asr_full_experiment.sbatch"), ROOT.parent / "report/asr-rag-full-protocol.md"]}
    inputs = dict(original["sources"])
    inputs["rag/data/rag_corpus/qrels_practice.tsv"] = sha(args.data_dir / "qrels_practice.tsv")
    config = {"models": MODELS, "conditions": ["raw", "normalized"], "rules": list(RULES), "tau": TAU,
              "topk": 5, "cohorts": cohorts, "sources": sources, "input_sources": inputs,
              "normalizer_sha256": NORMALIZER_SHA256, "policy_version": POLICY_VERSION, "policy": POLICY,
              "tokenizer": original["tokenizer"], "glm_snapshot": GLM_SNAPSHOT, "asr_cuda": "all",
              "glm_max_new_tokens": 100, "api_options": {a: options(a) for a in MODELS if a != "glm"},
              "api_max_retries": 0, "api_workers": 3, "task_seed": 26, "new_spend_guard_usd": 1,
              "selection_sha256": sha(args.out / "selection.json"), "pilot_config_sha256": sha(args.pilot_run / "config.json"),
              "pilot_retrieval_sha256": sha(args.pilot_run / "retrieval.jsonl"),
              "normalized_pilot_retrieval_sha256": sha(args.normalized_pilot / "normalized/retrieval.jsonl")}
    freeze(args.out / "config.json", config)
    versions = {n: {"version": importlib.metadata.version(n), "path": importlib.util.find_spec(m).origin}
                for n, m in [("numpy", "numpy"), ("sentence-transformers", "sentence_transformers"),
                             ("openai", "openai"), ("httpx2", "httpx2"), ("tokenizers", "tokenizers"), ("torch", "torch")]}
    write_json(args.out / ("environment-" + os.environ["SLURM_JOB_ID"] + ".json"), {"utc": utc(),
               "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.getfqdn(), "python": sys.version,
               "versions": versions, "stage": args.stage, "retry_failed": args.retry_failed,
               "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
               "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode()})
    if args.stage == "all" and Path(snapshot_download(MODELS["glm"], local_files_only=True)).name != GLM_SNAPSHOT:
        raise RuntimeError("GLM cached revision changed")
    transcripts = seed_and_load(args, selection)
    if args.stage in ["all", "api"]:
        transcripts = transcribe(args, selection, transcripts)
    retrieve(args, selection, config, transcripts)


if __name__ == "__main__":
    main()
