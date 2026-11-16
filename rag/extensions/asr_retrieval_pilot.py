"""Paired four-model ASR pilot; execute only in an Edinburgh Slurm allocation."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import socket
import statistics
import subprocess
import sys
import time
import wave

import numpy as np
import httpx2
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from openai import OpenAI
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.asr_adapter import ASRAdapter
from src.data import CourseData
from src.encoder import LocalEncoder
from src.metrics import recall_at_k
from src.retriever import ExactDenseRetriever

MODELS = {
    "glm": "zai-org/GLM-ASR-Nano-2512",
    "whisper": "openai/whisper-large-v3",
    "mai": "microsoft/mai-transcribe-2",
    "voxtral": "mistralai/voxtral-small-transcribe-3-0",
}
TOKENIZER_REPO = "mistralai/Mistral-Small-4-119B-2603"
TOKENIZER_REVISION = "a11f36bebf709121056b1dbcc943d1c6afbe494d"
RULES = ("top5", "cutoff")
TAU = 0.8925


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"Frozen configuration changed: {path.name}")
    else:
        write_json(path, value)


def utc():
    return datetime.now(timezone.utc).isoformat()


def safe_error(exc, key):
    message = str(exc).replace(key, "[REDACTED]")
    return re.sub(r"sk-or-[A-Za-z0-9_-]+", "[REDACTED]", message)[:1500]


def recording(data_dir, item):
    path = (data_dir / item["audio_file"]).resolve()
    path.relative_to(data_dir.resolve())
    with wave.open(str(path), "rb") as audio:
        properties = {
            "sample_rate": audio.getframerate(), "channels": audio.getnchannels(),
            "sample_width": audio.getsampwidth(), "frames": audio.getnframes(),
            "compression": audio.getcomptype(),
        }
    if (properties["sample_rate"], properties["channels"], properties["sample_width"], properties["compression"]) != (16000, 1, 2, "NONE"):
        raise ValueError("Course pilot requires the original mono 16 kHz PCM16 WAVs")
    if properties["frames"] <= 0:
        raise ValueError("Empty recording")
    audio_hash = sha(path)
    if item.get("sha256") and item["sha256"] != audio_hash:
        raise RuntimeError("Audio differs from the course manifest")
    return dict(query_id=str(item["query_id"]), audio_file=str(path.relative_to(data_dir.resolve())),
                audio_sha256=audio_hash, seconds=properties["frames"] / properties["sample_rate"], **properties)


def transcript_path(out, arm, query_id):
    name = hashlib.sha256(query_id.encode()).hexdigest()[:20]
    return out / "transcripts" / arm / (name + ".json")


def cached_transcript(out, arm, item):
    path = transcript_path(out, arm, item["query_id"])
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["audio_sha256"] != item["audio_sha256"] or result["model"] != MODELS[arm]:
        raise RuntimeError("Transcript cache provenance mismatch")
    return result if result.get("status") == "complete" and result.get("transcript", "").strip() else None


def summarize(rows, selection, out, failures, billing):
    by_cell = {(r["query_id"], r["arm"], r["rule"]): r for r in rows}
    paired = [q for q in selection if all((q, a, p) in by_cell for a in MODELS for p in RULES)]
    table = []
    for rule in RULES:
        for arm in MODELS:
            current = [by_cell[q, arm, rule] for q in paired]
            baseline = [by_cell[q, "glm", rule] for q in paired]
            table.append({
                "arm": arm, "model": MODELS[arm], "rule": rule, "paired_queries": len(paired),
                "mean_recall": statistics.mean(r["recall"] for r in current) if paired else None,
                "delta_recall_pp_vs_glm": statistics.mean(100 * (r["recall"] - b["recall"]) for r, b in zip(current, baseline)) if paired else None,
                "mean_context_tokens": statistics.mean(r["context_tokens"] for r in current) if paired else None,
                "delta_context_tokens_vs_glm": statistics.mean(r["context_tokens"] - b["context_tokens"] for r, b in zip(current, baseline)) if paired else None,
                "mean_documents": statistics.mean(r["documents"] for r in current) if paired else None,
                "fallback_queries": sum(r["fallback"] for r in current),
            })
    interactions = []
    for arm in list(MODELS)[1:]:
        contrasts = [r for r in table if r["arm"] == arm]
        interactions.append({"arm": arm, "cutoff_minus_top5_delta_recall_pp":
            contrasts[1]["delta_recall_pp_vs_glm"] - contrasts[0]["delta_recall_pp_vs_glm"] if paired else None})
    complete = len(paired) == 8 and len(rows) == 64 and not failures
    summary = {"status": "complete" if complete else "partial", "frozen_queries": selection,
               "complete_paired_queries": paired, "retrieval_cells": len(rows), "table": table,
               "interaction": interactions, "failures": failures, "billing": billing,
               "token_cost_definition": "Sum of individually tokenized retrieved document texts; no special tokens, question or prompt scaffolding.",
               "limitations": ["Eight reused dev queries; exploratory, no independent accuracy/superiority claim.",
                               "Original qrels remain fixed across ASR models.",
                               "No answer generation or judging; context tokens are not billed API tokens."]}
    write_json(out / "summary.json", summary)
    lines = ["# Four-model ASR retrieval pilot", "", f"Status: {summary['status']}; complete paired queries: {len(paired)}/8.", "",
             "| ASR | Retrieval | Recall (%) | Change vs GLM (pp) | Context tokens | Token change vs GLM | Mean docs |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in table:
        if paired:
            lines.append(f"| {row['arm']} | {row['rule']} | {100*row['mean_recall']:.3f} | {row['delta_recall_pp_vs_glm']:+.3f} | {row['mean_context_tokens']:.2f} | {row['delta_context_tokens_vs_glm']:+.2f} | {row['mean_documents']:.2f} |")
    lines.extend(["", "Token counts cover retrieved document texts only. Interpret as an eight-query exploratory pilot.", ""])
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "paired_queries": len(paired), "retrieval_cells": len(rows), "table": table}, ensure_ascii=False), flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path.home() / ".config/edin-mls/openrouter.env")
    parser.add_argument("--tokenizer-metadata", type=Path, default=Path.home() / ".cache/mls-asr-pilot/tokenizer-metadata.json")
    parser.add_argument("--budget-usd", type=float, default=1.0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/rag_corpus")
    parser.add_argument("--encoder-dir", type=Path, default=ROOT / "models/enc_a")
    parser.add_argument("--tokenizer-file", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not socket.getfqdn().endswith(".inf.ed.ac.uk"):
        raise RuntimeError("Run only in an Edinburgh Slurm allocation")
    if not 0 < args.budget_usd <= 1:
        raise ValueError("This eight-query pilot has a maximum USD 1 guard")
    os.umask(0o077)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()
    data = CourseData(data_dir)
    qrels = data.load_qrels("dev")
    items = {str(i["query_id"]): i for i in data.audio_items("dev")}
    eligible = sorted(q for q in items if qrels.get(q) and (data_dir / items[q]["audio_file"]).is_file())
    if len(eligible) < 8:
        raise RuntimeError("Fewer than eight eligible dev recordings")
    selected = random.Random(26).sample(eligible, 8)
    audio_rows = [recording(data_dir, items[q]) for q in selected]
    freeze(out / "selection.json", {"split": "dev", "seed": 26, "eligible_ids": eligible, "selected_ids": selected, "audio": audio_rows})
    retriever = ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy", data_dir / "embeddings/corpus_enc_a_ids.json")
    if len(retriever.record_ids) != 24000 or len(set(retriever.record_ids)) != 24000:
        raise RuntimeError("Expected 24,000 unique corpus records")
    for q in selected:
        if not set(qrels[q]) <= set(retriever.record_ids):
            raise RuntimeError("Positive qrels contain records absent from the corpus")
    tokenizer_metadata = json.loads(args.tokenizer_metadata.read_text())
    if (tokenizer_metadata["repo"], tokenizer_metadata["revision"]) != (TOKENIZER_REPO, TOKENIZER_REVISION):
        raise RuntimeError("Report tokenizer identity mismatch")
    tokenizer_file = args.tokenizer_file or Path(tokenizer_metadata["path"])
    if sha(tokenizer_file) != tokenizer_metadata["sha256"]:
        raise RuntimeError("Tokenizer file hash mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    tokenizer.encode("A small tokenization check.", add_special_tokens=False)
    encoder = LocalEncoder(args.encoder_dir, device="cpu")
    probe = encoder.encode("A small encoder compatibility check.")
    retriever.retrieve(probe, k=5)
    assert recall_at_k(["a", "b"], {"b", "d"}, 2) == 0.5
    hashes = {str(p.relative_to(ROOT.parent)): sha(p) for p in [Path(__file__), ROOT / "src/data.py", ROOT / "src/encoder.py", ROOT / "src/retriever.py", ROOT / "src/metrics.py", ROOT / "src/asr_adapter.py", ROOT / "src/asr_worker.py"]}
    for relative in ["corpus.jsonl", "qrels_dev.tsv", "student_audio_manifest.jsonl", "embeddings/corpus_enc_a.npy", "embeddings/corpus_enc_a_ids.json"]:
        hashes["rag/data/rag_corpus/" + relative] = sha(data_dir / relative)
    hashes.update({"rag/models/enc_a/" + str(p.relative_to(args.encoder_dir)): sha(p) for p in args.encoder_dir.rglob("*") if p.is_file()})
    hashes.update({str(p.relative_to(ROOT.parent)): sha(p) for p in (ROOT.parent / "hw1-asr/csrc").glob("*.cu*")})
    glm_snapshot = Path(snapshot_download(MODELS["glm"], local_files_only=True)).name
    config = {"models": MODELS, "sample_seed": 26, "query_ids": selected, "tau": TAU, "topk": 5,
              "asr_cuda": os.getenv("ASR_CUDA", "all"), "glm_max_new_tokens": 100,
              "glm_snapshot": glm_snapshot,
              "tokenizer": tokenizer_metadata, "sources": hashes, "budget_usd": args.budget_usd,
              "api_options": {"language": "en", "temperature": 0, "response_format": "json", "max_retries": 0},
              "mai_provider_options": {"options": {"azure": {"enhancedMode": {"enabled": True, "model": "MAI-Transcribe-2", "modelOptions": {"transcribeStyle": "verbatim"}}, "diarization": {"enabled": False}}}}}
    freeze(out / "config.json", config)
    versions = {n: {"version": importlib.metadata.version(n), "path": importlib.util.find_spec(m).origin}
                for n, m in [("numpy", "numpy"), ("sentence-transformers", "sentence_transformers"), ("openai", "openai"), ("python-dotenv", "dotenv"), ("tokenizers", "tokenizers"), ("huggingface-hub", "huggingface_hub"), ("httpx2", "httpx2")]}
    write_json(out / ("environment-" + os.environ["SLURM_JOB_ID"] + ".json"),
               {"utc": utc(), "hostname": socket.getfqdn(), "slurm_job_id": os.environ["SLURM_JOB_ID"], "python": sys.executable, "versions": versions,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(), "command": sys.argv,
                "data_dir": str(data_dir), "encoder_dir": str(args.encoder_dir), "tokenizer_file": str(tokenizer_file)})
    write_json(out / "preflight.json", {"selected_ids": selected, "eligible_queries": len(eligible), "audio_seconds": sum(a["seconds"] for a in audio_rows), "encoder_dimensions": int(probe.size), "corpus_size": 24000, "encoder_tokenizer_retriever_checked": True})
    if args.validate_only:
        print("CPU APIs and frozen sample validated inside Slurm; no ASR or paid calls.", flush=True)
        return 0
    load_dotenv(args.env_file, override=True)
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is absent")
    failures = []
    transcripts = {}
    with OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0, timeout=90) as client:
        catalogue = client.get("/models?output_modalities=transcription", cast_to=dict)
        write_json(out / "openrouter-catalogue.json", catalogue)
        models = {m["id"]: m for m in catalogue["data"]}
        rates = {arm: float(models[model]["pricing"]["prompt"]) / (3600 if arm == "mai" else 1)
                 for arm, model in MODELS.items() if arm != "glm"}
        estimates = {(a["query_id"], arm): max(a["seconds"], 1) * rates[arm] for a in audio_rows for arm in rates}
        reserve = sum(estimates.values()) * 2 + 0.02
        if reserve > args.budget_usd:
            raise RuntimeError("Conservative estimated spend exceeds the pilot guard")
        account = client.get("/key", cast_to=dict)["data"]
        billing = {"before": {k: account.get(k) for k in ["usage", "limit", "limit_remaining"]}, "estimated_usd": sum(estimates.values()), "reserved_usd": reserve,
                   "rate_units": {"whisper": "USD/second", "mai": "USD/hour divided by 3600", "voxtral": "USD/second"}}
        write_json(out / "billing.json", billing)
        if account.get("limit_remaining") is not None and account["limit_remaining"] < reserve:
            raise RuntimeError("Insufficient key allowance for the conservative reservation")
        # Confirm the local baseline before spending on external transcriptions.
        with ASRAdapter() as asr:
            for audio in audio_rows:
                qid = audio["query_id"]
                result = cached_transcript(out, "glm", audio)
                if result is None:
                    start = time.perf_counter()
                    text = asr.transcribe(data_dir / audio["audio_file"])
                    result = {"status": "complete", "query_id": qid, "audio_sha256": audio["audio_sha256"], "model": MODELS["glm"], "transcript": text,
                              "seconds": time.perf_counter() - start, "utc": utc(), "asr_cuda": config["asr_cuda"], "max_new_tokens": 100}
                    write_json(transcript_path(out, "glm", qid), result)
                transcripts[qid, "glm"] = result
                print(json.dumps({"stage": "transcribed", "query_id": qid, "arm": "glm"}), flush=True)
        tasks = [(audio, arm) for audio in audio_rows for arm in rates]
        random.Random(26).shuffle(tasks)
        recorded_cost = 0.0
        for audio, arm in tasks:
            qid = audio["query_id"]
            result = cached_transcript(out, arm, audio)
            if result is None:
                started = time.perf_counter()
                try:
                    payload = {"model": MODELS[arm], "language": "en", "temperature": 0, "response_format": "json",
                               "input_audio": {"data": base64.b64encode((data_dir / audio["audio_file"]).read_bytes()).decode("ascii"), "format": "wav"}}
                    if arm == "mai":
                        payload["provider"] = config["mai_provider_options"]
                    response = client.post("/audio/transcriptions", cast_to=httpx2.Response, body=payload)
                    raw = response.json()
                    text = str(raw.get("text", "")).strip()
                    if not text:
                        raise RuntimeError("Provider returned an empty transcript")
                    result = {"status": "complete", "query_id": qid, "audio_sha256": audio["audio_sha256"], "model": MODELS[arm], "transcript": text,
                              "seconds": time.perf_counter() - started, "utc": utc(), "raw_response": raw,
                              "response_headers": {h: response.headers.get(h) for h in ["x-request-id", "x-openrouter-provider", "openrouter-provider", "x-generation-id", "date"]},
                              "requested_options": {k: v for k, v in payload.items() if k != "input_audio"}, "estimated_cost_usd": estimates[qid, arm]}
                    write_json(transcript_path(out, arm, qid), result)
                except Exception as exc:
                    error = {"query_id": qid, "arm": arm, "status": "failed", "type": type(exc).__name__, "message": safe_error(exc, key), "request_id": getattr(exc, "request_id", None), "utc": utc()}
                    failures.append(error)
                    write_json(out / "failures.json", failures)
                    print(json.dumps({"stage": "transcription_failed", "query_id": qid, "arm": arm, "type": type(exc).__name__}), flush=True)
                    continue
            transcripts[qid, arm] = result
            cost = (result.get("raw_response", {}).get("usage") or {}).get("cost")
            recorded_cost += float(cost) if cost is not None else result.get("estimated_cost_usd", estimates[qid, arm])
            if recorded_cost > args.budget_usd:
                raise RuntimeError("Recorded/estimated spend exceeds the pilot guard")
            print(json.dumps({"stage": "transcribed", "query_id": qid, "arm": arm}), flush=True)
        account = client.get("/key", cast_to=dict)["data"]
        billing.update(after={k: account.get(k) for k in ["usage", "limit", "limit_remaining"]}, recorded_or_estimated_usd=recorded_cost)
        write_json(out / "billing.json", billing)
    rows = []
    token_cache = {}
    for audio in audio_rows:
        qid = audio["query_id"]
        for arm in MODELS:
            transcript = transcripts.get((qid, arm))
            if transcript is None:
                continue
            text = transcript["transcript"]
            start = time.perf_counter()
            vector = encoder.encode(text)
            encode_ms = 1000 * (time.perf_counter() - start)
            vector = np.asarray(vector, dtype=np.float32)
            vector /= np.linalg.norm(vector)
            start = time.perf_counter()
            similarities = retriever.embeddings @ vector
            order = np.argsort(-similarities, kind="stable")
            sorted_scores = similarities[order].astype(np.float64)
            ranking_ms = 1000 * (time.perf_counter() - start)
            if not np.isfinite(sorted_scores).all() or not np.all(np.diff(sorted_scores) <= 0):
                raise RuntimeError("Invalid cosine ranking")
            matching = int(np.count_nonzero(sorted_scores >= TAU))
            for rule, count in [("top5", 5), ("cutoff", max(1, matching))]:
                ids = [retriever.record_ids[int(i)] for i in order[:count]]
                for rid in ids:
                    if rid not in token_cache:
                        token_cache[rid] = len(tokenizer.encode(data.record_text(rid), add_special_tokens=False).ids)
                hits = len(set(ids) & set(qrels[qid]))
                recall = recall_at_k(ids, qrels[qid], count)
                if recall != hits / len(qrels[qid]) or len(set(ids)) != count:
                    raise RuntimeError("Retrieval bookkeeping validation failed")
                rows.append({"query_id": qid, "arm": arm, "model": MODELS[arm], "rule": rule, "recall": recall, "relevant_hits": hits, "total_relevant": len(qrels[qid]),
                             "context_tokens": sum(token_cache[rid] for rid in ids), "documents": count, "record_ids": ids, "selected_scores": sorted_scores[:count].tolist(),
                             "top_similarity": float(sorted_scores[0]), "fallback": rule == "cutoff" and matching == 0,
                             "transcript_tokens": len(tokenizer.encode(text, add_special_tokens=False).ids), "encoding_ms": encode_ms, "ranking_ms": ranking_ms})
    (out / "retrieval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write_json(out / "document-token-cache.json", token_cache)
    complete = summarize(rows, selected, out, failures, billing)
    write_json(out / "completion.json", {"status": "complete" if complete else "partial", "utc": utc(), "slurm_job_id": os.environ["SLURM_JOB_ID"], "transcripts": len(transcripts), "retrieval_cells": len(rows)})
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
