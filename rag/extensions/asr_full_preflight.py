"""Read-only Slurm inventory and OpenRouter spend preflight; no inference."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import urllib.request
import wave

if not os.getenv("SLURM_JOB_ID") or not socket.getfqdn().endswith(".inf.ed.ac.uk"):
    raise SystemExit("Run in Edinburgh Slurm")
os.umask(0o077)
root = Path.home() / "edin-mls-asr-pilot"
sys.path.insert(0, str(root / "rag"))
from src.data import CourseData

state = Path.home() / ".cache/mls-asr-full"
state.mkdir(parents=True, exist_ok=True)
data_dir = root / "rag/data/rag_corpus"
data = CourseData(data_dir)
manifest = data.load_audio_manifest()
pilot = json.loads((root / "rag/results/asr-four-model-pilot-20261116/selection.json").read_text())
splits = sorted({row["split"] for row in manifest})
qrels = {split: data.load_qrels(split) for split in splits}
rows = []
seen = set()
for item in sorted(manifest, key=lambda row: (row["split"], row["query_id"])):
    qid, split = item["query_id"], item["split"]
    if qid in seen or not qrels[split].get(qid):
        raise RuntimeError("Duplicate query or missing positive qrels: " + qid)
    seen.add(qid)
    if not set(qrels[split][qid]).issubset(data.records):
        raise RuntimeError("Positive qrel absent from corpus")
    path = (data_dir / item["audio_file"]).resolve()
    path.relative_to(data_dir.resolve())
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != item["sha256"]:
        raise RuntimeError("Audio hash mismatch")
    with wave.open(str(path), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (16000, 1, 2, "NONE"):
            raise RuntimeError("Unexpected audio format")
        seconds = audio.getnframes() / audio.getframerate()
    rows.append({"query_id": qid, "split": split, "audio_file": item["audio_file"],
                 "audio_sha256": digest, "seconds": seconds})
selection = {"audio": rows, "query_ids": [r["query_id"] for r in rows],
             "pilot_ids": pilot["selected_ids"], "splits": splits,
             "counts_by_split": {s: sum(r["split"] == s for r in rows) for s in splits},
             "manifest_sha256": hashlib.sha256((data_dir / "student_audio_manifest.jsonl").read_bytes()).hexdigest()}
target = state / "selection.json"
if target.exists() and json.loads(target.read_text()) != selection:
    raise RuntimeError("Frozen full selection changed")
target.write_text(json.dumps(selection, indent=2) + "\n")
secret_file = Path.home() / ".config/edin-mls/openrouter.env"
key = next(line.split("=", 1)[1].strip().strip("\"'") for line in secret_file.read_text().splitlines()
           if line.startswith("OPENROUTER_API_KEY="))
def get(suffix):
    req = urllib.request.Request("https://openrouter.ai/api/v1/" + suffix,
                                 headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)
catalogue = get("models?output_modalities=transcription")
account = get("key")["data"]
models = {m["id"]: m for m in catalogue["data"]}
ids = {"whisper": "openai/whisper-large-v3", "mai": "microsoft/mai-transcribe-2",
       "voxtral": "mistralai/voxtral-small-transcribe-3-0"}
rates = {arm: float(models[model]["pricing"]["prompt"]) / (3600 if arm == "mai" else 1)
         for arm, model in ids.items()}
remaining = [r for r in rows if r["query_id"] not in pilot["selected_ids"]]
estimate = sum(max(row["seconds"], 1) * rate for row in remaining for rate in rates.values())
result = {"status": "pass", "slurm_job_id": os.environ["SLURM_JOB_ID"],
          "queries": len(rows), "counts_by_split": selection["counts_by_split"],
          "pilot_cached_queries": len(pilot["selected_ids"]), "new_queries": len(remaining),
          "planned_new_api_requests": len(remaining) * 3,
          "audio_seconds_total": sum(r["seconds"] for r in rows),
          "audio_seconds_remaining": sum(r["seconds"] for r in remaining),
          "rates_usd_per_second": rates, "estimated_new_usd": estimate,
          "conservative_reserve_usd": 2 * estimate + 0.02,
          "key_usage_before_usd": account.get("usage"), "key_limit_remaining_usd": account.get("limit_remaining"),
          "normalizer_sha256": hashlib.sha256((root / "rag/extensions/transcript_normalization.py").read_bytes()).hexdigest(),
          "python": sys.version, "packages": {n: {"version": importlib.metadata.version(n),
              "path": importlib.util.find_spec(m).origin} for n, m in [("numpy", "numpy"),
              ("sentence-transformers", "sentence_transformers"), ("openai", "openai"),
              ("httpx2", "httpx2"), ("tokenizers", "tokenizers")]}}
(state / "openrouter-catalogue-preflight.json").write_text(json.dumps(catalogue, indent=2) + "\n")
(state / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
if result["conservative_reserve_usd"] > 1:
    raise RuntimeError("Estimated reserve exceeds USD 1 full-run guard")
