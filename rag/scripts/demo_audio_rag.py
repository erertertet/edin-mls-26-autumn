"""Live MP3 -> Fuji3 CUDA ASR -> three retrieval policies -> answer -> judge.

The local launcher needs only Python and OpenSSH. All model execution happens
on Fuji3, using the existing ~/ragexp environment. See demo_audio_rag.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import uuid

PRESETS = {
    "value-investing": {
        "query_id": "q_92cf5957e0e552dc",
        "question": "Are there any recommended software tools for value investing strategies?",
        "historical": {"top2": [2, 1.0], "absolute": [1, 0.0], "relative": [3, 0.5]},
    },
}
POLICIES = ("top2", "absolute", "relative")
GENERATOR_SYSTEM = (
    "Use the supplied evidence to answer the user's question. "
    "Do not invent facts that are unsupported by the evidence."
)


def selected_count(scores, policy, tau=0.8925000000000001, delta=0.009000000000000001):
    """Sorted full-corpus similarities, inclusive thresholds, top-1 fallback."""
    if len(scores) == 0 or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("Expected nonempty finite similarities")
    if policy == "top2":
        return min(2, len(scores)), False
    if policy not in ("absolute", "relative"):
        raise ValueError("Unknown policy")
    threshold = tau if policy == "absolute" else float(scores[0]) - delta
    count = 0
    for score in scores:
        if float(score) < threshold:
            break
        count += 1
    return max(1, count), count == 0


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def reference_text(args):
    if args.gold_text is not None:
        text = args.gold_text.strip()
    elif args.gold_file:
        text = args.gold_file.read_text(encoding="utf-8").strip()
    else:
        return None
    if not text:
        raise ValueError("Reference evidence is empty")
    return text


def ssh_args(host):
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", host) or host.startswith("-"):
        raise ValueError("Use a hostname or SSH config alias for --host")
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host]


def remote_capture(host, command):
    result = subprocess.run(ssh_args(host) + [command], capture_output=True, timeout=40)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout.decode("utf-8").strip()


def launcher(args):
    if args.list_presets:
        print(json.dumps(PRESETS, indent=2))
        return 0
    preset = PRESETS.get(args.preset)
    if args.audio and not args.audio.is_file():
        raise ValueError(f"Audio file not found: {args.audio}")
    if not args.audio and not args.preset:
        raise ValueError("Provide an audio file, or --preset value-investing for the supplied recording")
    if args.query_id and preset and args.query_id != preset["query_id"]:
        raise ValueError("--query-id conflicts with the preset")
    gold_text = reference_text(args)
    if gold_text and (args.query_id or preset):
        raise ValueError("Choose either reference text or a benchmark query/preset")
    qid = args.query_id or (preset["query_id"] if preset else None)
    if args.budget <= 0:
        raise ValueError("--budget must be positive")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    out = (args.out or Path(__file__).resolve().parents[1] / "results" / ("live-demo-" + run_id)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Ask the remote login for its actual home; never interpolate a local HOME.
    home = remote_capture(args.host, 'printf "%s" "$HOME"')
    repo = args.remote_repo or home + "/ragexp/edin-mls-26-autumn"
    activate = args.remote_activate or home + "/ragexp/activate.sh"
    remote = home + "/ragexp/live-demos/" + run_id
    if not all(re.fullmatch(r"/[A-Za-z0-9_./-]+", x) for x in (home, repo, activate, remote)):
        raise ValueError("Remote paths must be absolute and use letters, numbers, /, _, - or .")
    config = {
        "repo": repo, "query_id": qid, "split": args.split,
        "preset": args.preset, "expected_question": preset["question"] if preset else None,
        "audio": "input" + args.audio.suffix.lower() if args.audio else None,
        "gold_file": "gold.txt" if gold_text else None,
        "model": args.model, "judge": args.judge, "budget": args.budget,
        "cuda": args.cuda, "gpu": args.gpu, "retrieval_only": args.retrieval_only,
        "tau": 0.8925000000000001, "delta": 0.009000000000000001,
    }
    write_json(out / "config.json", config)
    print(f"Run directory: {out}\nRemote directory: {args.host}:{remote}", flush=True)
    if preset:
        print("Read aloud: " + preset["question"], flush=True)
    remote_capture(args.host, "mkdir -p " + shlex.quote(remote))
    uploads = [(Path(__file__).resolve(), "demo_audio_rag.py"), (out / "config.json", "config.json")]
    if args.audio:
        uploads.append((args.audio.resolve(), config["audio"]))
    if gold_text:
        (out / "gold.txt").write_text(gold_text, encoding="utf-8")
        uploads.append((out / "gold.txt", "gold.txt"))
    for source, name in uploads:
        print(f"UPLOAD {source.name} -> {name}", flush=True)
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(source), f"{args.host}:{remote}/{name}"],
                       check=True, timeout=180)
    command = " && ".join([
        "source " + shlex.quote(activate),
        "cd " + shlex.quote(remote),
        "python -u demo_audio_rag.py --worker config.json",
    ])
    process = subprocess.Popen(ssh_args(args.host) + ["bash -lc " + shlex.quote(command)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (out / "console.log").open("w", encoding="utf-8") as log:
        try:
            for line in iter(process.stdout.readline, b""):
                text = line.decode("utf-8", errors="replace")
                print(text, end="", flush=True)
                log.write(text)
                log.flush()
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            print("Interrupted SSH. Inspect the remote run before retrying; in-flight calls may finish.")
            return 130
        finally:
            process.stdout.close()
    for name in ("events.jsonl", "results.json", "error.json"):
        result = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{args.host}:{remote}/{name}", str(out / name)],
                                capture_output=True, timeout=60)
        if result.returncode and name == "results.json" and code == 0:
            raise RuntimeError("Run completed but results download failed; use the printed remote directory")
    print(f"\nSaved console.log, events.jsonl and results.json in {out}")
    return code


class Logger:
    def __init__(self, path):
        self.path = path
        self.start = time.perf_counter()

    def emit(self, stage, message, **data):
        elapsed = time.perf_counter() - self.start
        print(f"\n[{elapsed:8.2f}s] {stage}\n{message}", flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"elapsed_s": elapsed, "stage": stage, "message": message, **data},
                               ensure_ascii=False) + "\n")


class MeteredAPI:
    def __init__(self, config, log):
        from openai import OpenAI
        import urllib.request
        key = os.getenv("OPENROUTER_API_KEY") or os.getenv("ELM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("No API key in the remote environment or rag/.env")
        base = os.getenv("ELM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        # This demo relies on OpenRouter's usage.cost and public model pricing.
        if base != "https://openrouter.ai/api/v1":
            raise ValueError("This demo expects the existing OpenRouter endpoint")
        with urllib.request.urlopen(base + "/models", timeout=30) as response:
            models = {m["id"]: m for m in json.load(response)["data"]}
        self.models = {name: models[name] for name in (config["model"], config["judge"])}
        self.client = OpenAI(api_key=key, base_url=base, timeout=180, max_retries=0)
        self.config, self.log, self.spent = config, log, 0.0

    def call(self, policy, role, messages):
        model = self.config["judge" if role == "judge" else "model"]
        output_cap = 768 if role == "judge" else 2048
        # Byte count plus overhead is deliberately conservative for text tokenisation.
        input_bound = sum(len(m["content"].encode("utf-8")) + 256 for m in messages)
        info = self.models[model]
        if input_bound + output_cap > int(info["context_length"]):
            raise ValueError("Context preflight failed; evidence will not be silently truncated")
        price = info["pricing"]
        reservation = input_bound * float(price["prompt"]) + output_cap * float(price["completion"])
        reservation += float(price.get("request", 0) or 0)
        if not math.isfinite(reservation) or reservation < 0:
            raise ValueError("Invalid API pricing")
        if self.spent + reservation > self.config["budget"]:
            raise ValueError(f"Budget preflight: ${self.spent:.4f} spent + ${reservation:.4f} reserved exceeds budget")
        self.log.emit(role.upper(), f"{policy}: calling {model}", reserved_usd=reservation)
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model, messages=messages, temperature=0 if role == "judge" else 0.1,
            max_tokens=output_cap, extra_body={"reasoning": {"enabled": False}},
        )
        usage = response.usage.model_dump() if response.usage else {}
        raw = response.choices[0].message.content or ""
        cost = usage.get("cost")
        record = {"request_id": response.id, "model": response.model, "role": role, "policy": policy,
                  "usage": usage, "latency_s": time.perf_counter() - start,
                  "finish_reason": response.choices[0].finish_reason, "content": raw}
        self.log.emit("API COMPLETE", f"{role}: {usage.get('prompt_tokens')} input tokens, cost ${cost}", **record)
        if cost is None or not math.isfinite(float(cost)) or float(cost) < 0:
            raise ValueError("API usage cost missing/invalid; stopping before further calls")
        self.spent += float(cost)
        if not raw.strip():
            raise ValueError("API returned empty content; response metadata was saved")
        return raw.strip(), record


def worker(config_path):
    # Prevent accidental local CUDA use even if someone manually passes --worker.
    if os.name != "posix" or not socket.gethostname().split(".")[0].startswith("fuji3"):
        raise RuntimeError("The worker must run on Fuji3; use the local SSH launcher")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = config_path.resolve().parent
    log = Logger(out / "events.jsonl")
    repo = Path(config["repo"])
    sys.path.insert(0, str(repo / "rag"))
    if config["gpu"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    from dotenv import load_dotenv
    load_dotenv(repo / "rag/.env")
    import numpy as np
    from src.asr_adapter import ASRAdapter
    from src.encoder import LocalEncoder
    from src.retriever import ExactDenseRetriever
    from src.data import CourseData
    from src.context_builder import SimpleContextBuilder
    from src.evaluation.answer_judge import Judge, JUDGE_SYSTEM_PROMPT, build_public_gold
    data_dir = repo / "rag/data/rag_corpus"
    data = CourseData(data_dir)
    audio = out / config["audio"] if config["audio"] else None
    if audio is None:
        item = next(x for x in data.audio_items(config["split"]) if x["query_id"] == config["query_id"])
        audio = data_dir / item["audio_file"]
    log.emit("AUDIO", f"Input: {audio.name}\nCUDA kernels: {config['cuda']}",
             sha256=hashlib.sha256(audio.read_bytes()).hexdigest(), host=socket.gethostname())
    wav = out / "decoded.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(audio),
                    "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
    import wave
    with wave.open(str(wav)) as stream:
        duration = stream.getnframes() / stream.getframerate()
    if not 0 < duration <= 120:
        raise ValueError("Use a spoken question between 0 and 120 seconds")
    log.emit("AUDIO READY", f"Decoded {duration:.2f} seconds to 16 kHz mono PCM16 WAV")
    if config["gold_file"]:
        text = (out / config["gold_file"]).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError("Reference evidence is empty")
        gold = [{"record_id": "user-reference", "text": text}]
    elif config["query_id"]:
        gold = build_public_gold(data_dir, config["split"])[config["query_id"]]["gold_documents"]
    else:
        gold = []
    log.emit("REFERENCE", (f"Independent judge evidence: {len(gold)} documents. Used only by the judge."
                           if gold else "No reference supplied. Running all three answer pipelines without Gemini scoring."),
             query_id=config["query_id"], intended_question=config["expected_question"])
    api = None if config["retrieval_only"] else MeteredAPI(config, log)
    log.emit("LOADING", "Loading encoder on Fuji3 CPU, corpus vectors and CUDA ASR model once")
    encoder = LocalEncoder(repo / "rag/models/enc_a", device="cpu")
    retriever = ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy",
                                    data_dir / "embeddings/corpus_enc_a_ids.json")
    builder = SimpleContextBuilder()
    results = {"config": config, "audio_seconds": duration, "policies": [], "complete": False}
    with ASRAdapter(asr_root=repo / "hw1-asr", cuda_groups=config["cuda"], max_new_tokens=256) as asr:
        asr.prepare()
        log.emit("READY", "Models loaded. Repeating ASR, encoding, retrieval and generation for each policy.")
        for policy in POLICIES:
            start = time.perf_counter()
            log.emit("POLICY", f"{policy}: top-2 / absolute tau={config['tau']:.4f} / relative delta={config['delta']:.3f}")
            transcript = asr.transcribe(wav)
            asr_s = time.perf_counter() - start
            log.emit("ASR TRANSCRIPT", transcript, policy=policy, asr_s=asr_s)
            embedding_start = time.perf_counter()
            vector = encoder.encode(transcript)
            vector = np.asarray(vector, dtype=np.float32)
            vector /= np.linalg.norm(vector)
            scores = retriever.embeddings @ vector
            order = np.argsort(-scores, kind="stable")
            ranked_scores = scores[order]
            k, fallback = selected_count(ranked_scores, policy, config["tau"], config["delta"])
            ids = [retriever.record_ids[int(i)] for i in order[:k]]
            retrieval_s = time.perf_counter() - embedding_start
            documents = [{"rank": i + 1, "record_id": rid, "similarity": float(ranked_scores[i]),
                          "text": data.record_text(rid)} for i, rid in enumerate(ids)]
            log.emit("RETRIEVAL", f"{policy}: {k} documents from {len(scores)} ranked records; fallback={fallback}",
                     policy=policy, documents=k, fallback=fallback, encoding_retrieval_s=retrieval_s)
            for doc in documents:
                log.emit("DOCUMENT", f"Rank {doc['rank']}  {doc['record_id']}  cosine={doc['similarity']:.6f}\n{doc['text']}",
                         policy=policy, **doc)
            row = {"policy": policy, "transcript": transcript, "k": k, "fallback": fallback,
                   "documents": documents, "asr_s": asr_s, "encoding_retrieval_s": retrieval_s}
            if api:
                prompt = builder.build(transcript, ids, data)["prompt"]
                answer, record = api.call(policy, "generator", [
                    {"role": "system", "content": GENERATOR_SYSTEM}, {"role": "user", "content": prompt}])
                row.update(answer=answer, generator=record, audio_to_answer_s=time.perf_counter() - start)
                log.emit("FINAL ANSWER", answer, policy=policy)
                # Save the generated answer before the judge call, including on failure.
                results["policies"].append(row)
                write_json(out / "results.json", results)
                if not gold:
                    log.emit("JUDGE SKIPPED", "No reference evidence supplied. Final answer is complete.", policy=policy)
                    continue
                payload = {"question": transcript, "candidate_answer": answer,
                           "gold_evidence": [{"record_id": d["record_id"], "text": d["text"]} for d in gold]}
                judge_prompt = ('Return JSON exactly as {"correctness":0,"grounding":0,"completeness":0,'
                                '"explanation":"brief reason"}.\n\nINPUT:\n' + json.dumps(payload, ensure_ascii=False))
                for attempt in range(2):
                    raw, record = api.call(policy, "judge", [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": judge_prompt}])
                    try:
                        judgement = Judge.parse(raw)
                        break
                    except (ValueError, KeyError, TypeError):
                        if attempt:
                            raise ValueError("Gemini returned invalid score JSON twice; raw outputs are saved")
                        log.emit("JUDGE RETRY", "Invalid score JSON. Retrying once with the same blind payload.")
                score = sum(judgement[d] for d in ("correctness", "grounding", "completeness")) / 12
                row.update(judgement=judgement, score=score, judge=record)
                log.emit("GEMINI JUDGEMENT", f"Final score: {score:.3f}/1\n" + json.dumps(judgement, indent=2), policy=policy)
            else:
                results["policies"].append(row)
            write_json(out / "results.json", results)
    results.update(complete=True, reported_api_usd=api.spent if api else 0,
                   distinct_selection_sets=len({tuple(d["record_id"] for d in r["documents"]) for r in results["policies"]}))
    write_json(out / "results.json", results)
    lines = ["Policy       Docs   Input tokens   Judge score   Audio-to-answer s"]
    for r in results["policies"]:
        lines.append(f"{r['policy']:12} {r['k']:4}   {r.get('generator', {}).get('usage', {}).get('prompt_tokens', '-'):>12}"
                     f"   {str(round(r['score'], 3)) if 'score' in r else '-':>11}   {r.get('audio_to_answer_s', 0):.2f}")
    lines.append(f"Distinct retrieval sets: {results['distinct_selection_sets']}/3. Reported API cost: ${results['reported_api_usd']:.6f}")
    log.emit("COMPARISON", "\n".join(lines))
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path, nargs="?", help="Local MP3/WAV/M4A recording to upload")
    p.add_argument("--preset", choices=PRESETS)
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--query-id", help="Benchmark qrels ID for the recorded question")
    p.add_argument("--split", choices=("dev", "practice"), default="practice")
    reference = p.add_mutually_exclusive_group()
    reference.add_argument("--gold-text", help="Optional inline reference evidence to enable Gemini scoring")
    reference.add_argument("--gold-file", type=Path, help="Optional UTF-8 reference file to enable Gemini scoring")
    p.add_argument("--host", default="fuji3", help="SSH config alias, e.g. fuji3 or wgfuji3")
    p.add_argument("--remote-repo", help="Default: remote ~/ragexp/edin-mls-26-autumn")
    p.add_argument("--remote-activate", help="Default: remote ~/ragexp/activate.sh")
    p.add_argument("--out", type=Path, help="New local output directory")
    p.add_argument("--gpu", help="Remote GPU index/UUID; default uses activate.sh")
    p.add_argument("--cuda", default="all", help="ASR CUDA groups; default all custom kernels")
    p.add_argument("--model", default="mistralai/mistral-small-2603")
    p.add_argument("--judge", default="google/gemini-2.5-flash")
    p.add_argument("--budget", type=float, default=0.50, help="API spend guard in USD, including Gemini")
    p.add_argument("--retrieval-only", action="store_true", help="Run audio/retrieval only, with no paid calls")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return p


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args()
    try:
        raise SystemExit(worker(args.worker) if args.worker else launcher(args))
    except Exception as exc:
        # Provider exceptions can contain raw response payloads. Keep only type/status.
        message = str(exc) if isinstance(exc, (ValueError, FileNotFoundError, RuntimeError)) else type(exc).__name__
        print(f"ERROR: {message}", file=sys.stderr)
        if args.worker:
            write_json(args.worker.resolve().parent / "error.json", {"type": type(exc).__name__, "message": message})
        raise SystemExit(1)
