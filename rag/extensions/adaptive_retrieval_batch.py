"""Resumable parallel RAG comparison, with dev-only threshold tuning.

Requires a locked partition.json and queries.jsonl containing prepared dev
queries. The run directory is created by the experiment setup; credentials stay
in rag/.env. Independent API calls run in threads; ASR uses one persistent worker.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_retrieval_pilot import (
    ROOT, ARMS, PLANNER_PROMPT, selected_count, ASRAdapter, LocalEncoder,
    ExactDenseRetriever, CourseData, SimpleContextBuilder, recall_at_k,
)
from src.evaluation.answer_judge import Judge, JUDGE_SYSTEM_PROMPT, build_public_gold

GENERATOR_SYSTEM = "Use the supplied evidence to answer the user's question. Do not invent facts that are unsupported by the evidence."


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class Run:
    def __init__(self, args):
        self.args = args
        self.out = args.out
        self.plan = json.loads((self.out / "partition.json").read_text())
        self.data = CourseData(ROOT / "data/rag_corpus")
        self.items = {item["query_id"]: item for item in self.data.load_audio_manifest()}
        self.query_cache = {row["query_id"]: row for row in read_rows(self.out / "queries.jsonl")}
        self.candidates = {(row["query_id"], row["arm"]): row for row in read_rows(self.out / "candidates.jsonl")}
        self.results = {(row["query_id"], row["arm"]): row for row in read_rows(self.out / "results.jsonl")}
        self.planners = {row["query_id"]: row for row in read_rows(self.out / "planners.jsonl")}
        self.requests = read_rows(self.out / "requests.jsonl")
        self.spent = sum(float(row["usage"]["cost"]) for row in self.requests)
        self.reserved = 0.0
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.local = threading.local()
        self.started = time.perf_counter()
        self.errors = []
        self.phases = {
            "dev_tuning": self.plan["tuning_query_ids"],
            "heldout_test": self.plan["heldout_test_query_ids"],
            "seen_exploratory": self.plan["exploratory_seen_query_ids"],
        }
        self.total = sum(len(ids) for ids in self.phases.values()) * len(ARMS)
        self.qrels = {split: self.data.load_qrels(split) for split in ["practice", "dev"]}
        self.gold = {split: build_public_gold(ROOT / "data/rag_corpus", split) for split in ["practice", "dev"]}
        self.retriever = ExactDenseRetriever(ROOT / "data/rag_corpus/embeddings/corpus_enc_a.npy", ROOT / "data/rag_corpus/embeddings/corpus_enc_a_ids.json")
        self.builder = SimpleContextBuilder()
        self.encoder = None
        self.asr = None
        self.freeze_config()

    def append(self, name, row):
        with self.lock:
            with (self.out / name).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def freeze_config(self):
        config = {
            "generator": self.args.model, "planner": self.args.model, "judge": self.args.judge,
            "arms": ARMS, "api_workers": self.args.workers, "budget_usd": self.args.budget,
            "temperature_generator": 0.1, "temperature_planner": 0, "temperature_judge": 0,
            "max_tokens_generator": 2048, "max_tokens_planner": 64, "max_tokens_judge": 768,
            "reasoning_enabled": False, "planner_prompt": PLANNER_PROMPT,
            "selection_objective": "maximum dev macro qrels F1; ties prefer fewer mean documents",
            "absolute_similarity_grid": np.linspace(0.75, 0.99, 97).tolist(),
            "relative_gap_grid": np.linspace(0.0, 0.08, 81).tolist(),
            "min_documents": 1, "max_documents": 20, "asr_cuda": os.getenv("ASR_CUDA"),
            "gpu_uuid": os.getenv("CUDA_VISIBLE_DEVICES"),
            "partition_sha256": hashlib.sha256((self.out / "partition.json").read_bytes()).hexdigest(),
            "notes": [
                "Dev is used for threshold tuning; test is a newly locked, previously unused practice subset.",
                "No official test split is supplied. Seen practice queries are reported separately.",
                "Same generator/judge and decoding settings for all arms. Raw query-only planner is not tuned.",
                "ASR and CPU encoding are executed once per unique query and shared across policy arms.",
                "service_e2e_ms sums measured warm ASR/encoding with branch service time; it excludes queue wait and is reconstructed.",
                "observed_audio_to_answer_ms is available only for freshly transcribed queries; it includes API queue wait.",
                "API concurrency affects latency; compare policies within this run, not directly against the sequential pilot.",
            ],
        }
        path = self.out / "config.json"
        if path.exists():
            if json.loads(path.read_text()) != config:
                raise RuntimeError("Run configuration changed; use a fresh output directory")
        else:
            atomic_json(path, config)
        self.config = config

    def tune(self):
        path = self.out / "frozen-parameters.json"
        if path.exists():
            self.parameters = json.loads(path.read_text())
            return
        ids = self.phases["dev_tuning"]
        if not all(qid in self.query_cache for qid in ids):
            raise RuntimeError("Prepare every dev transcript and retrieval score before tuning")
        rows = []
        for arm, values in [
            ("absolute_radius", self.config["absolute_similarity_grid"]),
            ("relative_gap", self.config["relative_gap_grid"]),
        ]:
            for value in values:
                precisions, recalls, f1s, counts = [], [], [], []
                for qid in ids:
                    query = self.query_cache[qid]
                    k, _ = selected_count(query["similarities"], arm, value if arm == "absolute_radius" else 0, value if arm == "relative_gap" else 0)
                    chosen = query["record_ids"][:k]
                    relevant = set(self.qrels["dev"][qid])
                    hits = len(set(chosen) & relevant)
                    precision, recall = hits / k, hits / len(relevant)
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
                    precisions.append(precision); recalls.append(recall); f1s.append(f1); counts.append(k)
                rows.append({"arm": arm, "value": value, "macro_precision": statistics.mean(precisions), "macro_recall": statistics.mean(recalls), "macro_f1": statistics.mean(f1s), "mean_documents": statistics.mean(counts)})
        winners = {arm: max((row for row in rows if row["arm"] == arm), key=lambda row: (row["macro_f1"], -row["mean_documents"])) for arm in ["absolute_radius", "relative_gap"]}
        parameters = {
            "tau": winners["absolute_radius"]["value"], "delta": winners["relative_gap"]["value"],
            "objective": self.config["selection_objective"], "dev_queries": len(ids),
            "selected": winners, "frozen_at": datetime.now(timezone.utc).isoformat(),
            "test_queries_evaluated_at_freeze": 0,
        }
        assert not set(ids) & set(self.phases["heldout_test"])
        assert not any(row["phase"] == "heldout_test" for row in self.results.values())
        atomic_json(self.out / "tuning-grid.json", rows)
        atomic_json(path, parameters)
        self.parameters = parameters
        print("FROZEN", json.dumps(parameters), flush=True)

    def client(self):
        if not hasattr(self.local, "client"):
            self.local.client = OpenAI(
                api_key=os.environ.get("ELM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("ELM_BASE_URL", "https://openrouter.ai/api/v1"),
                timeout=120, max_retries=0,
            )
        return self.local.client

    def call(self, phase, qid, arm, role, messages):
        reservation = {"generator": 0.06, "planner": 0.01, "judge": 0.40}[role]
        for attempt in range(3):
            with self.lock:
                if self.stop.is_set() or self.spent + self.reserved + reservation > self.args.budget:
                    self.stop.set()
                    raise RuntimeError("Budget guard or fatal-stop flag reached")
                self.reserved += reservation
            started = time.perf_counter()
            try:
                kwargs = {
                    "model": self.args.judge if role == "judge" else self.args.model,
                    "messages": messages, "temperature": 0.1 if role == "generator" else 0,
                    "max_tokens": {"generator": 2048, "planner": 64, "judge": 768}[role],
                    "extra_body": {"reasoning": {"enabled": False}},
                }
                if role == "planner":
                    kwargs["response_format"] = {"type": "json_object"}
                response = self.client().chat.completions.create(**kwargs)
                usage = response.usage.model_dump() if response.usage else {}
                if usage.get("cost") is None:
                    raise RuntimeError("OpenRouter cost missing; stop to avoid untracked spending")
                record = {
                    "phase": phase, "query_id": qid, "arm": arm, "role": role,
                    "attempt": attempt + 1, "id": response.id, "model": response.model,
                    "usage": usage, "latency_ms": (time.perf_counter() - started) * 1000,
                    "finish_reason": response.choices[0].finish_reason,
                }
                with self.lock:
                    self.spent += float(usage["cost"])
                    self.requests.append(record)
                    self.append("requests.jsonl", record)
                return response, record
            except Exception as error:
                status = getattr(error, "status_code", None)
                self.append("api-errors.jsonl", {"phase": phase, "query_id": qid, "arm": arm, "role": role, "attempt": attempt + 1, "error_type": type(error).__name__, "status": status})
                if status in [401, 402, 403] or (isinstance(error, RuntimeError) and "cost missing" in str(error)):
                    self.stop.set()
                    raise
                if attempt == 2:
                    raise
                time.sleep((2 ** attempt) + random.random())
            finally:
                with self.lock:
                    self.reserved -= reservation

    def prepare_query(self, qid):
        if qid in self.query_cache:
            return self.query_cache[qid], None
        if self.encoder is None:
            self.encoder = LocalEncoder(ROOT / "models/enc_a", device="cpu")
        if self.asr is None:
            self.asr = ASRAdapter()
            self.asr.prepare()
            warm = self.asr.transcribe(ROOT / "data/rag_corpus" / self.items[qid]["audio_file"])
            self.encoder.encode(warm)
        query_start = time.perf_counter()
        started = time.perf_counter()
        transcript = self.asr.transcribe(ROOT / "data/rag_corpus" / self.items[qid]["audio_file"])
        asr_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        vector = self.encoder.encode(transcript)
        embedding_ms = (time.perf_counter() - started) * 1000
        retrieval = self.retriever.retrieve(vector, k=20)
        query = {
            "query_id": qid, "source_split": self.items[qid]["split"], "transcript": transcript,
            "asr_ms": asr_ms, "embedding_ms": embedding_ms, "vector": vector.tolist(),
            "record_ids": retrieval["record_ids"], "similarities": retrieval["scores"],
            "retrieval_ms": retrieval["latency_ms"], "asr_cuda": os.getenv("ASR_CUDA"),
        }
        self.query_cache[qid] = query
        self.append("queries.jsonl", query)
        return query, query_start

    def task(self, phase, query, arm, submitted, query_start):
        qid = query["query_id"]
        worker_start = time.perf_counter()
        key = (qid, arm)
        with self.lock:
            candidate = self.candidates.get(key)
        if candidate is None:
            planner_ms, fallback = 0.0, False
            k = 5 if arm == "fixed5" else 20
            if arm == "llm_depth":
                with self.lock:
                    plan = self.planners.get(qid)
                if plan is None:
                    response, record = self.call(phase, qid, arm, "planner", [{"role": "system", "content": PLANNER_PROMPT}, {"role": "user", "content": query["transcript"]}])
                    raw = response.choices[0].message.content or ""
                    try:
                        k = json.loads(raw)["k"]
                        if type(k) is not int or not 1 <= k <= 20:
                            raise ValueError("Invalid predicted k")
                    except (ValueError, KeyError, TypeError):
                        k, fallback = 5, True
                    plan = {"query_id": qid, "k": k, "fallback": fallback, "raw": raw, "planner_ms": record["latency_ms"]}
                    with self.lock:
                        self.planners[qid] = plan
                        self.append("planners.jsonl", plan)
                k, fallback, planner_ms = plan["k"], plan["fallback"], plan["planner_ms"]
            started = time.perf_counter()
            retrieval = self.retriever.retrieve(np.asarray(query["vector"], dtype=np.float32), k=k)
            if arm in ["absolute_radius", "relative_gap"]:
                k, fallback = selected_count(retrieval["scores"], arm, self.parameters["tau"], self.parameters["delta"])
            ids = retrieval["record_ids"][:k]
            built = self.builder.build(query["transcript"], ids, self.data)
            retrieval_context_ms = (time.perf_counter() - started) * 1000
            response, record = self.call(phase, qid, arm, "generator", [{"role": "system", "content": GENERATOR_SYSTEM}, {"role": "user", "content": built["prompt"]}])
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                raise RuntimeError("Empty answer")
            generation_end = time.perf_counter()
            relevant = set(self.qrels[query["source_split"]][qid])
            hits = len(set(ids) & relevant)
            precision, recall = hits / k, hits / len(relevant)
            candidate = {
                "phase": phase, "query_id": qid, "source_split": query["source_split"],
                "arm": arm, "k": k, "fallback": fallback, "transcript": query["transcript"],
                "answer": answer, "record_ids": ids, "similarities": retrieval["scores"][:k],
                "prompt_characters": len(built["prompt"]), "recall": recall, "precision": precision,
                "retrieval_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
                "asr_ms": query["asr_ms"], "embedding_ms": query["embedding_ms"],
                "planner_ms": planner_ms, "retrieval_context_ms": retrieval_context_ms,
                "generation_ms": record["latency_ms"], "queue_ms": (worker_start - submitted) * 1000,
                "service_e2e_ms": query["asr_ms"] + query["embedding_ms"] + planner_ms + retrieval_context_ms + record["latency_ms"],
                "branch_wall_ms": (generation_end - submitted) * 1000,
                "observed_audio_to_answer_ms": (generation_end - query_start) * 1000 if query_start else None,
                "finish_reason": record["finish_reason"],
            }
            with self.lock:
                self.candidates[key] = candidate
                self.append("candidates.jsonl", candidate)
        evidence = self.gold[query["source_split"]][qid]["gold_documents"]
        payload = {"question": query["transcript"], "candidate_answer": candidate["answer"], "gold_evidence": [{"record_id": doc["record_id"], "text": doc["text"]} for doc in evidence]}
        judge_prompt = 'Return JSON exactly as {"correctness":0,"grounding":0,"completeness":0,"explanation":"brief reason"}.\n\nINPUT:\n' + json.dumps(payload, ensure_ascii=False)
        for attempt in range(3):
            response, _ = self.call(phase, qid, arm, "judge", [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": judge_prompt}])
            try:
                scores = Judge.parse(response.choices[0].message.content or "")
                break
            except (ValueError, KeyError, TypeError):
                if attempt == 2:
                    raise
        result = {**candidate, "scores": scores, "score": sum(scores[name] for name in ["correctness", "grounding", "completeness"]) / 12}
        with self.lock:
            self.results[key] = result
            self.append("results.jsonl", result)
            self.status("running")
        return result

    def status(self, state):
        with self.lock:
            atomic_json(self.out / "status.json", {
                "state": state, "completed_answers": len(self.results), "expected_answers": self.total,
                "prepared_queries": len(self.query_cache), "requests": len(self.requests),
                "reported_api_usd": self.spent, "reserved_usd": self.reserved,
                "elapsed_seconds_this_process": time.perf_counter() - self.started,
                "completed_by_phase": dict(Counter(row["phase"] for row in self.results.values())),
                "failed_tasks_this_process": len(self.errors), "updated_at": datetime.now(timezone.utc).isoformat(),
            })

    def consume(self, futures, block=False):
        if not futures:
            return
        done, _ = wait(futures, timeout=None if block else 0, return_when=FIRST_COMPLETED)
        for future in done:
            info = futures.pop(future)
            try:
                row = future.result()
                print(f"[{len(self.results)}/{self.total}] {row['phase']} {row['arm']} {row['query_id']} k={row['k']} score={row['score']:.3f} spend=${self.spent:.4f}", flush=True)
            except Exception as error:
                failure = {**info, "error_type": type(error).__name__}
                self.errors.append(failure)
                self.append("failed-tasks.jsonl", failure)
                print("TASK ERROR", json.dumps(failure), flush=True)
                if len(self.errors) >= 10:
                    self.stop.set()

    def execute(self):
        self.tune()
        rng = random.Random(20261117)
        self.status("running")
        try:
            with ThreadPoolExecutor(max_workers=self.args.workers) as executor:
                futures = {}
                for phase, ids in self.phases.items():
                    order = list(ids)
                    rng.shuffle(order)
                    for qid in order:
                        missing = [arm for arm in ARMS if (qid, arm) not in self.results]
                        if not missing:
                            continue
                        while len(futures) >= self.args.workers * 2 and not self.stop.is_set():
                            self.consume(futures, block=True)
                        if self.stop.is_set():
                            break
                        query, query_start = self.prepare_query(qid)
                        rng.shuffle(missing)
                        for arm in missing:
                            submitted = time.perf_counter()
                            future = executor.submit(self.task, phase, query, arm, submitted, query_start)
                            futures[future] = {"phase": phase, "query_id": qid, "arm": arm}
                        self.consume(futures)
                    while futures:
                        self.consume(futures, block=True)
                    if self.stop.is_set():
                        break
        finally:
            if self.asr is not None:
                self.asr.close()
        self.status("reconciling")
        self.reconcile()
        self.summarise()
        complete = len(self.results) == self.total
        self.status("complete" if complete else "incomplete")
        if not complete:
            raise RuntimeError("Incomplete run: resume same directory to retry missing tasks")

    def reconcile(self):
        existing = {row["id"] for row in read_rows(self.out / "billing.jsonl") if row.get("billing")}
        key = os.environ.get("ELM_API_KEY") or os.environ.get("OPENAI_API_KEY")

        def lookup(row):
            url = "https://openrouter.ai/api/v1/generation?" + urllib.parse.urlencode({"id": row["id"]})
            req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=30) as response:
                        data = json.load(response)["data"]
                    self.append("billing.jsonl", {"id": row["id"], "billing": {name: data.get(name) for name in ["total_cost", "provider_name", "native_tokens_prompt", "native_tokens_completion", "native_tokens_reasoning", "finish_reason"]}})
                    return
                except Exception as error:
                    if attempt == 2:
                        self.append("billing.jsonl", {"id": row["id"], "billing": None, "error_type": type(error).__name__})
                    else:
                        time.sleep(1)
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lookup, [row for row in self.requests if row["id"] not in existing]))

    def summarise(self):
        billing = {row["id"]: row["billing"] for row in read_rows(self.out / "billing.jsonl") if row.get("billing")}
        costs = Counter()
        for row in self.requests:
            cost = billing.get(row["id"], {}).get("total_cost")
            if cost is None:
                cost = row["usage"]["cost"]
            costs[(row["phase"], row["arm"], row["role"])] += float(cost)
        result = {
            "complete": len(self.results) == self.total, "unique_queries": len(self.query_cache),
            "completed_answers": len(self.results), "expected_answers": self.total,
            "total_api_usd": sum(costs.values()), "api_requests": len(self.requests),
            "billing_records": len(billing), "parameters": self.parameters, "phases": {},
        }
        for phase in self.phases:
            arms = []
            for arm in ARMS:
                rows = [row for row in self.results.values() if row["phase"] == phase and row["arm"] == arm]
                if not rows:
                    continue
                item = {"arm": arm, "queries": len(rows)}
                for field in ["k", "score", "recall", "precision", "retrieval_f1", "service_e2e_ms", "generation_ms", "planner_ms", "queue_ms", "prompt_characters"]:
                    item["mean_" + field] = statistics.mean(row[field] for row in rows)
                for dimension in ["correctness", "grounding", "completeness"]:
                    item["mean_" + dimension] = statistics.mean(row["scores"][dimension] for row in rows)
                item["inference_usd"] = costs[(phase, arm, "generator")] + costs[(phase, arm, "planner")]
                item["judge_usd"] = costs[(phase, arm, "judge")]
                item["inference_usd_per_1000"] = item["inference_usd"] / len(rows) * 1000
                item["length_limited_answers"] = sum(row["finish_reason"] == "length" for row in rows)
                item["fallbacks"] = sum(row["fallback"] for row in rows)
                arms.append(item)
            comparisons = []
            baseline = {row["query_id"]: row for row in self.results.values() if row["phase"] == phase and row["arm"] == "fixed5"}
            rng = np.random.default_rng(20261117)
            for arm in ARMS[1:]:
                paired = [row for row in self.results.values() if row["phase"] == phase and row["arm"] == arm and row["query_id"] in baseline]
                if not paired:
                    continue
                entry = {"arm": arm, "baseline": "fixed5", "paired_queries": len(paired)}
                for field in ["score", "recall", "service_e2e_ms"]:
                    differences = np.array([row[field] - baseline[row["query_id"]][field] for row in paired])
                    samples = rng.choice(differences, size=(5000, len(paired)), replace=True).mean(axis=1)
                    entry[field + "_difference"] = float(differences.mean())
                    entry[field + "_bootstrap_95pct"] = np.percentile(samples, [2.5, 97.5]).tolist()
                comparisons.append(entry)
            result["phases"][phase] = {"by_arm": arms, "paired_vs_fixed5": comparisons}
        atomic_json(self.out / "summary.json", result)
        print("SUMMARY", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--budget", type=float, default=40)
    parser.add_argument("--model", default="mistralai/mistral-small-2603")
    parser.add_argument("--judge", default="google/gemini-2.5-flash")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("Use between 1 and 16 API workers")
    load_dotenv(ROOT / ".env")
    Run(args).execute()


if __name__ == "__main__":
    main()
