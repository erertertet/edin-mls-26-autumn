"""Independently recount the full ASR experiment in Slurm; no encoder/ASR/API calls."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import sys
import wave

import numpy as np
from tokenizers import Tokenizer

ARMS = ("glm", "whisper", "mai", "voxtral")
CONDITIONS = ("raw", "normalized")
RULES = ("top5", "cutoff")
MODELS = {"glm": "zai-org/GLM-ASR-Nano-2512", "whisper": "openai/whisper-large-v3",
          "mai": "microsoft/mai-transcribe-2", "voxtral": "mistralai/voxtral-small-transcribe-3-0"}
NORMALIZER_SHA256 = "fff6993b2f1ae85db3a65a6c645a6e2ac3ad0ec415eb82af3f73a763de8e9dd2"
GLM_SNAPSHOT = "61ba4e0b3309b6656edea3e93e419f7bd5c61957"
TAU = 0.8925
MAI_OPTIONS = {"options": {"azure": {"enhancedMode": {"enabled": True, "model": "MAI-Transcribe-2",
               "modelOptions": {"transcribeStyle": "verbatim"}}, "diarization": {"enabled": False}}}}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(actual, expected, message, tolerance=1e-12):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def keyed(rows, fields):
    result = {tuple(row[field] for field in fields): row for row in rows}
    require(len(result) == len(rows), "Duplicate records: " + ",".join(fields))
    return result


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def transcript_file(run, qid, arm):
    return run / "transcripts" / arm / (text_digest(qid)[:20] + ".json")


def expected_options(arm):
    result = {"model": MODELS[arm], "language": "en", "temperature": 0, "response_format": "json"}
    if arm == "mai":
        result["provider"] = MAI_OPTIONS
    return result


def request_id(value):
    raw = value.get("raw_response", {})
    return raw.get("id") or next((value.get("response_headers", {}).get(name)
        for name in ("x-request-id", "x-generation-id") if value.get("response_headers", {}).get(name)), None)


def audit(args):
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"), "Run in Edinburgh Slurm")
    sys.path.insert(0, str(args.root / "rag"))
    sys.path.insert(0, str(args.root / "rag/extensions"))
    from src.data import CourseData
    from src.retriever import ExactDenseRetriever
    from transcript_normalization import POLICY, POLICY_VERSION, normalize_transcript

    run = args.run
    config = read(run / "config.json")
    selection = read(run / "selection.json")
    require(digest(run / "selection.json") == config["selection_sha256"], "Selection changed")
    normalizer = args.root / "rag/extensions/transcript_normalization.py"
    require(digest(normalizer) == config["normalizer_sha256"] == NORMALIZER_SHA256, "Frozen normalizer changed")
    require(config["policy_version"] == POLICY_VERSION == "acronym-punctuation-v1" and config["policy"] == POLICY,
            "Normalization policy declaration changed")
    require(config["models"] == MODELS and config["conditions"] == list(CONDITIONS)
            and config["rules"] == list(RULES) and config["tau"] == TAU and config["topk"] == 5,
            "Experiment settings changed")
    require(config["glm_snapshot"] == GLM_SNAPSHOT and config["asr_cuda"] == "all"
            and config["glm_max_new_tokens"] == 100, "GLM settings changed")
    require(config["api_options"] == {a: expected_options(a) for a in ARMS if a != "glm"}
            and config["api_max_retries"] == 0 and config["api_workers"] == 3
            and config["task_seed"] == 26 and config["new_spend_guard_usd"] == 1, "API settings changed")
    require(digest(args.tokenizer) == config["tokenizer"]["sha256"], "Tokenizer bytes changed")
    require(config["tokenizer"]["repo"] == "mistralai/Mistral-Small-4-119B-2603"
            and config["tokenizer"]["revision"] == "a11f36bebf709121056b1dbcc943d1c6afbe494d",
            "Tokenizer identity changed")

    def input_path(relative):
        if relative.startswith("rag/data/rag_corpus/"):
            return args.data_dir / relative.removeprefix("rag/data/rag_corpus/")
        if relative.startswith("rag/models/enc_a/"):
            return args.encoder_dir / relative.removeprefix("rag/models/enc_a/")
        return args.root / relative

    for relative, expected in config["sources"].items():
        require(digest(args.root / relative) == expected, "Frozen experiment source mismatch: " + relative)
    for relative, expected in config["input_sources"].items():
        require(digest(input_path(relative)) == expected, "Original input/source mismatch: " + relative)

    pilot = args.root / "rag/results/asr-four-model-pilot-20261116"
    normalized_pilot = args.root / "rag/results/asr-four-model-normalized-20261116"
    pilot_config = read(pilot / "config.json")
    require(digest(pilot / "config.json") == config["pilot_config_sha256"], "Pilot config changed")
    require(digest(pilot / "retrieval.jsonl") == config["pilot_retrieval_sha256"], "Pilot raw retrieval changed")
    require(digest(normalized_pilot / "normalized/retrieval.jsonl") == config["normalized_pilot_retrieval_sha256"],
            "Pilot normalized retrieval changed")
    require(pilot_config["models"] == MODELS and pilot_config["glm_snapshot"] == GLM_SNAPSHOT
            and pilot_config["asr_cuda"] == "all" and pilot_config["glm_max_new_tokens"] == 100
            and pilot_config["mai_provider_options"] == MAI_OPTIONS,
            "Imported pilot settings mismatch")
    require(all(config["input_sources"].get(p) == value for p, value in pilot_config["sources"].items()),
            "Original pilot inputs not preserved")

    data = CourseData(args.data_dir)
    manifest_path = args.data_dir / "student_audio_manifest.jsonl"
    require(digest(manifest_path) == selection["manifest_sha256"], "Manifest changed")
    manifest = keyed(data.load_audio_manifest(), ["query_id"])
    selected_rows = selection["audio"]
    audios = keyed(selected_rows, ["query_id"])
    ids = selection["query_ids"]
    require(len(ids) == len(set(ids)) == len(audios) == len(manifest) == 475
            and set(audios) == set(manifest) == {(q,) for q in ids}, "Full manifest coverage mismatch")
    require([a["query_id"] for a in selected_rows] == ids, "Selection order mismatch")
    require(set(selection["splits"]) == {"practice", "dev"}, "Unexpected audio splits")
    split_ids = {split: [a["query_id"] for a in selected_rows if a["split"] == split] for split in selection["splits"]}
    require({s: len(v) for s, v in split_ids.items()} == selection["counts_by_split"] == {"practice": 350, "dev": 125},
            "Split sizes changed")
    pilot_ids = selection["pilot_ids"]
    require(pilot_ids == read(pilot / "selection.json")["selected_ids"]
            and len(pilot_ids) == len(set(pilot_ids)) == 8 and set(pilot_ids) <= set(split_ids["dev"]),
            "Pilot cohort changed")
    expected_cohorts = {"all": ids, "practice": split_ids["practice"], "dev": split_ids["dev"], "pilot": pilot_ids,
                        "remaining_dev": [q for q in split_ids["dev"] if q not in pilot_ids],
                        "remaining_all": [q for q in ids if q not in pilot_ids]}
    require(config["cohorts"] == expected_cohorts, "Frozen cohort membership mismatch")
    gold = {split: data.load_qrels(split) for split in split_ids}
    total_seconds = 0.0
    for qid in ids:
        audio, item = audios[qid,], manifest[qid,]
        require(audio["split"] == item["split"] and audio["audio_file"] == item["audio_file"], "Manifest audio mismatch")
        path = (args.data_dir / audio["audio_file"]).resolve()
        path.relative_to(args.data_dir.resolve())
        require(digest(path) == audio["audio_sha256"], "Audio hash mismatch: " + qid)
        if item.get("sha256"):
            require(item["sha256"] == audio["audio_sha256"], "Course manifest SHA mismatch")
        with wave.open(str(path), "rb") as stream:
            require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype())
                    == (16000, 1, 2, "NONE"), "Original PCM audio properties changed")
            seconds = stream.getnframes() / stream.getframerate()
        require(seconds > 0, "Empty recording")
        close(audio["seconds"], seconds, "Audio duration mismatch")
        total_seconds += seconds
        require(qid in gold[audio["split"]] and gold[audio["split"]][qid], "No positive qrels for selected query")
        require(set(gold[audio["split"]][qid]) <= set(data.records), "Qrel record absent from corpus")

    origins = keyed(read(run / "cache-origins.json"), ["query_id", "arm"])
    require(set(origins) == {(q, a) for q in pilot_ids for a in ARMS}, "Need 32 original cache origins")
    mappings = keyed(read(run / "transcript-mappings.json"), ["query_id", "arm"])
    pairs = {(q, a) for q in ids for a in ARMS}
    require(set(mappings) == pairs and len(mappings) == 1900, "Need all 1900 unique mappings")
    transcript_paths = set()
    transcripts, external_identifiers, external_costs = {}, [], []
    new_transcript_attempt_paths = set()
    for qid, arm in sorted(pairs):
        mapping = mappings[qid, arm]
        audio = audios[qid,]
        path = transcript_file(run, qid, arm)
        transcript_paths.add(path.resolve())
        value = read(path)
        require(value["query_id"] == qid and value["model"] == mapping["model"] == MODELS[arm]
                and value["status"] == "complete" and isinstance(value["transcript"], str)
                and value["transcript"].strip(), "Transcript identity/status mismatch")
        require(value["audio_sha256"] == mapping["audio_sha256"] == audio["audio_sha256"]
                and mapping["split"] == audio["split"], "Transcript audio/split provenance mismatch")
        require(mapping["source_sha256"] == digest(path) and mapping["raw"] == value["transcript"],
                "Raw mapping/source hash mismatch")
        require(mapping["normalized"] == normalize_transcript(mapping["raw"]) and mapping["normalized"]
                and normalize_transcript(mapping["normalized"]) == mapping["normalized"], "Frozen normalized mapping mismatch")
        for field in ("raw", "normalized"):
            require(text_digest(mapping[field]) == mapping[field + "_text_sha256"], "Mapping text hash mismatch")
        if qid in pilot_ids:
            origin = origins[qid, arm]
            source = transcript_file(pilot, qid, arm)
            require(Path(origin["source_path"]).resolve() == source.resolve()
                    and digest(source) == digest(path) == origin["source_sha256"]
                    and origin["pilot_config_sha256"] == config["pilot_config_sha256"], "Pilot cache is not byte-identical")
        if arm == "glm":
            require(value["asr_cuda"] == "all" and value["max_new_tokens"] == 100, "GLM decoding changed")
            if qid not in pilot_ids:
                require(value["glm_snapshot"] == GLM_SNAPSHOT, "New GLM checkpoint changed")
        else:
            require(value["requested_options"] == expected_options(arm), "External request options changed")
            require(isinstance(value["raw_response"].get("text"), str)
                    and value["raw_response"]["text"].strip() == value["transcript"], "Response/transcript text mismatch")
            identifier = request_id(value)
            if identifier:
                external_identifiers.append(identifier)
            cost = (value["raw_response"].get("usage") or {}).get("cost")
            if cost is not None:
                require(math.isfinite(cost) and cost >= 0, "Invalid successful response cost")
                external_costs.append(cost)
            if qid not in pilot_ids:
                attempt_path = (run / value["request_attempt"]).resolve()
                attempt_path.relative_to((run / "api-attempts").resolve())
                new_transcript_attempt_paths.add(attempt_path)
                attempt = read(attempt_path)
                require(attempt["query_id"] == qid and attempt["arm"] == arm and attempt["status"] == "complete"
                        and attempt["model"] == MODELS[arm] and attempt["raw_response"] == value["raw_response"]
                        and attempt["response_headers"] == value["response_headers"], "Successful request attempt mismatch")
        transcripts[qid, arm] = value
    require(transcript_paths == {p.resolve() for p in (run / "transcripts").glob("*/*.json")}, "Unaccounted transcript files")
    require(len(external_identifiers) == len(set(external_identifiers)), "Duplicate external request identifiers")

    attempts = []
    attempt_keys, completed_attempt_paths = {}, set()
    for path in sorted((run / "api-attempts").glob("*/*.json")):
        value = read(path)
        qid, arm = value["query_id"], value["arm"]
        require(qid in expected_cohorts["remaining_all"] and arm in ARMS[1:] and path.parent.name == arm
                and value["model"] == MODELS[arm], "Unexpected paid request attempt")
        require(value["status"] in ("started", "failed", "complete") and value.get("utc") and value.get("slurm_job_id"),
                "Invalid API attempt state/provenance")
        stem, number = path.stem.rsplit("-", 1)
        require(stem == text_digest(qid)[:20] and number.isdigit() and 1 <= int(number) <= 3, "Invalid attempt filename/index")
        attempt_keys.setdefault((qid, arm), []).append(int(number))
        require(math.isfinite(value["estimated_cost_usd"]) and value["estimated_cost_usd"] >= 0, "Invalid estimated cost")
        cost = value.get("response_cost_usd")
        if cost is not None:
            require(math.isfinite(cost) and cost >= 0, "Invalid attempt cost")
            close(cost, (value.get("raw_response", {}).get("usage") or {})["cost"], "Attempt response cost mismatch")
        if value["status"] == "complete":
            completed_attempt_paths.add(path.resolve())
        attempts.append(value)
    require(set(attempt_keys) == {(q, a) for q in expected_cohorts["remaining_all"] for a in ARMS[1:]},
            "Missing/unexpected new API attempt cells")
    require(all(sorted(numbers) == list(range(1, len(numbers) + 1)) and len(numbers) <= 3
                for numbers in attempt_keys.values()), "Attempt sequence/retry ceiling mismatch")
    require(completed_attempt_paths == new_transcript_attempt_paths and len(completed_attempt_paths) == 1401,
            "Completed attempts and new successful transcripts disagree")
    billing = read(run / "billing.json")
    billing_start = read(run / "billing-start.json")
    new_costs = [a["response_cost_usd"] for a in attempts if a.get("response_cost_usd") is not None]
    failed_or_ambiguous = [a for a in attempts if a["status"] != "complete"]
    unknown = any(a.get("response_cost_usd") is None for a in failed_or_ambiguous)
    require(billing["new_attempts"] == len(attempts)
            and billing["failed_or_ambiguous_attempts"] == len(failed_or_ambiguous)
            and billing["failed_or_ambiguous_cost_is_unknown"] == unknown, "Billing attempt/unknown coverage mismatch")
    close(billing["new_response_cost_sum_usd"], math.fsum(new_costs), "New response cost sum mismatch")
    close(billing["all_successful_response_cost_sum_usd"], math.fsum(external_costs), "All successful response cost mismatch")
    require(billing["successful_responses_with_cost"] == len(external_costs), "Successful cost coverage mismatch")
    require(billing_start["guard_usd"] == 1 and 0 <= billing_start["conservative_reserve_usd"] <= 1,
            "Initial spend guard mismatch")
    preflights = list(run.glob("spend-preflight-*.json"))
    require(preflights, "No spend preflight")
    for path in preflights:
        value = read(path)
        require(0 <= value["reserved_usd"] <= 1 and value["pending_requests"] >= 0, "Invalid spend preflight")
    require(math.fsum(new_costs) <= 1, "Known new response charges exceed the USD 1 guard")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows = jsonl(run / "retrieval.jsonl")
    cells = keyed(rows, ["query_id", "condition", "arm", "rule"])
    expected_cells = {(q, c, a, r) for q in ids for c in CONDITIONS for a in ARMS for r in RULES}
    require(len(cells) == 7600 and set(cells) == expected_cells, "Need all 7600 retrieval cells")
    saved = np.load(run / "query-vectors.npz", allow_pickle=False)
    keys, vectors = saved["keys"], saved["vectors"]
    require(keys.shape == (3800, 3) and vectors.shape == (3800, 384) and vectors.dtype == np.float32,
            "Invalid saved query-vector schema")
    vector_rows = {tuple(key): index for index, key in enumerate(keys.tolist())}
    require(len(vector_rows) == len(keys) and set(vector_rows) == {(q, a, c) for q in ids for a in ARMS for c in CONDITIONS},
            "Saved vector coverage mismatch")
    require(np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5, rtol=0),
            "Saved vectors are not finite normalized float32 vectors")
    retriever = ExactDenseRetriever(args.data_dir / "embeddings/corpus_enc_a.npy",
                                    args.data_dir / "embeddings/corpus_enc_a_ids.json")
    require(len(retriever.record_ids) == len(set(retriever.record_ids)) == 24000, "Corpus-vector coverage mismatch")
    tokens, maximum_score_drift = {}, 0.0
    for index, ((qid, arm, condition), vector_index) in enumerate(vector_rows.items(), 1):
        vector = vectors[vector_index]
        # The recorded vector is already normalized; repeating normalization changes thresholds.
        scores = retriever.embeddings @ vector
        order = np.argsort(-scores, kind="stable")
        ordered_scores = scores[order].astype(np.float64)
        matching = int(np.count_nonzero(ordered_scores >= TAU))
        for rule, count in (("top5", 5), ("cutoff", max(1, matching))):
            row = cells[qid, condition, arm, rule]
            expected_ids = [retriever.record_ids[int(i)] for i in order[:count]]
            require(row["record_ids"] == expected_ids and row["documents"] == count
                    and len(set(row["record_ids"])) == count, "Full ranking/uncapped selection mismatch")
            require(len(row["selected_scores"]) == count, "Selected score length mismatch")
            drift = float(np.max(np.abs(ordered_scores[:count] - row["selected_scores"])))
            require(drift <= 1e-6, "Recomputed ranking scores exceed 1e-6 tolerance")
            maximum_score_drift = max(maximum_score_drift, drift)
            close(row["top_similarity"], float(ordered_scores[0]), "Top similarity mismatch", 1e-6)
            require(row["fallback"] == (rule == "cutoff" and matching == 0), "Cutoff fallback mismatch")
            split = audios[qid,]["split"]
            require(row["split"] == split and row["model"] == MODELS[arm], "Retrieval split/model mismatch")
            relevant = gold[split][qid]
            hits = len(set(expected_ids) & set(relevant))
            require(row["relevant_hits"] == hits and row["total_relevant"] == len(relevant), "Qrel counts mismatch")
            close(row["recall"], hits / len(relevant), "Recall mismatch")
            for rid in expected_ids:
                if rid not in tokens:
                    tokens[rid] = len(tokenizer.encode(data.record_text(rid), add_special_tokens=False).ids)
            require(row["context_tokens"] == sum(tokens[rid] for rid in expected_ids), "Document token mismatch")
            require(row["transcript_tokens"] == len(tokenizer.encode(mappings[qid, arm][condition], add_special_tokens=False).ids),
                    "Query transcript token mismatch")
        if index % 500 == 0 or index == len(vector_rows):
            print(json.dumps({"audit_stage": "full_rankings", "checked": index, "total": len(vector_rows)}), flush=True)
    require(read(run / "document-token-cache.json") == tokens, "Saved document-token cache mismatch")

    summary = read(run / "summary.json")
    require(summary["status"] == "complete" and summary["observed_cells"] == 7600
            and summary["expected_cells_per_query"] == 16 and set(summary["cohorts"]) == set(expected_cohorts),
            "Full summary coverage mismatch")
    table_count, change_count = 0, 0
    cohort_report = {}
    for name, selected in expected_cohorts.items():
        item = summary["cohorts"][name]
        n = len(selected)
        require(item["status"] == "complete" and item["expected_complete"] and item["expected_queries"] == n
                and item["expected_cells"] == item["observed_cells"] == n * 16 and item["paired_queries"] == n
                and item["complete_paired_queries"] == selected and not item["missing_queries"] and not item["missing_cells"],
                "Cohort coverage mismatch: " + name)
        table = keyed(item["table"], ["condition", "arm", "rule"])
        require(set(table) == {(c, a, r) for c in CONDITIONS for a in ARMS for r in RULES}, "Cohort table settings mismatch")
        for (condition, arm, rule), row in table.items():
            require(row["paired_queries"] == n, "Cohort table pairing mismatch")
            values = [cells[q, condition, arm, rule] for q in selected]
            baseline = [cells[q, condition, "glm", rule] for q in selected]
            expected = {"mean_recall": mean(v["recall"] for v in values),
                        "mean_context_tokens": mean(v["context_tokens"] for v in values),
                        "mean_documents": mean(v["documents"] for v in values),
                        "fallback_queries": sum(v["fallback"] for v in values),
                        "delta_recall_pp_vs_glm": mean(100 * (v["recall"] - b["recall"]) for v, b in zip(values, baseline)),
                        "delta_context_tokens_vs_glm": mean(v["context_tokens"] - b["context_tokens"] for v, b in zip(values, baseline))}
            for field, expected_value in expected.items():
                close(row[field], expected_value, "Cohort statistic mismatch: " + name + "/" + field)
            table_count += 1
        changes = keyed(item["normalized_minus_raw"], ["arm", "rule"])
        require(set(changes) == {(a, r) for a in ARMS for r in RULES}, "Normalization contrast coverage mismatch")
        for (arm, rule), row in changes.items():
            require(row["paired_queries"] == n, "Contrast pairing mismatch")
            raw = [cells[q, "raw", arm, rule] for q in selected]
            normalized = [cells[q, "normalized", arm, rule] for q in selected]
            recall_delta = [v["recall"] - r["recall"] for v, r in zip(normalized, raw)]
            token_delta = [v["context_tokens"] - r["context_tokens"] for v, r in zip(normalized, raw)]
            close(row["recall_change_pp"], mean(100 * d for d in recall_delta), "Normalized recall delta mismatch")
            close(row["context_token_change"], mean(token_delta), "Normalized token delta mismatch")
            expected_counts = {"queries_with_recall_improved": sum(d > 0 for d in recall_delta),
                "queries_with_recall_worsened": sum(d < 0 for d in recall_delta),
                "queries_with_recall_unchanged": sum(d == 0 for d in recall_delta),
                "queries_with_token_increase": sum(d > 0 for d in token_delta),
                "queries_with_token_decrease": sum(d < 0 for d in token_delta),
                "queries_with_token_unchanged": sum(d == 0 for d in token_delta),
                "queries_with_changed_selected_ids": sum(v["record_ids"] != r["record_ids"] for v, r in zip(normalized, raw))}
            require(all(row[k] == v for k, v in expected_counts.items()), "Normalization contrast counts mismatch")
            change_count += 1
        agreement = item["text_agreement"]
        require(agreement["complete_mapped_queries"] == selected and not agreement["missing_mappings"], "Mapping coverage mismatch")
        agreement_rows = keyed(agreement["per_query"], ["query_id"])
        require(set(agreement_rows) == {(q,) for q in selected}, "Text agreement query coverage mismatch")
        raw_equal, normalized_equal = 0, 0
        for qid in selected:
            raw_distinct = len({mappings[qid, a]["raw"] for a in ARMS})
            norm_distinct = len({mappings[qid, a]["normalized"] for a in ARMS})
            require(agreement_rows[qid,] == {"query_id": qid, "available_models": 4,
                    "raw_distinct_texts": raw_distinct, "normalized_distinct_texts": norm_distinct}, "Text agreement mismatch")
            raw_equal += raw_distinct == 1
            normalized_equal += norm_distinct == 1
        require(agreement["raw_identical_queries"] == raw_equal
                and agreement["normalized_identical_queries"] == normalized_equal, "Text agreement totals mismatch")
        cohort_report[name] = {"queries": n, "raw_identical_queries": raw_equal, "normalized_identical_queries": normalized_equal}

    for qid in ids:
        for condition in CONDITIONS:
            for left_index, left in enumerate(ARMS):
                for right in ARMS[left_index + 1:]:
                    if mappings[qid, left][condition] != mappings[qid, right][condition]:
                        continue
                    require(np.array_equal(vectors[vector_rows[qid, left, condition]], vectors[vector_rows[qid, right, condition]]),
                            "Identical texts have unequal recorded vectors")
                    for rule in RULES:
                        one, two = cells[qid, condition, left, rule], cells[qid, condition, right, rule]
                        require(all(one[field] == two[field] for field in ("record_ids", "documents", "recall", "context_tokens",
                                    "selected_scores", "fallback", "transcript_tokens")), "Identical texts have unequal retrieval outputs")

    pilot_count, pilot_drift = 0, 0.0
    for condition, path in (("raw", pilot / "retrieval.jsonl"),
                            ("normalized", normalized_pilot / "normalized/retrieval.jsonl")):
        previous = keyed(jsonl(path), ["query_id", "arm", "rule"])
        require(set(previous) == {(q, a, r) for q in pilot_ids for a in ARMS for r in RULES}, "Pilot coverage mismatch")
        for (qid, arm, rule), old in previous.items():
            row = cells[qid, condition, arm, rule]
            require(all(row[field] == old[field] for field in ("record_ids", "recall", "context_tokens", "documents", "fallback",
                        "transcript_tokens", "relevant_hits", "total_relevant")), "Original pilot control mismatch")
            require(len(row["selected_scores"]) == len(old["selected_scores"]), "Pilot score count mismatch")
            drift = max(abs(a - b) for a, b in zip(row["selected_scores"], old["selected_scores"]))
            require(drift <= 1e-6, "Original pilot score drift exceeds tolerance")
            pilot_drift = max(pilot_drift, drift)
            pilot_count += 1
    control = read(run / "pilot-control-check.json")
    require(control["status"] == "pass" and control["cells"] == pilot_count == 128, "Pilot control declaration mismatch")
    close(control["max_abs_score_drift"], pilot_drift, "Pilot drift declaration mismatch")
    completion = read(run / "completion.json")
    require(completion["status"] == "complete" and completion["transcripts"] == 1900 and completion["queries"] == 475
            and completion["retrieval_cells"] == completion["expected_retrieval_cells"] == 7600,
            "Completion declaration mismatch")
    require(read(run / ("environment-" + completion["slurm_job_id"] + ".json"))["slurm_job_id"] == completion["slurm_job_id"],
            "Completion environment missing/mismatched")
    artifacts = ("selection.json", "config.json", "cache-origins.json", "transcript-mappings.json", "query-vectors.npz",
                 "retrieval.jsonl", "summary.json", "completion.json", "billing.json", "pilot-control-check.json")
    result = {"status": "pass", "errors": [], "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "experiment_slurm_job_id": completion["slurm_job_id"], "queries": 475, "audio_seconds_checked": total_seconds,
              "transcripts_checked": 1900, "byte_identical_imported_pilot_transcripts": 32, "text_hashes_checked": 3800,
              "full_rankings_recomputed": 3800, "corpus_records_per_ranking": 24000, "retrieval_cells_checked": 7600,
              "cohort_summary_rows_checked": table_count, "normalization_contrasts_checked": change_count,
              "unique_documents_retokenized": len(tokens), "pilot_control_cells_checked": pilot_count,
              "max_abs_ranking_score_drift": maximum_score_drift, "max_abs_pilot_score_drift": pilot_drift,
              "normalizer_sha256": NORMALIZER_SHA256, "source_hashes_checked": len(config["sources"]),
              "input_source_hashes_checked": len(config["input_sources"]), "cohorts": cohort_report,
              "new_api_attempts": len(attempts), "failed_or_ambiguous_attempts": len(failed_or_ambiguous),
              "failed_or_ambiguous_cost_is_unknown": unknown, "successful_response_cost_entries": len(external_costs),
              "distinct_successful_request_identifiers": len(external_identifiers),
              "successful_responses_without_request_identifier": 1425 - len(external_identifiers),
              "new_response_cost_sum_usd": math.fsum(new_costs),
              "all_successful_response_cost_sum_usd": math.fsum(external_costs),
              "key_usage_snapshot_delta_usd": billing["key_usage_after_usd"] - billing_start["usage_usd"],
              "audit_script_sha256": digest(__file__), "numpy_version": np.__version__,
              "tokenizers_version": importlib.metadata.version("tokenizers"),
              "artifact_sha256": {name: digest(run / name) for name in artifacts},
              "audit_scope": "Original sources/audio/mappings, saved-vector full ranking, uncapped cutoff, qrel/token accounting and independent cohort aggregates; no encoder/ASR/API execution."}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run", "data-dir", "tokenizer"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path)
    args = parser.parse_args()
    if args.encoder_dir is None:
        args.encoder_dir = args.root / "rag/models/enc_a"
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"), "Run in Edinburgh Slurm")
    os.umask(0o077)
    try:
        result = audit(args)
    except Exception as exc:
        result = {"status": "fail", "slurm_job_id": os.environ["SLURM_JOB_ID"],
                  "errors": [{"type": type(exc).__name__, "message": str(exc)}], "audit_script_sha256": digest(__file__)}
        (args.run / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise
    (args.run / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
