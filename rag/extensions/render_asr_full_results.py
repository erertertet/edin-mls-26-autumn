"""Render audited saved full-ASR aggregates as Markdown, exclusively in Slurm."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket

ARMS = ("glm", "whisper", "mai", "voxtral")
RULES = ("top5", "cutoff")
NAMES = {"glm": "GLM-ASR-Nano", "whisper": "Whisper large-v3",
         "mai": "MAI-Transcribe 2", "voxtral": "Voxtral Small Transcribe 3"}
COHORTS = (("all", "All supplied audio"), ("dev", "Dev"),
           ("remaining_dev", "Remaining dev, excluding the pilot"),
           ("practice", "Practice"), ("remaining_all", "Remaining audio, excluding the pilot"),
           ("pilot", "Original pilot"))
NORMALIZER_SHA256 = "fff6993b2f1ae85db3a65a6c645a6e2ac3ad0ec415eb82af3f73a763de8e9dd2"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def number(value, digits=3, signed=False):
    require(value is not None and math.isfinite(value), "Missing/nonfinite complete-result value")
    return format(value, ("+" if signed else "") + "." + str(digits) + "f")


def policy(rule):
    return "top-5" if rule == "top5" else "cutoff 0.8925"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"),
            "Run only inside an Edinburgh Slurm allocation")
    os.umask(0o077)
    run = args.run.resolve()
    inputs = {name: read(run / (name + ".json"))
              for name in ("summary", "config", "completion", "billing", "audit")}
    summary, config, completion, billing, audit = (inputs[name] for name in inputs)
    require(summary["status"] == completion["status"] == "complete"
            and audit["status"] == "pass", "Render a final note only after complete coverage and audit pass")
    require(completion["slurm_job_id"] == audit["experiment_slurm_job_id"], "Experiment/audit job mismatch")
    require(config["normalizer_sha256"] == audit["normalizer_sha256"] == NORMALIZER_SHA256,
            "Frozen normalization SHA mismatch")
    for name in ("summary", "config", "completion", "billing"):
        path = run / (name + ".json")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(audit["artifact_sha256"].get(path.name) == digest, "Artifact changed since audit: " + path.name)
    require(completion["queries"] == audit["queries"] == len(config["cohorts"]["all"]) == 475
            and completion["transcripts"] == audit["transcripts_checked"] == 1900
            and completion["retrieval_cells"] == audit["retrieval_cells_checked"] == 7600,
            "Full coverage does not match the frozen 475-query experiment")
    environments = [read(path) for path in sorted(run.glob("environment-*.json"))]
    require(any(value["slurm_job_id"] == completion["slurm_job_id"] for value in environments),
            "Missing completed-experiment environment record")
    date = completion.get("utc", "").split("T", 1)[0]
    lines = ["# Full-audio ASR × formatting normalization × retrieval results" + (" — " + date + " UTC" if date else ""), "",
        "The frozen experiment completed **475/475 supplied audio queries**: **350 practice and 125 dev**, "
        "with **1,900 four-model transcripts and all 7,600 retrieval cells**. "
        "The exact 32 original pilot transcripts were reused; the remaining 467 recordings were completed with the same four ASR arms. "
        "This note presents saved audited aggregates; it does not regenerate or revise transcripts.", "",
        "## Fixed settings and provenance", "",
        "- Normalization policy: `" + config["policy_version"] + "`; unchanged file SHA256: `" + config["normalizer_sha256"] + "`.",
        "- The same rule is applied to every model: initialism merging, Unicode/case/punctuation/hyphen and whitespace normalization. "
        "Raw text is retained privately; no reference-based correction, domain dictionary or corpus-text normalization is applied.",
        "- Same supplied `enc_a` on CPU with its existing dtype behavior, 24,000 corpus records and their existing vectors; "
        "independent encoding of each raw and normalized transcript; full float32 cosine ranking with stable corpus-row tie breaking.",
        "- Fixed top-5, or all scores >=0.8925 with top-1 fallback if none qualify and no document-count cap. "
        "Original positive qrels are selected by each recording's own practice/dev split. No threshold retuning is included.",
        "- Context cost is the sum of individually tokenized selected document texts, without special tokens, "
        "question/system text or prompt framing. It is **not a billed LLM prompt-token count**.",
        "- Tokenizer: `" + config["tokenizer"]["repo"] + "`, revision `" + config["tokenizer"]["revision"] + "`, SHA256 `" + config["tokenizer"]["sha256"] + "`.",
        "- GLM snapshot: `" + config["glm_snapshot"] + "`; `ASR_CUDA=" + config["asr_cuda"] + "`, greedy decoding and "
        + str(config["glm_max_new_tokens"]) + " new-token ceiling. OpenRouter requests retain the pilot's English/temperature-zero/JSON options "
        "and MAI verbatim setting. External service aliases are not immutable checkpoint identifiers.",
        "- All experiment execution, tests and audits take place in Edinburgh Slurm. "
        "No answer generation, judging or reference-based WER evaluation is included.", "",
        "| Environment Slurm job | Stage | Source commit | Host | Recorded worktree |",
        "| --- | --- | --- | --- | --- |"]
    for value in environments:
        lines.append("| `" + str(value["slurm_job_id"]) + "` | " + str(value.get("stage", "unavailable"))
                     + " | `" + str(value.get("git_head", "unavailable")) + "` | `"
                     + str(value.get("hostname", "unavailable")) + "` | "
                     + ("clean" if not value.get("git_status") else "modified; see private environment record") + " |")
    lines += ["", "Completed-experiment job: **" + str(completion["slurm_job_id"]) + "**. "
              "Independent audit job: **" + str(audit["slurm_job_id"]) + "**, status **pass**. "
              "Rendering allocation Slurm job: **" + os.environ["SLURM_JOB_ID"] + "**.", "",
              "## Coverage and cohort results", "",
              "Each cohort uses one common complete query set across all 16 settings "
              "(four ASR models × raw/normalized × two retrieval rules). "
              "Recall is the macro mean of original-query labelled-evidence recall. "
              "Changes below are **normalized minus raw**; recall changes are percentage points (pp). "
              "Token changes use the document-text measure above. These are descriptive measurements, without significance claims."]
    for name, title in COHORTS:
        item = summary["cohorts"][name]
        n = len(config["cohorts"][name])
        require(item["status"] == "complete" and item["expected_complete"]
                and item["paired_queries"] == item["expected_queries"] == n
                and not item["missing_cells"] and not item["text_agreement"]["missing_mappings"],
                "Incomplete cohort: " + name)
        table = {(r["condition"], r["arm"], r["rule"]): r for r in item["table"]}
        changes = {(r["arm"], r["rule"]): r for r in item["normalized_minus_raw"]}
        require(len(table) == 16 and len(changes) == 8, "Unexpected cohort table size")
        agreement = item["text_agreement"]
        mapped_n = len(agreement["complete_mapped_queries"])
        require(mapped_n == n, "Incomplete mapped-query denominator")
        lines += ["", "### " + title + " (" + str(n) + " queries)", "",
            "Complete paired coverage: **" + str(n) + "/" + str(n) + " queries**, **"
            + str(item["observed_cells"]) + "/" + str(item["expected_cells"]) + " retrieval cells**; missing cells: **0**.", "",
            "Four-model text identity, counted per original query: **" + str(agreement["raw_identical_queries"])
            + "/" + str(mapped_n) + " raw**, **" + str(agreement["normalized_identical_queries"]) + "/"
            + str(mapped_n) + " normalized**. "
            "Identical normalized inputs can lead to identical deterministic retrieval outputs; this does not establish equal general ASR quality.", "",
            "| ASR | Retrieval | Raw recall (%) | Normalized recall (%) | Δ recall (pp) | Raw tokens | Normalized tokens | Δ tokens |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for rule in RULES:
            for arm in ARMS:
                raw, normalized = table["raw", arm, rule], table["normalized", arm, rule]
                change = changes[arm, rule]
                lines.append("| " + NAMES[arm] + " | " + policy(rule) + " | "
                    + number(100 * raw["mean_recall"]) + " | " + number(100 * normalized["mean_recall"])
                    + " | " + number(change["recall_change_pp"], signed=True) + " | "
                    + number(raw["mean_context_tokens"]) + " | " + number(normalized["mean_context_tokens"])
                    + " | " + number(change["context_token_change"], signed=True) + " |")
        lines += ["", "Query-level changes after normalization; every count uses the same **" + str(n) + "** paired queries. "
                  "Token decrease/increase counts describe context size, not an accuracy-adjusted benefit.", "",
                  "| ASR | Retrieval | Recall gain / loss / tie | Token decrease / increase / tie | Changed selected-ID lists |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for rule in RULES:
            for arm in ARMS:
                change = changes[arm, rule]
                recalls = [change["queries_with_recall_" + suffix] for suffix in ("improved", "worsened", "unchanged")]
                tokens = [change["queries_with_token_" + suffix] for suffix in ("decrease", "increase", "unchanged")]
                require(sum(recalls) == sum(tokens) == n, "Change-count denominator mismatch")
                lines.append("| " + NAMES[arm] + " | " + policy(rule) + " | "
                             + " / ".join(map(str, recalls)) + " | " + " / ".join(map(str, tokens))
                             + " | " + str(change["queries_with_changed_selected_ids"]) + " |")
    lines += ["", "## Independent audit and cost accounting", "",
        "The independent audit passed for **" + str(audit["transcripts_checked"]) + " transcripts**, **"
        + str(audit["text_hashes_checked"]) + " text hashes**, **" + str(audit["full_rankings_recomputed"])
        + " saved-vector rankings over " + str(audit["corpus_records_per_ranking"]) + " records each**, and **"
        + str(audit["retrieval_cells_checked"]) + " retrieval cells**. It checked uncapped cutoff completeness, "
        "original qrel recall, **" + str(audit["unique_documents_retokenized"]) + " unique retokenized documents**, **"
        + str(audit["cohort_summary_rows_checked"]) + " cohort-summary rows** and **"
        + str(audit["normalization_contrasts_checked"]) + " normalization contrasts**. "
        "It did not rerun ASR or the encoder.", "",
        "The original pilot's **" + str(audit["pilot_control_cells_checked"]) + " raw/normalized control cells** "
        "were reproduced; maximum selected-score drift was **" + number(audit["max_abs_pilot_score_drift"], 12)
        + "**, within the fixed 1e-6 gate. Independent full-ranking score drift was **"
        + number(audit["max_abs_ranking_score_drift"], 12) + "**.", "",
        "- **New-attempt response costs:** USD **" + number(billing["new_response_cost_sum_usd"], 12)
        + "**, from **" + str(billing["new_attempts"]) + "** newly logged attempts. "
        "This includes a failed attempt only when that attempt returned cost metadata.",
        "- **Reported costs for all successful transcript responses, including the cached pilot:** USD **"
        + number(billing["all_successful_response_cost_sum_usd"], 12) + "**. Cost metadata covers **"
        + str(billing["successful_responses_with_cost"]) + "/"
        + str(completion["queries"] * (len(ARMS) - 1)) + "** successful external responses. "
        "This has a different scope from new-attempt costs and excludes failed-attempt charges.",
        "- Failed or ambiguous attempts: **" + str(billing["failed_or_ambiguous_attempts"]) + "**. "
        "Unresolved failed/ambiguous cost: **" + ("yes; missing cost metadata is not zero cost" if billing["failed_or_ambiguous_cost_is_unknown"] else "no") + "**."]
    usage_path = run / "key-usage-after.json"
    if usage_path.exists():
        usage = read(usage_path)
        lines += ["- Later read-only key-usage snapshot (" + str(usage["utc"]) + ", Slurm **"
                  + str(usage["slurm_job_id"]) + "**): usage change USD **"
                  + number(usage["key_usage_delta_usd"], 12) + "**; separately recorded new-response costs USD **"
                  + number(usage["new_response_cost_sum_usd"], 12) + "**. "
                  "These separate measurements are not assumed to be bit-for-bit equal."]
    else:
        lines += ["- Account-usage snapshot change recorded by the audit: USD **"
                  + number(audit["key_usage_snapshot_delta_usd"], 12) + "**. "
                  "This snapshot and response-cost totals are separate measurements; exact reconciliation is not assumed."]
    lines += ["", "## Limits and interpretation boundaries", "",
        "The formatting rule was chosen after inspecting the original eight dev queries. "
        "It was fixed before the remaining recordings, but the public dev data had already informed cutoff selection; "
        "the pilot and remaining-dev cohorts are not independent held-out tests. Practice and dev results are therefore also shown separately.", "",
        "The combined rule changes initialisms, case, punctuation, hyphens and spacing together. "
        "These are not component-specific causal ablations or universal spelling/recognition corrections. "
        "One transcript per audio/model is used; cached pilot calls and new API calls span separate requests, "
        "and the external service aliases do not guarantee immutable model revisions. "
        "The GLM 100-new-token ceiling is retained.", "",
        "No independent reference transcript, WER evaluation, answer-quality evaluation, repeated-ASR uncertainty estimate "
        "or cross-model inference-latency benchmark is included. Original qrels may be incomplete. "
        "A policy using fewer tokens while recovering less evidence is not a matched-recall efficiency win. "
        "No statistical significance or general model-superiority claim is made by these tables.", "",
        "## Reproduction and preserved evidence", "",
        "Use [the frozen full protocol](asr-rag-full-protocol.md), "
        "[experiment runner](../rag/extensions/asr_full_experiment.py), "
        "[Slurm submission](../rag/extensions/run_asr_full_experiment.sbatch), "
        "[cohort summary](../rag/extensions/asr_full_summary.py), "
        "[independent audit](../rag/extensions/audit_asr_full.py) and "
        "[presentation renderer](../rag/extensions/render_asr_full_results.py). "
        "Run coursework only inside Edinburgh Slurm allocations.", "",
        "The private run directory retains the frozen selection/configuration, cache origins, transcript mappings, "
        "request attempts/responses, query vectors, per-query retrieval rows, summaries, billing, environments and audit. "
        "This public note embeds aggregates only; raw audio, transcript text, mappings, responses and credentials are not included.", "",
        "| Audited artifact | SHA256 |", "| --- | --- |"]
    for name in ("config.json", "summary.json", "retrieval.jsonl", "query-vectors.npz"):
        lines.append("| `" + name + "` | `" + audit["artifact_sha256"][name] + "` |")
    output = args.output.resolve()
    require(output not in {(run / (name + ".json")).resolve() for name in inputs}, "Output would overwrite input evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": "rendered", "output": str(output), "cohorts": [name for name, _ in COHORTS],
                      "slurm_job_id": os.environ["SLURM_JOB_ID"]}), flush=True)


if __name__ == "__main__":
    main()
