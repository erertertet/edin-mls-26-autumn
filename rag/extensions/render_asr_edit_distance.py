"""Render audited, saved ASR edit-distance aggregates without new inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket


COHORTS = {"all": 475, "practice": 350, "dev": 125, "pilot": 8,
           "remaining_dev": 117, "remaining_all": 467}
PAIRS = ("overall", "glm__whisper", "glm__mai", "glm__voxtral",
         "whisper__mai", "whisper__voxtral", "mai__voxtral")
UNITS = ("characters", "words")
ARMS = {"glm": "GLM-ASR-Nano", "whisper": "Whisper large-v3",
        "mai": "MAI-Transcribe 2", "voxtral": "Voxtral Small Transcribe 3"}
NORMALIZER_SHA256 = "fff6993b2f1ae85db3a65a6c645a6e2ac3ad0ec415eb82af3f73a763de8e9dd2"
METRICS = ("absolute", "normalized_distance")
ESTIMATES = ("raw", "normalized", "delta")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def number(value, scale=1, digits=3, signed=False):
    require(finite(value), "Cannot render a nonfinite estimate")
    return format(value * scale, ("+" if signed else "") + "." + str(digits) + "f")


def estimate(row, metric, condition, scale=1, digits=3, signed=False):
    item = row[metric]
    ci = item[condition + "_ci"]
    return (number(item[condition + "_mean"], scale, digits, signed) + " ["
            + number(ci["low"], scale, digits, signed) + ", "
            + number(ci["high"], scale, digits, signed) + "]")


def pair_label(pair):
    if pair == "overall":
        return "Overall: six-pair query mean"
    return " / ".join(ARMS[arm] for arm in pair.split("__"))


def validate_summary(data):
    require(data.get("status") == "complete", "Edit-distance summary is not complete")
    metadata = data["metadata"]
    require(metadata["normalizer_sha256"] == NORMALIZER_SHA256, "Frozen normalizer SHA mismatch")
    for field in ("new_asr_calls", "new_encoder_calls", "new_api_calls"):
        require(metadata.get(field) == 0, "Missing zero-activity evidence: " + field)
    bootstrap = metadata["bootstrap"]
    require(bootstrap["replicates"] == 5000 and bootstrap["seed"] == 20261117
            and bootstrap["method"] == "percentile" and bootstrap["confidence"] == 0.95,
            "Unexpected frozen bootstrap settings")
    require(metadata.get("change_tolerance") == 1e-12, "Unexpected NED-change tie tolerance")
    order = metadata["cohort_order"]
    require(len(order) == len(COHORTS) and set(order) == set(COHORTS), "Unexpected bootstrap cohort order")
    require(set(metadata["pair_labels"]) == set(PAIRS) and set(metadata["units"]) == set(UNITS),
            "Unexpected declared pair/unit coverage")
    require(set(data["cohorts"]) == set(COHORTS), "Unexpected cohort coverage")
    for cohort, count in COHORTS.items():
        require(data["cohorts"][cohort]["queries"] == count, "Unexpected query count: " + cohort)
        expected_seed = bootstrap["seed"] + order.index(cohort)
        require(bootstrap["seeds"][cohort] == data["cohorts"][cohort]["seed"] == expected_seed,
                "Cohort bootstrap seed mismatch")
    rows = data["rows"]
    require(len(rows) == 84, "Expected all 84 aggregate rows")
    indexed = {}
    for row in rows:
        key = (row["cohort"], row["pair"], row["unit"])
        require(key[0] in COHORTS and key[1] in PAIRS and key[2] in UNITS,
                "Unexpected aggregate row key")
        require(key not in indexed, "Duplicate aggregate row key")
        count = row["queries"]
        require(integer(count) and count == COHORTS[key[0]], "Wrong row query count")
        indexed[key] = row
        for metric in METRICS:
            values = row[metric]
            for condition in ESTIMATES:
                require(finite(values[condition + "_mean"]), "Nonfinite aggregate mean")
                ci = values[condition + "_ci"]
                require(finite(ci["low"]) and finite(ci["high"]) and ci["low"] <= ci["high"],
                        "Missing or unordered aggregate interval")
            require(math.isclose(values["delta_mean"], values["normalized_mean"] - values["raw_mean"],
                                 rel_tol=1e-9, abs_tol=1e-11), "Paired mean delta mismatch")
            for condition in ("raw", "normalized"):
                require(values[condition + "_mean"] >= 0 and values[condition + "_ci"]["low"] >= -1e-12,
                        "Negative distance estimate")
                if metric == "normalized_distance":
                    require(values[condition + "_mean"] <= 1
                            and values[condition + "_ci"]["high"] <= 1 + 1e-12,
                            "Normalized distance outside [0,1]")
        zero = row["zero_distance"]
        require(all(integer(zero[field]) and 0 <= zero[field] <= count
                    for field in ("raw_queries", "normalized_queries")), "Invalid zero-distance counts")
        changes = row["change_counts"]
        require(all(integer(changes[field]) and changes[field] >= 0
                    for field in ("decrease", "tie", "increase"))
                and sum(changes[field] for field in ("decrease", "tie", "increase")) == count,
                "Incomplete paired change counts")
    for cohort in COHORTS:
        for unit in UNITS:
            overall = indexed[(cohort, "overall", unit)]
            for metric in METRICS:
                for condition in ESTIMATES:
                    expected = sum(indexed[(cohort, pair, unit)][metric][condition + "_mean"]
                                   for pair in PAIRS[1:]) / 6
                    require(math.isclose(overall[metric][condition + "_mean"], expected,
                                         rel_tol=1e-9, abs_tol=1e-11), "Overall six-pair mean mismatch")
        identity = data["cohorts"][cohort]["character_identity"]
        require(identity == indexed[(cohort, "overall", "characters")]["zero_distance"],
                "Character identity and zero-distance counts differ")
    require(indexed[("all", "overall", "characters")]["zero_distance"]
            == {"raw_queries": 195, "normalized_queries": 390},
            "Original 195/390 exact-character identity was not reproduced")
    return indexed


def validate_audit(audit, run, metadata):
    require(audit.get("status") == "pass" and not audit.get("errors"), "Independent audit has not passed")
    require(str(audit["analysis_slurm_job_id"]) == str(metadata["slurm_job_id"]),
            "Audit refers to another analysis job")
    inputs = audit["input_sha256"]
    require(inputs["summary.json"] == digest(run / "summary.json"), "Audit summary hash mismatch")
    required = {"summary.json", "analysis-config.json", "distances.npz", "bootstrap-values.npz"}
    require(required.issubset(inputs), "Audit is missing required evidence hashes")
    for name, value in inputs.items():
        require(name in required | {"completion.json"}, "Unexpected audited input filename")
        require(name in inputs and valid_digest(inputs[name]), "Audit does not bind required input: " + name)
        require(digest(run / name) == value, "Audited input changed: " + name)
    require(audit["summary_rows_checked"] == 84
            and audit["all_unit_comparisons_checked"] == 11400
            and audit["transcript_mappings_checked"] == 1900,
            "Incomplete independent audit coverage")
    require(audit["bootstrap_replicates_per_cohort"] == 5000,
            "Audit bootstrap coverage mismatch")


def write_csv(rows, path):
    fields = ["cohort", "pair", "unit", "queries"]
    for metric in METRICS:
        for condition in ESTIMATES:
            fields += [metric + "_" + condition + "_mean",
                       metric + "_" + condition + "_ci_low",
                       metric + "_" + condition + "_ci_high"]
    fields += ["zero_distance_raw_queries", "zero_distance_normalized_queries",
               "ned_decrease_queries", "ned_tie_queries", "ned_increase_queries"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {field: row[field] for field in fields[:4]}
            for metric in METRICS:
                for condition in ESTIMATES:
                    prefix = metric + "_" + condition
                    flat[prefix + "_mean"] = row[metric][condition + "_mean"]
                    for bound in ("low", "high"):
                        flat[prefix + "_ci_" + bound] = row[metric][condition + "_ci"][bound]
            for condition in ("raw", "normalized"):
                flat["zero_distance_" + condition + "_queries"] = row["zero_distance"][condition + "_queries"]
            for direction in ("decrease", "tie", "increase"):
                flat["ned_" + direction + "_queries"] = row["change_counts"][direction]
            writer.writerow(flat)


def table(rows, include_unit=False, changes=False):
    headers = ["Model pair"] + (["Unit"] if include_unit else [])
    headers += ["Mean edits: raw → normalized", "Mean NED (%): raw → normalized",
                "Paired NED Δ (pp), 95% CI", "Zero queries: raw → normalized"]
    if changes:
        headers += ["NED decrease / tie / increase"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = [pair_label(row["pair"])] + ([row["unit"]] if include_unit else [])
        cells += [estimate(row, "absolute", "raw") + " → " + estimate(row, "absolute", "normalized"),
                  estimate(row, "normalized_distance", "raw", scale=100)
                  + " → " + estimate(row, "normalized_distance", "normalized", scale=100),
                  estimate(row, "normalized_distance", "delta", scale=100, signed=True),
                  str(row["zero_distance"]["raw_queries"]) + " → "
                  + str(row["zero_distance"]["normalized_queries"])]
        if changes:
            cells += [" / ".join(str(row["change_counts"][direction])
                                 for direction in ("decrease", "tie", "increase"))]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def report_lines(data, indexed, audit, csv_name):
    metadata = data["metadata"]
    lines = ["# Four-ASR transcript disagreement: edit distances", "",
             "This saved-data follow-up measures the magnitude of differences among four ASR outputs for all "
             "**475** supplied audio queries. It compares the existing raw text with the saved frozen-normalized "
             "text; no transcript or normalization rule is changed. **New ASR calls: 0; encoder calls: 0; external API calls: 0.**", "",
             "## Findings", ""]
    for unit in UNITS:
        row = indexed[("all", "overall", unit)]
        lines.append("- Across all queries, the six-pair mean " + unit + " NED changes from "
                     + number(row["normalized_distance"]["raw_mean"], scale=100) + "% to "
                     + number(row["normalized_distance"]["normalized_mean"], scale=100)
                     + "%; paired change "
                     + estimate(row, "normalized_distance", "delta", scale=100, signed=True) + " percentage points.")
    lines += ["", "Four-model exact-character identity reproduces **195/475 raw → 390/475 normalized** queries. "
              "The remaining **85** normalized text disagreements are retained in every all-query mean. "
              "These distances quantify output convergence; they do not establish better recognition accuracy or retrieval.", "",
              "## All-query overall comparison", "",
              "Every estimate in the tables is a mean with a pointwise 95% percentile interval in brackets. "
              "Δ is normalized minus raw; a negative NED Δ means less inter-model disagreement. "
              "The overall value averages six model-pair distances within each query, then averages queries.", ""]
    lines += table([indexed[("all", "overall", unit)] for unit in UNITS], include_unit=True, changes=True)
    for cohort, title in (("all", "All queries"), ("remaining_dev", "Remaining dev")):
        lines += ["", "## " + title + ": six model pairs (n=" + str(COHORTS[cohort]) + ")", ""]
        if cohort == "remaining_dev":
            lines += ["This is the same 117-query evaluation cohort used in the prior calibration-transfer analysis. "
                      "It has already been inspected; this edit-distance follow-up is descriptive, not a new holdout.", ""]
        for unit in UNITS:
            lines += ["### " + unit.title(), ""]
            lines += table([indexed[(cohort, pair, unit)] for pair in PAIRS[1:]], changes=True)
    lines += ["", "## Definitions and interpretation", "",
              "Both metrics use unit-cost Levenshtein insertions, deletions and substitutions (each cost 1). "
              "Character distance operates on Python Unicode code points, preserving raw case, punctuation and whitespace. "
              "Word distance operates on `text.split()` sequences: case and punctuation remain significant, while pure "
              "whitespace differences disappear. No extra processor or raw-text cleanup is used. "
              "The implementation reuses [RapidFuzz's Levenshtein API](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html), "
              "pinned to [RapidFuzz 3.14.6](https://pypi.org/project/RapidFuzz/3.14.6/).", "",
              "For each query and model pair, `NED = distance / max(left_length, right_length)` in the selected unit. "
              "This symmetric ratio is 0 for two empty sequences and 1 when only one is empty. "
              "Reported NED is the mean of per-query ratios, not total edits divided by total length. "
              "Normalization changes lengths and token boundaries, so absolute-edit changes and NED changes can differ. "
              "The NED change counts use a tie tolerance of `abs(delta) <= 1e-12`.", "",
              "A pair's zero count measures equality in that unit. For the overall row it requires zero distance "
              "for all six pairs. Character zero therefore means all four saved strings are exactly identical; "
              "word zero only means identical whitespace-token sequences and can hide spacing differences.", "",
              "These are **reference-free disagreement measures, not WER/CER**: there is no independent reference "
              "transcript in this analysis. Shared wrong words can yield zero distance, and a small edit can alter "
              "negation or an important entity. Reduced disagreement after the frozen formatting rule does not "
              "demonstrate lower transcription error, semantic equivalence, higher evidence recall, lower context "
              "cost or improved downstream answers. Those retrieval questions remain in the "
              "[full ASR/retrieval results](asr-rag-full-results.md) and "
              "[matched-recall analysis](asr-matched-curves-results.md).", "",
              "## Paired uncertainty and cohort limits", "",
              "Each cohort uses **5,000 query-level bootstrap replicates**, percentile 95% intervals, base seed "
              "**20261117** plus its saved cohort-order index. Within each replicate, the same sampled query "
              "indices apply to all six pairs, both units and both text conditions. Paired normalized-minus-raw "
              "intervals use those shared replicate means. The six pairs are not independent observations; "
              "the overall interval resamples queries, not 2,850 pair rows. Intervals are pointwise and "
              "unadjusted for multiple comparisons. They reflect query resampling conditional on the fixed saved "
              "transcripts, excluding ASR generation and provider variability.", "",
              "The six cohorts overlap: all 475 = practice 350 + dev 125; dev 125 = pilot 8 + remaining dev 117; "
              "remaining all 467 excludes the pilot. They are not six independent replications. The normalization "
              "rule was developed after examining the pilot, and all-query results have already been inspected. "
              "The pilot intervals are especially limited by n=8. Bootstrap uncertainty does not account for "
              "shared ASR biases, changes to hosted model versions, or earlier analytical choices.", "",
              "## Provenance and independent implementation check", "",
              "- Analysis Slurm job: `" + str(metadata["slurm_job_id"]) + "`.",
              "- Independent audit Slurm job: `" + str(audit["slurm_job_id"]) + "`; status **pass**.",
              "- Analysis Git commit: `" + str(metadata["git_head"]) + "`.",
              "- Frozen normalization policy: `" + str(metadata["policy_version"]) + "`.",
              "- Frozen normalizer SHA256: `" + NORMALIZER_SHA256 + "`.",
              "- Audit checked **1,900** mappings, **11,400** character/word distance values and **84** aggregate rows, "
              "with **5,000** bootstrap replicates per cohort.", "",
              "The audit reconstructs aggregates and intervals from saved inputs and cross-checks distances "
              "against RapidFuzz's separate Python implementation. This is an implementation check using "
              "the same library project, not an independent scientific replication. Original full-run "
              "source hashes remain bound in the audit. The [frozen edit-distance protocol](asr-edit-distance-protocol.md) "
              "describes the definitions and source checks.", "",
              "| Audited analysis input | SHA256 |", "| --- | --- |"]
    for name, value in sorted(audit["input_sha256"].items()):
        require(valid_digest(value), "Invalid audited input SHA")
        lines.append("| `" + Path(name).name + "` | `" + value + "` |")
    if audit.get("full_input_sha256"):
        lines += ["", "| Original full-run evidence | SHA256 |", "| --- | --- |"]
        for name, value in sorted(audit["full_input_sha256"].items()):
            require(valid_digest(value), "Invalid original full-run evidence SHA")
            lines.append("| `" + Path(name).name + "` | `" + value + "` |")
    lines += ["", "## Aggregate artifacts and remaining cohorts", "",
              "[All 84 aggregate rows as CSV](" + csv_name + ") include unrounded estimates and intervals for "
              "absolute edits and NED. The CSV contains no transcript text or query IDs. Per-query distances, "
              "transcripts and bootstrap query indices remain private.", "",
              "<details>", "<summary>Remaining cohort rows (including the remaining-dev overall values)</summary>", ""]
    appendix = [("remaining_dev", ("overall",))] + [(cohort, PAIRS)
                for cohort in ("practice", "dev", "pilot", "remaining_all")]
    for cohort, pairs in appendix:
        lines += ["### " + cohort + " (n=" + str(COHORTS[cohort]) + ")", ""]
        lines += table([indexed[(cohort, pair, unit)] for unit in UNITS for pair in pairs],
                       include_unit=True, changes=True)
        lines += [""]
    lines += ["</details>"]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"),
            "Run only inside an Edinburgh Slurm allocation")
    os.umask(0o077)
    run, output = args.run.resolve(), args.output.resolve()
    require(output.suffix.lower() == ".md", "Output must be a Markdown path")
    summary_path, audit_path = run / "summary.json", run / "audit.json"
    require(summary_path.is_file() and audit_path.is_file(), "Complete summary and passing audit are required")
    data, audit = read(summary_path), read(audit_path)
    indexed = validate_summary(data)
    validate_audit(audit, run, data["metadata"])
    csv_path = output.parent / "asr-edit-distance.csv"
    manifest_path = output.parent / "render-manifest.json"
    sources = {path.resolve() for path in (summary_path, audit_path, Path(__file__))}
    require(not sources.intersection({output, csv_path.resolve(), manifest_path.resolve()}),
            "Renderer output would overwrite source evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [indexed[(cohort, pair, unit)] for cohort in COHORTS for pair in PAIRS for unit in UNITS]
    write_csv(rows, csv_path)
    output.write_text("\n".join(report_lines(data, indexed, audit, csv_path.name)) + "\n", encoding="utf-8")
    manifest = {"status": "rendered", "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "analysis_slurm_job_id": data["metadata"]["slurm_job_id"],
                "audit_slurm_job_id": audit["slurm_job_id"], "aggregate_rows": len(rows),
                "renderer_source_sha256": digest(Path(__file__)),
                "input_sha256": {"summary.json": digest(summary_path), "audit.json": digest(audit_path)},
                "output_sha256": {path.name: digest(path) for path in (output, csv_path)},
                "new_asr_calls": 0, "new_encoder_calls": 0, "new_api_calls": 0}
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
