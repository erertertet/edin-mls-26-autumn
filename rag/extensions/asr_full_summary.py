"""Paired summaries for a frozen four-model, two-condition retrieval experiment."""
from itertools import product
import math
import statistics

ARMS = ("glm", "whisper", "mai", "voxtral")
CONDITIONS = ("raw", "normalized")
RULES = ("top5", "cutoff")
SETTINGS = tuple(product(CONDITIONS, ARMS, RULES))


def _mean(values):
    return statistics.mean(values) if values else None


def build_summary(rows, cohorts, mappings):
    """Summarize every setting on one complete-query intersection per cohort.

    Missing cells remain explicit. Text agreement uses the four-model mapping
    intersection, whose coverage is reported separately from metric coverage.
    This function performs no model, file, network or normalization operations.
    """
    if "all" not in cohorts:
        raise ValueError("The frozen cohorts must include all")
    universe = set(cohorts["all"])
    for name, ids in cohorts.items():
        if len(ids) != len(set(ids)) or not set(ids).issubset(universe):
            raise ValueError("Duplicate or unknown query in cohort: " + name)
    cells = {}
    for row in rows:
        setting = (row["condition"], row["arm"], row["rule"])
        key = (row["query_id"], *setting)
        if row["query_id"] not in universe or setting not in SETTINGS or key in cells:
            raise ValueError("Duplicate or unplanned retrieval cell: " + str(key))
        if not math.isfinite(row["recall"]) or not 0 <= row["recall"] <= 1:
            raise ValueError("Invalid recall: " + str(key))
        if (not math.isfinite(row["context_tokens"]) or not math.isfinite(row["documents"])
                or row["context_tokens"] < 0 or row["documents"] < 1):
            raise ValueError("Invalid token/document count: " + str(key))
        cells[key] = row
    mapped = {}
    for item in mappings:
        key = (item["query_id"], item["arm"])
        if key[0] not in universe or key[1] not in ARMS or key in mapped:
            raise ValueError("Duplicate or unplanned transcript mapping: " + str(key))
        if not isinstance(item["raw"], str) or not isinstance(item["normalized"], str):
            raise ValueError("Transcript mapping values must be strings")
        mapped[key] = item

    summaries = {}
    for name, selected in cohorts.items():
        missing = [{"query_id": q, "condition": condition, "arm": arm, "rule": rule}
                   for q in selected for condition, arm, rule in SETTINGS
                   if (q, condition, arm, rule) not in cells]
        missing_ids = {item["query_id"] for item in missing}
        paired = [q for q in selected if q not in missing_ids]
        n = len(paired)
        table = []
        changes = []
        for condition, rule, arm in product(CONDITIONS, RULES, ARMS):
            values = [cells[q, condition, arm, rule] for q in paired]
            baseline = [cells[q, condition, "glm", rule] for q in paired]
            table.append({"condition": condition, "rule": rule, "arm": arm, "paired_queries": n,
                "mean_recall": _mean([r["recall"] for r in values]),
                "mean_context_tokens": _mean([r["context_tokens"] for r in values]),
                "mean_documents": _mean([r["documents"] for r in values]),
                "fallback_queries": sum(r["fallback"] for r in values),
                "delta_recall_pp_vs_glm": _mean([100 * (r["recall"] - b["recall"])
                                                 for r, b in zip(values, baseline)]),
                "delta_context_tokens_vs_glm": _mean([r["context_tokens"] - b["context_tokens"]
                                                     for r, b in zip(values, baseline)])})
        for rule, arm in product(RULES, ARMS):
            raw = [cells[q, "raw", arm, rule] for q in paired]
            normalized = [cells[q, "normalized", arm, rule] for q in paired]
            recall_delta = [n["recall"] - r["recall"] for r, n in zip(raw, normalized)]
            token_delta = [n["context_tokens"] - r["context_tokens"] for r, n in zip(raw, normalized)]
            changes.append({"arm": arm, "rule": rule, "paired_queries": len(paired),
                "recall_change_pp": _mean([100 * value for value in recall_delta]),
                "context_token_change": _mean(token_delta),
                "queries_with_recall_improved": sum(value > 0 for value in recall_delta),
                "queries_with_recall_worsened": sum(value < 0 for value in recall_delta),
                "queries_with_recall_unchanged": sum(value == 0 for value in recall_delta),
                "queries_with_token_increase": sum(value > 0 for value in token_delta),
                "queries_with_token_decrease": sum(value < 0 for value in token_delta),
                "queries_with_token_unchanged": sum(value == 0 for value in token_delta),
                "queries_with_changed_selected_ids": sum(r["record_ids"] != n["record_ids"]
                                                         for r, n in zip(raw, normalized))})
        missing_mappings = [{"query_id": q, "arm": arm} for q in selected for arm in ARMS
                            if (q, arm) not in mapped]
        complete_mapped = [q for q in selected if all((q, arm) in mapped for arm in ARMS)]
        per_query = []
        for q in selected:
            available = [mapped[q, arm] for arm in ARMS if (q, arm) in mapped]
            complete = len(available) == len(ARMS)
            per_query.append({"query_id": q, "available_models": len(available),
                "raw_distinct_texts": len({item["raw"] for item in available}) if complete else None,
                "normalized_distinct_texts": len({item["normalized"] for item in available}) if complete else None})
        expected_complete = not missing
        status = "complete" if expected_complete and not missing_mappings else "partial"
        summaries[name] = {"status": status, "expected_complete": expected_complete,
            "expected_queries": len(selected), "expected_cells": len(selected) * len(SETTINGS),
            "observed_cells": len(selected) * len(SETTINGS) - len(missing),
            "paired_queries": n, "complete_paired_queries": paired,
            "missing_queries": [q for q in selected if q in missing_ids], "missing_cells": missing,
            "table": table, "normalized_minus_raw": changes,
            "text_agreement": {"complete_mapped_queries": complete_mapped,
                "missing_mappings": missing_mappings,
                "raw_identical_queries": sum(item["raw_distinct_texts"] == 1 for item in per_query),
                "normalized_identical_queries": sum(item["normalized_distinct_texts"] == 1 for item in per_query),
                "per_query": per_query}}
    return {"status": "complete" if all(c["status"] == "complete" for c in summaries.values()) else "partial",
            "expected_cells_per_query": len(SETTINGS), "observed_cells": len(rows), "cohorts": summaries}
