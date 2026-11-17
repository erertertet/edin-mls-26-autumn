"""Small complete-case and cohort controls; execute only in cluster Slurm."""
import hashlib
from itertools import product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extensions"))
from asr_full_summary import build_summary


def example():
    cohorts = {"all": ["p", "d"], "practice": ["p"], "dev": ["d"],
               "pilot": ["p"], "remaining_dev": ["d"], "remaining_all": ["d"]}
    rows = []
    mappings = []
    for q, condition, arm, rule in product(["p", "d"], ["raw", "normalized"],
                                         ["glm", "whisper", "mai", "voxtral"], ["top5", "cutoff"]):
        row = {"query_id": q, "split": "practice" if q == "p" else "dev",
               "condition": condition, "arm": arm, "rule": rule,
               "recall": 0.5 if q == "p" else 0.25, "context_tokens": 100 if q == "p" else 200,
               "documents": 5 if rule == "top5" else 1, "fallback": rule == "cutoff",
               "record_ids": [q + "-doc"]}
        if rule == "top5" and arm == "whisper":
            if condition == "raw":
                row.update(recall=0.75 if q == "p" else 0.125,
                           context_tokens=150 if q == "p" else 240)
                if q == "p":
                    row["record_ids"] = ["p-other"]
            else:
                row.update(recall=0.5, context_tokens=120 if q == "p" else 180)
        rows.append(row)
    for q, arm in product(["p", "d"], ["glm", "whisper", "mai", "voxtral"]):
        raw = ("ISA" if arm == "whisper" else "I S A") if q == "p" else "query two"
        normalized = "isa" if q == "p" else ("query too" if arm == "whisper" else "query two")
        mappings.append({"query_id": q, "arm": arm, "raw": raw, "normalized": normalized,
                         "raw_text_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                         "normalized_text_sha256": hashlib.sha256(normalized.encode()).hexdigest()})
    return rows, cohorts, mappings


class FullSummaryTests(unittest.TestCase):
    def test_cohorts_paired_changes_and_text_agreement(self):
        rows, cohorts, mappings = example()
        result = build_summary(rows, cohorts, mappings)
        self.assertEqual(result["status"], "complete")
        full = result["cohorts"]["all"]
        raw = next(r for r in full["table"] if (r["condition"], r["arm"], r["rule"]) == ("raw", "whisper", "top5"))
        self.assertEqual(raw["mean_recall"], 0.4375)
        self.assertEqual(raw["delta_recall_pp_vs_glm"], 6.25)
        self.assertEqual(raw["mean_context_tokens"], 195)
        self.assertEqual(raw["delta_context_tokens_vs_glm"], 45)
        changes = next(r for r in full["normalized_minus_raw"] if (r["arm"], r["rule"]) == ("whisper", "top5"))
        self.assertEqual(changes["recall_change_pp"], 6.25)
        self.assertEqual(changes["context_token_change"], -45)
        self.assertEqual(changes["queries_with_recall_improved"], 1)
        self.assertEqual(changes["queries_with_recall_worsened"], 1)
        self.assertEqual(changes["queries_with_changed_selected_ids"], 1)
        self.assertEqual(full["text_agreement"]["raw_identical_queries"], 1)
        self.assertEqual(full["text_agreement"]["normalized_identical_queries"], 1)
        dev = result["cohorts"]["remaining_dev"]
        self.assertEqual(dev["paired_queries"], 1)
        self.assertEqual(next(r for r in dev["table"] if (r["condition"], r["arm"], r["rule"]) == ("raw", "whisper", "top5"))["mean_recall"], 0.125)

    def test_one_missing_cell_excludes_query_from_all_sixteen_rows(self):
        rows, cohorts, mappings = example()
        rows = [r for r in rows if (r["query_id"], r["condition"], r["arm"], r["rule"]) != ("d", "normalized", "voxtral", "cutoff")]
        full = build_summary(rows, cohorts, mappings)["cohorts"]["all"]
        self.assertEqual(full["status"], "partial")
        self.assertFalse(full["expected_complete"])
        self.assertEqual(full["missing_queries"], ["d"])
        self.assertEqual(len(full["missing_cells"]), 1)
        self.assertEqual(full["observed_cells"], 31)
        self.assertEqual(len(full["table"]), 16)
        self.assertTrue(all(row["paired_queries"] == 1 for row in full["table"]))
        raw = next(r for r in full["table"] if (r["condition"], r["arm"], r["rule"]) == ("raw", "whisper", "top5"))
        self.assertEqual(raw["mean_recall"], 0.75)
        self.assertEqual(raw["delta_recall_pp_vs_glm"], 25)

    def test_no_complete_queries_is_not_zero_recall(self):
        _, cohorts, mappings = example()
        full = build_summary([], cohorts, mappings)["cohorts"]["all"]
        self.assertEqual(full["status"], "partial")
        self.assertEqual(full["paired_queries"], 0)
        self.assertTrue(all(r["mean_recall"] is None and r["mean_context_tokens"] is None for r in full["table"]))

    def test_duplicate_cell_and_missing_mapping_are_visible(self):
        rows, cohorts, mappings = example()
        with self.assertRaises(ValueError):
            build_summary(rows + [rows[0]], cohorts, mappings)
        full = build_summary(rows, cohorts, mappings[:-1])["cohorts"]["all"]
        self.assertTrue(full["expected_complete"])
        self.assertEqual(full["status"], "partial")
        self.assertEqual(len(full["text_agreement"]["missing_mappings"]), 1)
        self.assertIsNone(full["text_agreement"]["per_query"][1]["normalized_distinct_texts"])


if __name__ == "__main__":
    unittest.main()
