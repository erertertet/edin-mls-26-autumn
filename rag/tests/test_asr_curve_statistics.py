"""Statistic/estimand controls; execute only in an Edinburgh Slurm allocation."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extensions"))
from asr_curve_statistics import aggregate, mix_at_recall, mix_evaluate


def curve(recall, tokens, documents, hits, macro_precision=None):
    recall, tokens, documents, hits = (np.asarray(x, dtype=float)
                                      for x in (recall, tokens, documents, hits))
    return {"macro_recall": recall, "mean_tokens": tokens, "mean_documents": documents,
            "mean_hits": hits, "micro_precision": hits / documents,
            "macro_precision": hits / documents if macro_precision is None else np.asarray(macro_precision)}


class CurveStatisticsTests(unittest.TestCase):
    def test_macro_micro_and_common_query_multiplicities(self):
        args = (np.array([[1, 2], [0, 1]]), np.array([[1, 4], [3, 4]]),
                np.array([[10, 40], [30, 80]]), np.array([2, 10]))
        result = aggregate(*args)
        np.testing.assert_allclose(result["macro_recall"], [.25, .55])
        np.testing.assert_allclose(result["macro_precision"], [.5, .375])
        np.testing.assert_allclose(result["micro_precision"], [.25, .375])
        weighted = aggregate(*args, weights=[2, 1])
        self.assertAlmostEqual(weighted["macro_recall"][0], 1 / 3)
        self.assertAlmostEqual(weighted["macro_precision"][0], 2 / 3)
        self.assertAlmostEqual(weighted["micro_precision"][0], .4)
        duplicate = aggregate(*(a[[0, 0, 1]] for a in args))
        for field in weighted:
            np.testing.assert_allclose(weighted[field], duplicate[field])

    def test_plateau_cost_tiebreak_and_original_indices(self):
        c = curve([.6, .3, .3 + 5e-13, .45], [100, 40, 30, 70], [10, 4, 3, 7], [3, 1, 1, 2])
        result = mix_at_recall(c, [.8, .95, .96, .9], .3)
        self.assertTrue(result["supported"])
        self.assertEqual((result["low_index"], result["high_index"]), (2, 2))
        self.assertEqual(result["mean_tokens"], 30)
        middle = mix_at_recall(c, [.8, .95, .96, .9], .375)
        self.assertEqual((middle["low_index"], middle["high_index"]), (2, 3))
        self.assertAlmostEqual(middle["achieved_recall"], .375)
        self.assertAlmostEqual(middle["mean_tokens"], 50, places=8)
        tied = curve([.2, .2, .2], [10, 10, 10], [3, 2, 2], [1, 1, 1])
        chosen = mix_at_recall(tied, [1, 3, 2], .2)
        self.assertEqual(chosen["low_index"], 2)

    def test_micro_precision_is_ratio_of_expected_totals(self):
        c = curve([.2, .6], [10, 90], [1, 9], [.5, .9], [.5, .1])
        result = mix_at_recall(c, [1, 9], .4)
        self.assertAlmostEqual(result["high_weight"], .5)
        self.assertAlmostEqual(result["macro_precision"], .3)
        self.assertAlmostEqual(result["micro_precision"], .14)
        self.assertNotAlmostEqual(result["micro_precision"], .3)

    def test_common_policy_weight_is_not_a_query_label_oracle(self):
        c = aggregate([[1, 1], [0, 2]], [[1, 3], [1, 3]], [[10, 30], [10, 30]], [1, 2])
        result = mix_at_recall(c, [1, 3], .75)
        self.assertAlmostEqual(result["high_weight"], .5)
        self.assertAlmostEqual(result["achieved_recall"], .75)
        self.assertAlmostEqual(result["mean_tokens"], 20)
        # Choosing the cheap policy only for q1 and expensive only for q2 would
        # instead give recall=1 at cost=20; this function does not do that.
        self.assertNotAlmostEqual(result["achieved_recall"], 1)

    def test_evaluation_keeps_calibration_weight_despite_recall_shift(self):
        calibration = curve([.2, .6], [10, 90], [1, 9], [.5, .9])
        evaluation = curve([.1, .3], [12, 96], [1, 9], [.2, .6])
        selected = mix_at_recall(calibration, [1, 9], .4)
        result = mix_evaluate(evaluation, selected)
        self.assertAlmostEqual(result["achieved_recall"], .2)
        self.assertAlmostEqual(result["mean_tokens"], 54)
        self.assertAlmostEqual(result["high_weight"], selected["high_weight"])
        self.assertFalse(mix_at_recall(evaluation, [1, 9], .4)["supported"])

    def test_unreachable_targets_endpoints_and_incomplete_selections(self):
        c = curve([.2, .6], [10, 90], [1, 9], [.5, .9])
        for target in [.1, .7, .2 - 1e-14, .6 + 1e-14]:
            unsupported = mix_at_recall(c, [1, 9], target)
            self.assertFalse(unsupported["supported"])
            self.assertIsNone(unsupported["low_index"])
            self.assertTrue(np.isnan(unsupported["mean_tokens"]))
            self.assertFalse(mix_evaluate(c, unsupported)["supported"])
        for target, expected in [(.2, 0), (.6, 1)]:
            result = mix_at_recall(c, [1, 9], target)
            self.assertEqual((result["low_index"], result["high_index"]), (expected, expected))
            self.assertEqual(result["high_weight"], 0)
        for invalid in [None, {}, {"supported": True, "low_index": 0}]:
            with self.assertRaises(ValueError):
                mix_evaluate(c, invalid)
        bad = dict(mix_at_recall(c, [1, 9], .4), high_weight=1.1)
        with self.assertRaises(ValueError):
            mix_evaluate(c, bad)

    def test_invalid_counts_weights_and_curve_ratios_fail_explicitly(self):
        for args, weights in [(([[1]], [[0]], [[10]], [1]), None),
                              (([[2]], [[1]], [[10]], [2]), None),
                              (([[1]], [[1]], [[10]], [0]), None),
                              (([[1]], [[1]], [[10]], [1]), [0]),
                              (([[1]], [[1]], [[10]], [1]), [-1])]:
            with self.assertRaises(ValueError):
                aggregate(*args, weights=weights)
        bad = curve([.2, .6], [10, 90], [1, 9], [.5, .9])
        bad["micro_precision"] = np.array([.9, .9])
        with self.assertRaises(ValueError):
            mix_at_recall(bad, [1, 9], .4)


if __name__ == "__main__":
    unittest.main()
