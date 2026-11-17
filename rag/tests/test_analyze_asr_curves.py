"""End-to-end estimand/pairing controls, executed only in cluster Slurm."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.stats import bootstrap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extensions"))
from analyze_asr_curves import CurveInference, METRICS, build_contrasts


def synthetic():
    cases = np.asarray([(a, c) for a in ("glm", "whisper", "mai", "voxtral") for c in ("raw", "normalized")])
    hits = np.tile([1, 2, 2, 1], (9, 8, 1)).astype(np.int32)
    docs = np.tile([1, 5, 3, 1], (9, 8, 1)).astype(np.int32)
    cache = {"query_ids": np.asarray([f"q{i}" for i in range(9)]), "cases": cases,
             "parameters": np.asarray([1, 5, .8925, 1.000001]),
             "families": np.asarray(["topk", "topk", "cutoff", "cutoff"]),
             "hits": hits, "docs": docs, "tokens": 10 * docs,
             "gold": np.asarray([2] * 5 + [4] * 3 + [2])}
    partition = {"calibration_ids": [f"q{i}" for i in range(5)],
                 "evaluation_ids": [f"q{i}" for i in range(5, 8)], "pilot_ids": ["q8"]}
    plan = {"targets": [.5, .75], "fixed_k": 5, "fixed_tau": .8925,
            "bootstrap": {"replicates": 50, "minimum_supported_fraction": .95}}
    return CurveInference(cache, partition, plan), plan


class AnalysisControls(unittest.TestCase):
    def test_calibration_does_not_force_evaluation_recall(self):
        inference, _ = synthetic()
        cm = inference.cal_matrix.mean(axis=0).reshape(inference.mean_shape)
        selected = inference.calibration_choices(cm)
        em = inference.eval_matrix.mean(axis=0).reshape(inference.mean_shape)
        _, points = inference.evaluate(cm, em, selected, details=True)
        deployed = next(p for p in points if p["panel"] == "calibrated" and p["target"] == .75)
        self.assertAlmostEqual(deployed["calibration_recall"], .75)
        self.assertAlmostEqual(deployed["metrics"]["achieved_recall"], .375)
        standardized = [p for p in points if p["panel"] == "matched" and p["target"] == .75]
        self.assertTrue(all(not p["common_target_supported"] for p in standardized))
        self.assertTrue(all(np.isnan(p["metrics"]["mean_tokens"]) for p in standardized))

    def test_two_group_bootstrap_preserves_cross_model_pairing(self):
        inference, plan = synthetic()
        result = bootstrap((np.arange(5), np.arange(3)), inference.statistic, n_resamples=50,
                           paired=False, vectorized=True, axis=-1, batch=10, method="percentile",
                           rng=np.random.default_rng(26))
        self.assertEqual(result.bootstrap_distribution.shape, (len(inference.specs) * len(METRICS), 50))
        self.assertEqual(np.concatenate(inference.cal_draws).shape, (50, 5))
        self.assertEqual(np.concatenate(inference.eval_draws).shape, (50, 3))
        cm = inference.cal_matrix.mean(axis=0).reshape(inference.mean_shape)
        em = inference.eval_matrix.mean(axis=0).reshape(inference.mean_shape)
        _, points = inference.evaluate(cm, em, details=True)
        draws = result.bootstrap_distribution.reshape(len(points), len(METRICS), 50)
        contrasts = build_contrasts(points, draws, plan)
        comparable = [c for c in contrasts if c["kind"] in ("normalization", "asr_vs_glm")
                      and np.isfinite(c["metrics"]["mean_tokens"])]
        self.assertTrue(comparable)
        for contrast in comparable:
            for metric in METRICS:
                self.assertEqual(contrast["metrics"][metric], 0)
                self.assertEqual(contrast["intervals"][metric]["low"], 0)
                self.assertEqual(contrast["intervals"][metric]["high"], 0)
                self.assertEqual(contrast["intervals"][metric]["valid_bootstraps"], 50)


if __name__ == "__main__":
    unittest.main()
