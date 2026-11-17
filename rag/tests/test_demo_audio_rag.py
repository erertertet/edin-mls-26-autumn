"""CPU-only demo policy/launcher checks. Run on Fuji3, not the local GPU."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/demo_audio_rag.py"
spec = importlib.util.spec_from_file_location("demo_audio_rag", path)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class DemoPolicyTests(unittest.TestCase):
    def test_reference_is_optional_for_audio_demo(self):
        args = demo.parser().parse_args(["question.mp3"])
        self.assertIsNone(demo.reference_text(args))
        self.assertFalse(args.retrieval_only)

    def test_inline_reference_and_empty_validation(self):
        args = demo.parser().parse_args(["question.mp3", "--gold-text", "  Independent evidence.  "])
        self.assertEqual(demo.reference_text(args), "Independent evidence.")
        args.gold_text = "  "
        with self.assertRaises(ValueError):
            demo.reference_text(args)

    def test_absolute_includes_boundary_and_falls_back(self):
        self.assertEqual(demo.selected_count([0.95, 0.90, 0.90, 0.89], "absolute", tau=0.90), (3, False))
        self.assertEqual(demo.selected_count([0.8, 0.7], "absolute", tau=0.9), (1, True))

    def test_relative_compares_with_top_not_adjacent_scores(self):
        self.assertEqual(demo.selected_count([1, 0.875, 0.75], "relative", delta=0.125), (2, False))

    def test_no_cutoff_document_cap(self):
        self.assertEqual(demo.selected_count([0.95] * 300, "absolute", tau=0.9), (300, False))
        self.assertEqual(demo.selected_count([0.95] * 300, "relative", delta=0.009), (300, False))

    def test_three_example_counts(self):
        scores = [0.895, 0.891, 0.889, 0.880]
        self.assertEqual([demo.selected_count(scores, p)[0] for p in demo.POLICIES], [2, 1, 3])

    def test_invalid_scores_and_policy(self):
        for scores in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                demo.selected_count(scores, "top2")
        with self.assertRaises(ValueError):
            demo.selected_count([1], "unknown")

    def test_ssh_hostname_cannot_inject_options_or_commands(self):
        self.assertEqual(demo.ssh_args("fuji3")[-1], "fuji3")
        for host in ("-oProxyCommand=bad", "fuji3; echo bad", "fuji3 bad"):
            with self.assertRaises(ValueError):
                demo.ssh_args(host)


if __name__ == "__main__":
    unittest.main(verbosity=2)
