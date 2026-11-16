"""Formatting equivalences and semantic-preservation checks; run in Slurm."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extensions"))
from transcript_normalization import normalize_transcript as normalize


class NormalizationTests(unittest.TestCase):
    def test_equivalent_initialisms(self):
        for variants in [("ISA", "I S A", "I.S.A.", "I. S. A."),
                         ("US", "U.S.", "U S", "u.s."),
                         ("UK", "U.K.", "U K")]:
            with self.subTest(variants=variants):
                values = {normalize("Use " + variant + " today!") for variant in variants}
                self.assertEqual(len(values), 1)

    def test_punctuation_without_word_concatenation(self):
        self.assertEqual(normalize("Long-term; SELF-employment?"), "long term self employment")
        self.assertEqual(normalize("transaction? Records online."), normalize("transaction records online"))
        self.assertEqual(normalize("  Café\u00a0[notes] \n "), "café notes")

    def test_preserve_meaning_bearing_characters(self):
        self.assertEqual(normalize("I can’t use Jane's £1,000.50 (net) at -5%."),
                         "i can't use jane's £1,000.50 net at -5%")
        self.assertNotEqual(normalize("can not invest"), normalize("can invest"))
        self.assertNotEqual(normalize("I am a saver"), normalize("I am saver"))
        self.assertNotEqual(normalize("A? B!"), normalize("AB"))
        self.assertEqual(normalize("i a"), "i a")

    def test_idempotent(self):
        for text in ["I S A", "U.S.", "U. S. A.", "Long-term café’s £1,000.50", "A? B!", "ＵＫ"]:
            with self.subTest(text=text):
                self.assertEqual(normalize(normalize(text)), normalize(text))


if __name__ == "__main__":
    unittest.main()
