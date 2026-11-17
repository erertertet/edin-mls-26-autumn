"""Verify installed CPU APIs and distinct full-set estimands inside Slurm."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import socket

import numpy as np
import rapidfuzz
from rapidfuzz.distance import Levenshtein, Levenshtein_py


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), (actual, expected)


def failure(function, *args):
    try:
        function(*args)
    except RuntimeError:
        return
    raise AssertionError("Malformed numerical input was accepted")


def main():
    assert os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk")
    assert np.__version__ == "2.4.6" and rapidfuzz.__version__ == "3.14.6"
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path("/home/s2487016/.cache/mls-asr-full-e2e")
    parser.add_argument("--summary", type=Path, default=cache / "summarize_asr_full_e2e.py")
    parser.add_argument("--auditor", type=Path, default=cache / "audit_asr_full_e2e.py")
    args = parser.parse_args()
    summary = load("full_e2e_summary_probe", args.summary)
    auditor = load("full_e2e_auditor_probe", args.auditor)

    # Generic hashable sequences, empty conventions, punctuation/case and Unicode code points.
    cases = [("kitten", "sitting", 3), ([], [], 0), ([], [1], 1),
             ([1, 2], [1, 3, 2], 1), ("A B", "A  B", 1),
             ("a", "A", 1), ("A.", "A", 1), ("a\u0301", "a", 1),
             (["A", "B"], ["A", "X", "B"], 1)]
    for left, right, expected in cases:
        maximum = max(len(left), len(right))
        ratio = expected / maximum if maximum else 0.0
        for implementation in (Levenshtein, Levenshtein_py):
            assert implementation.distance(left, right, weights=(1, 1, 1), processor=None) == expected
            assert implementation.distance(right, left, weights=(1, 1, 1), processor=None) == expected
            close(implementation.normalized_distance(left, right, weights=(1, 1, 1), processor=None), ratio)
        wanted = {"distance": expected, "max_length": maximum, "normalized_distance": ratio}
        assert summary.distance_record(left, right) == auditor.distance(left, right) == wanted
    spaced = {"reference": {"token_ids": [1, 2], "text_stripped": "A B"},
              "cuda": {"token_ids": [1, 3, 2], "text_stripped": "A  B"}}
    spaced_edits = summary.query_edits(spaced)
    assert [spaced_edits[unit]["distance"] for unit in ("tokens", "characters", "words")] == [1, 1, 0]

    # Unequal query sizes and unequal relative L2: macro and pooled quantities must differ.
    first_reference = np.array([[1., 0.]], dtype=np.float32)
    first_candidate = np.array([[2., 0.]], dtype=np.float32)
    second_reference = np.ones((2, 4), dtype=np.float32)
    second_candidate = np.full((2, 4), 3., dtype=np.float32)
    one = auditor.error_stats([(first_reference, first_candidate)], first_reference.shape)
    two = auditor.error_stats([(second_reference[:1], second_candidate[:1]),
                               (second_reference[1:], second_candidate[1:])], second_reference.shape)
    for item, expected in ((one, (2, 1., 1., 1., 1., .5, math.sqrt(.5), 1.)),
                           (two, (8, 2., 16., 32., 8., 2., 2., 2.))):
        summary.validate_error_stats(item)
        for key, value in zip(("elements", "max_abs", "sum_abs", "sum_sq", "sum_reference_sq",
                               "mean_abs", "rms", "relative_l2"), expected):
            close(item[key], value)
    records = [{"stats": {"logits": item}} for item in (one, two)]
    aggregate = summary.numerical_summary(records, "logits")
    independent = auditor.numerical_aggregate(records, "logits")
    auditor.compare_tree(aggregate, independent, "CPU numerical API probe")
    assert aggregate["elements"] == 10 and aggregate["max_abs"] == 2
    close(aggregate["macro_mean_abs"], 1.25)
    close(aggregate["pooled_mean_abs"], 1.7)
    close(aggregate["macro_rms"], (math.sqrt(.5) + 2) / 2)
    close(aggregate["pooled_rms"], math.sqrt(3.3))
    close(aggregate["macro_relative_l2"], 1.5)
    close(aggregate["pooled_relative_l2"], math.sqrt(33 / 9))
    close(aggregate["p95_query_max_abs"], 1.95)

    # Zero reference conventions, finite checks and mismatched array coverage.
    zeros = np.zeros((1, 2), dtype=np.float32)
    zero = auditor.error_stats([(zeros, zeros)], zeros.shape)
    undefined = auditor.error_stats([(zeros, np.ones_like(zeros))], zeros.shape)
    summary.validate_error_stats(zero)
    summary.validate_error_stats(undefined)
    assert zero["relative_l2"] == 0 and zero["relative_l2_defined"] is True
    assert undefined["relative_l2"] is None and undefined["relative_l2_defined"] is False
    for function in (summary.relative_l2, auditor.relative_l2):
        assert function(0, 0) == (0., True) and function(1, 0) == (None, False)
    undefined_aggregate = summary.numerical_summary([{"stats": {"logits": undefined}}], "logits")
    assert undefined_aggregate["relative_l2_undefined_queries"] == 1
    assert undefined_aggregate["macro_relative_l2"] is None and undefined_aggregate["pooled_relative_l2"] is None
    failure(auditor.error_stats, [(zeros, np.zeros((1, 3), dtype=np.float32))], zeros.shape)
    failure(auditor.error_stats, [(zeros, np.full_like(zeros, np.nan))], zeros.shape)
    failure(auditor.error_stats, [(zeros, zeros)], (1, 3))
    failure(summary.validate_error_stats, dict(one, rms=float("nan")))

    # Length gaps, EOS-bearing sequences, empty text and fixed-history versus greedy coverage.
    pairs = [([1, 9], [1, 9], "  ", ""), ([1, 2, 9], [1, 3, 4, 9], "A B", "A X B")]
    agreement_records = []
    for index, (left, right, text_left, text_right) in enumerate(pairs):
        modes = [{"token_ids": ids, "steps": len(ids), "text_raw": text, "text_stripped": text.strip(),
                  "eos_reached": True, "at_cap": False, "cap_without_eos": False}
                 for ids, text in ((left, text_left), (right, text_right))]
        record = {"reference": modes[0], "cuda": modes[1],
                  "greedy": {"token_sequence_equal": index == 0, "text_raw_equal": text_left == text_right,
                             "text_stripped_equal": text_left.strip() == text_right.strip(),
                             "common_prefix_tokens": 2 if index == 0 else 1, "comparable_prediction_steps": 2},
                  "reference_history": {"coverage_steps": len(left), "argmax_token_mismatch_count": index,
                                        "tail_rows": index, "replay_next_token_calls": index}}
        record["edits"] = summary.query_edits(record)
        agreement_records.append(record)
    agreements = summary.agreement_summary(agreement_records)
    auditor.compare_tree(agreements, auditor.agreement_aggregate(agreement_records), "CPU agreement API probe")
    assert agreements["positional_token_matches"] == 3 and agreements["max_length_token_positions"] == 6
    close(agreements["positional_token_identity"], .5)
    close(agreements["reference_history_argmax_identity"], .8)
    close(agreements["greedy_reference_history_coverage"], .8)
    close(agreements["reference_history_coverage_fraction"], 1.)
    assert agreements["reference_empty_text_queries"] == agreements["cuda_empty_text_queries"] == 1
    for unit in ("tokens", "characters", "words"):
        value = summary.edit_summary([record["edits"] for record in agreement_records], unit)
        auditor.compare_tree(value, auditor.edit_aggregate(agreement_records, unit), "CPU edit aggregation/" + unit)
    character_edits = summary.edit_summary([record["edits"] for record in agreement_records], "characters")
    close(character_edits["macro_normalized_distance"], .2)
    close(character_edits["pooled_normalized_distance"], .4)
    assert character_edits["both_empty_queries"] == 1

    print(json.dumps({"status": "pass", "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "numpy_version": np.__version__, "rapidfuzz_version": rapidfuzz.__version__,
        "summary_source": str(args.summary), "auditor_source": str(args.auditor),
        "checks": ["official C++/Python sequence APIs", "empty/Unicode/whitespace/EOS conventions",
                   "NumPy segmented numerical errors", "unequal-query macro versus pooled relative L2",
                   "undefined/invalid numerical inputs", "positional versus full-reference-history agreement",
                   "macro versus pooled edit distance with retained empty text"]}), flush=True)


if __name__ == "__main__":
    main()
