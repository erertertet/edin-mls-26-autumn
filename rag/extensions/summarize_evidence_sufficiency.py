"""Validate and plot the cached evidence-sufficiency experiment (CPU only)."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def envelope(x, y, target):
    hull = []
    for i in np.argsort(x):
        if hull and x[i] == x[hull[-1]]:
            if y[i] <= y[hull[-1]]:
                continue
            hull.pop()
        while len(hull) > 1:
            a, b = hull[-2:]
            cross = (x[b] - x[a]) * (y[i] - y[b]) - (y[b] - y[a]) * (x[i] - x[b])
            if cross < 0:
                break
            hull.pop()
        hull.append(i)
    if target < x[hull[0]] or target > x[hull[-1]]:
        return None
    return float(np.interp(target, x[hull], y[hull]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    args = p.parse_args()
    run = args.run
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    config = json.loads((run / "config.json").read_text())
    resultmap = {(r["phase"], r["query_id"]): r for r in rows(run / "query-results.jsonl")}
    tests = [resultmap[("test", qid)] for qid in config["test_ids"]]
    devs = [resultmap[("dev", qid)] for qid in config["dev_ids"]]
    refs = {r["query_id"]: r for r in rows(run / "references.jsonl")}
    calls = {r["key"]: r for r in rows(run / "calls.jsonl")}
    requests = rows(run / "requests.jsonl")
    frozen = json.loads((run / "frozen-parameters.json").read_text())
    original_queries = {r["query_id"]: r for r in rows(Path(config["base"]) / "queries.jsonl")}
    arms = ["fixed1", "fixed2", "fixed3", "fixed5", "fixed10", "fixed20", "mmr3", "mmr5", "absolute_cutoff", "evidence_stop", "onepass_select"]
    data = {arm: [next(a for a in q["arms"] if a["arm"] == arm) for q in tests] for arm in arms}
    eligible = np.array([q["reference_status"] == "assessable" for q in tests])
    metrics = {}
    rng = np.random.default_rng(20261118)
    samples = rng.integers(0, len(tests), size=(args.bootstrap, len(tests)))
    for arm, rs in data.items():
        fields = ["k", "context_tokens", "qrels_recall", "selector_usd", "selector_ms", "generator_usd", "generator_ms", "inference_usd", "branch_service_ms", "reconstructed_warm_ms"]
        m = {"arm": arm, "queries": len(rs), "assessable_queries": int(eligible.sum())}
        m.update({f"mean_{f}": statistics.mean(r[f] for r in rs) for f in fields})
        suff = np.array([r["direct_sufficient"] for r in rs], dtype=float)
        m["sufficiency_rate"] = float(suff.mean())
        m["sufficiency_bootstrap_95pct"] = np.percentile(suff[samples].mean(axis=1), [2.5, 97.5]).tolist()
        m["assessable_sufficiency_rate"] = float(suff[eligible].mean())
        m["reference_complete_rate"] = statistics.mean(float(r["reference_complete"]) for r in rs if r["reference_complete"] is not None)
        m["fact_coverage"] = statistics.mean(r["fact_coverage"] for r in rs if r["fact_coverage"] is not None)
        m["p95_branch_service_ms"] = float(np.percentile([r["branch_service_ms"] for r in rs], 95))
        metrics[arm] = m
    paired = []
    for arm in ["mmr3", "mmr5", "absolute_cutoff", "evidence_stop", "onepass_select"]:
        for baseline in ["fixed3", "fixed5", "fixed20"]:
            a, b = data[arm], data[baseline]
            row = {"arm": arm, "baseline": baseline}
            for field in ["direct_sufficient", "context_tokens", "inference_usd", "branch_service_ms"]:
                diff = np.array([float(x[field]) - float(y[field]) for x, y in zip(a, b)])
                row[field + "_delta"] = float(diff.mean())
                row[field + "_bootstrap_95pct"] = np.percentile(diff[samples].mean(axis=1), [2.5, 97.5]).tolist()
            paired.append(row)
    matched = []
    fixed_names = arms[:6]
    fy = np.array([[r["direct_sufficient"] for r in data[a]] for a in fixed_names], dtype=float)
    for cost in ["context_tokens", "inference_usd"]:
        fx = np.array([[r[cost] for r in data[a]] for a in fixed_names])
        for arm in arms[6:]:
            ax = np.array([r[cost] for r in data[arm]])
            ay = np.array([r["direct_sufficient"] for r in data[arm]], dtype=float)
            baseline = envelope(fx.mean(axis=1), fy.mean(axis=1), ax.mean())
            ci = []
            if baseline is not None:
                for s in samples:
                    base = envelope(fx[:, s].mean(axis=1), fy[:, s].mean(axis=1), ax[s].mean())
                    if base is not None:
                        ci.append(float(ay[s].mean() - base))
            matched.append({"arm": arm, "cost": cost, "mean_cost": float(ax.mean()),
                            "fixed_mixture_sufficiency": baseline,
                            "gain": float(ay.mean() - baseline) if baseline is not None else None,
                            "bootstrap_95pct": np.percentile(ci, [2.5, 97.5]).tolist() if ci else None,
                            "valid_bootstrap_samples": len(ci)})
    audit = {(r["query_id"], r["arm"]): r for r in rows(run / "audit.jsonl")}
    av = list(audit.values())
    agreement = {}
    if av:
        actual = np.array([r["primary_direct"] for r in av], dtype=bool)
        second = np.array([r["direct_sufficient"] for r in av], dtype=bool)
        observed = float((actual == second).mean())
        chance = float(actual.mean() * second.mean() + (1 - actual.mean()) * (1 - second.mean()))
        agreement = {"contexts": len(av), "queries": len({r["query_id"] for r in av}), "agreement": observed,
                     "cohen_kappa": (observed - chance) / (1 - chance) if chance < 1 else None,
                     "primary_positive_rate": float(actual.mean()), "audit_positive_rate": float(second.mean()),
                     "by_arm": {arm: {"n": sum(r["arm"] == arm for r in av),
                         "primary_rate": statistics.mean(float(r["primary_direct"]) for r in av if r["arm"] == arm),
                         "audit_rate": statistics.mean(float(r["direct_sufficient"]) for r in av if r["arm"] == arm)} for arm in ["fixed3", "evidence_stop", "onepass_select"]}}
    top = np.array([r["direct_sufficient"] for r in data["fixed20"]], dtype=bool)
    stop = np.array([r["direct_sufficient"] for r in data["evidence_stop"]], dtype=bool)
    early = np.array([r["stopped_early"] for r in data["evidence_stop"]], dtype=bool)
    diagnostics = {
        "top20_sufficient_queries": int(top.sum()), "top20_insufficient_queries": int((~top).sum()),
        "stop_selected_k": dict(Counter(r["k"] for r in data["evidence_stop"])),
        "early_stops": int(early.sum()), "false_early_stops": int((early & ~stop).sum()),
        "false_stop_rate_among_stops": float((early & ~stop).sum() / early.sum()) if early.sum() else None,
        "recoverable_premature_stops": int((early & ~stop & top).sum()),
        "recoverable_premature_rate_among_top20_sufficient": float((early & ~stop & top).sum() / top.sum()) if top.sum() else None,
        "fixed3_sufficient_top20_insufficient": int(sum(a["direct_sufficient"] and not b["direct_sufficient"] for a, b in zip(data["fixed3"], data["fixed20"]))),
        "strict_vs_direct_disagreement": {a: sum(r["reference_complete"] != r["direct_sufficient"] for r in rs if r["reference_complete"] is not None) for a, rs in data.items()},
        "already_sufficient_at_k_among_top20_sufficient": {str(k): {
            "count": int(sum(a["direct_sufficient"] and b["direct_sufficient"] for a, b in zip(data[f"fixed{k}"], data["fixed20"]))),
            "denominator": int(top.sum())} for k in [1, 2, 3, 5, 10]},
        "first_sufficient_tested_prefix": dict(Counter(next((str(k) for k in [1, 2, 3, 5, 10, 20] if data[f"fixed{k}"][i]["direct_sufficient"]), "none") for i in range(len(tests)))),
        "sufficient_but_incomplete_qrels_recall": {a: sum(r["direct_sufficient"] and r["qrels_recall"] < 1 for r in rs) for a, rs in data.items()},
        "raw_positive_rejected_by_quote_or_missing_check": {a: sum(calls[r["direct_call_key"]]["value"].get("sufficient") is True and not r["direct_sufficient"] for r in rs) for a, rs in data.items()},
    }
    statuses = Counter(refs[q["query_id"]]["status"] for q in tests)
    validation = {
        "dev_queries": len(devs), "test_queries": len(tests), "test_policy_results": sum(len(q["arms"]) for q in tests),
        "unique_api_ids": len({r["id"] for r in requests}), "api_requests": len(requests),
        "invalid_selected_ids": sum(len(r["record_ids"]) != len(set(r["record_ids"])) or not 1 <= len(r["record_ids"]) <= 20 for rs in data.values() for r in rs),
        "outside_candidate_pool": sum(not set(a["record_ids"]).issubset(original_queries[q["query_id"]]["record_ids"]) for q in tests for a in q["arms"]),
        "invalid_fact_coverage": sum(not 0 <= r["fact_coverage"] <= 1 for rs in data.values() for r in rs if r["fact_coverage"] is not None),
        "test_results_at_freeze": frozen["test_results_at_freeze"],
        "total_usd": sum(float(r["usage"]["cost"]) for r in requests),
        "cost_by_role": {role: sum(float(r["usage"]["cost"]) for r in requests if r["role"] == role) for role in sorted({r["role"] for r in requests})},
        "length_limited_calls": dict(Counter(r["role"] for r in requests if r["finish_reason"] == "length")),
        "audit_contexts": len(audit), "reference_statuses_test": dict(statuses),
    }
    assert validation["test_queries"] == 324 and validation["dev_queries"] == 125
    assert validation["test_policy_results"] == 324 * len(arms)
    assert validation["invalid_selected_ids"] == validation["invalid_fact_coverage"] == validation["test_results_at_freeze"] == 0
    assert validation["outside_candidate_pool"] == 0
    assert validation["total_usd"] <= config["budget_usd"]
    assert validation["audit_contexts"] == 90
    summary = {"metrics": list(metrics.values()), "paired": paired, "matched_budget": matched,
               "audit": agreement, "diagnostics": diagnostics, "validation": validation, "frozen_parameters": frozen}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(metrics.values()))))
        w.writeheader(); w.writerows(metrics.values())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    colors = {"mmr3": "#188479", "mmr5": "#42a898", "absolute_cutoff": "#9b8553", "evidence_stop": "#d55e00", "onepass_select": "#8051a8"}
    labels = {"mmr3": "Diversity: 3 docs", "mmr5": "Diversity: 5 docs", "absolute_cutoff": "Distance cutoff", "evidence_stop": "Evidence stopping", "onepass_select": "One-pass selection"}
    def plot(ax, cost, scale, xlabel, annotations=False):
        fx = [metrics[a][cost] * scale for a in fixed_names]
        fy = [metrics[a]["sufficiency_rate"] * 100 for a in fixed_names]
        ax.plot(fx, fy, "o-", color="#24476c", lw=2, label="Fixed top-k", zorder=3)
        for a, x, y in zip(fixed_names, fx, fy):
            ax.annotate(a.replace("fixed", "k="), (x, y), xytext=(3, -17), textcoords="offset points", fontsize=9)
        for a in arms[6:]:
            m = metrics[a]
            y = m["sufficiency_rate"] * 100
            lo, hi = np.array(m["sufficiency_bootstrap_95pct"]) * 100
            ax.errorbar(m[cost] * scale, y, yerr=[[y-lo], [hi-y]], fmt="D" if a.startswith("mmr") else "*" if a == "onepass_select" else "s",
                        color=colors[a], ms=10 if a == "onepass_select" else 7, capsize=3, label=labels[a], zorder=4)
        ax.set_xlabel(xlabel); ax.set_ylabel("Estimated sufficient contexts (%)")
        ax.grid(alpha=0.2)
        ax.set_ylim(max(0, min(fy)-8), min(100, max([m["sufficiency_bootstrap_95pct"][1] * 100 for m in metrics.values()])+5))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.1))
    plot(axes[0], "mean_context_tokens", 1, "Mean generator input tokens")
    plot(axes[1], "mean_inference_usd", 1000, "Inference USD / 1,000 queries (selector + generator)")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Does selecting enough evidence improve the cost–sufficiency trade-off?", fontsize=16)
    fig.text(0.5, 0.02, "324 reused test queries · fixed top-20 candidate pool · Gemini context labels, not human ground truth\nError bars: paired-query bootstrap marginal 95% intervals; evaluation costs excluded from inference axis", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    for ext in ["png", "svg"]: fig.savefig(out / f"evidence-sufficiency-frontier.{ext}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    plot(ax, "mean_branch_service_ms", .001, "Mean API branch service time (seconds)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Sufficiency versus selector + generation latency")
    fig.text(.5, .02, "16 concurrent query workers; sums sequential API calls, excludes evaluation and cached ASR\nA service-time comparison, not a fresh audio-to-answer wall-clock benchmark", ha="center", fontsize=10)
    fig.tight_layout(rect=(0,.08,1,1))
    for ext in ["png", "svg"]: fig.savefig(out / f"evidence-sufficiency-latency.{ext}", dpi=180)
    plt.close(fig)
    print(json.dumps({"validation": validation, "metrics": list(metrics.values()), "audit": agreement, "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
