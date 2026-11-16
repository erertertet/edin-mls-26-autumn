"""Retrieval-only token/recall sweeps over saved, fixed top-20 rankings.

No ASR, embedding, generation, planner or judge calls are made. Only a public
tokenizer may be downloaded. CPU plotting uses Matplotlib's headless backend.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CourseData
from src.context_builder import SimpleContextBuilder

SYSTEM = "Use the supplied evidence to answer the user's question. Do not invent facts that are unsupported by the evidence."
COLORS = {"fixed_k": "#24476C", "absolute_radius": "#D65F2B", "relative_gap": "#16877D"}
LABELS = {"fixed_k": "Fixed top-k", "absolute_radius": "Absolute distance cutoff", "relative_gap": "Relative score-gap cutoff"}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def upper_hull(x, y):
    """Upper concave envelope: query-independent random mixtures of fixed k."""
    hull = []
    for index in np.argsort(x):
        index = int(index)
        if hull and x[index] == x[hull[-1]]:
            if y[index] <= y[hull[-1]]:
                continue
            hull.pop()
        while len(hull) >= 2:
            a, b = hull[-2:]
            cross = (x[b] - x[a]) * (y[index] - y[b]) - (y[b] - y[a]) * (x[index] - x[b])
            if cross < 0:
                break
            hull.pop()
        hull.append(index)
    return np.array(hull, dtype=int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    partition = json.loads((args.run / "partition.json").read_text())
    config = json.loads((args.run / "config.json").read_text())
    parameters = json.loads((args.run / "frozen-parameters.json").read_text())
    metadata = json.loads((args.run / "tokenizer-metadata.json").read_text())
    tokenizer_path = hf_hub_download(metadata["repo"], "tokenizer.json", revision=metadata["revision"])
    tokenizer = Tokenizer.from_file(tokenizer_path)
    system_tokens = len(tokenizer.encode(SYSTEM, add_special_tokens=False).ids)
    builder = SimpleContextBuilder()
    data = CourseData(root / "data/rag_corpus")
    qrels = {split: data.load_qrels(split) for split in ["practice", "dev"]}
    queries = {row["query_id"]: row for row in read_rows(args.run / "queries.jsonl")}
    ids_all = sorted(queries)
    index = {qid: n for n, qid in enumerate(ids_all)}
    counts = np.zeros((len(ids_all), 20), dtype=np.int32)
    recall = np.zeros((len(ids_all), 20), dtype=np.float64)
    precision = np.zeros_like(recall)
    similarities = np.zeros_like(recall)
    for n, qid in enumerate(ids_all):
        query = queries[qid]
        relevant = set(qrels[query["source_split"]][qid])
        hits = 0
        for k in range(1, 21):
            chosen = query["record_ids"][:k]
            prompt = builder.build(query["transcript"], chosen, data)["prompt"]
            counts[n, k - 1] = system_tokens + len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            hits += query["record_ids"][k - 1] in relevant
            recall[n, k - 1] = hits / len(relevant)
            precision[n, k - 1] = hits / k
        similarities[n] = query["similarities"]
        if (n + 1) % 100 == 0:
            print(f"Counted token costs for {n+1}/{len(ids_all)} cached queries", flush=True)

    # Fit chat overhead on dev token accounting only; validate against held-out logs.
    api = {(row["query_id"], row["arm"]): row["usage"]["prompt_tokens"] for row in read_rows(args.run / "requests.jsonl") if row["role"] == "generator"}
    results = read_rows(args.run / "results.jsonl")
    dev_offsets = [api[(row["query_id"], row["arm"])] - counts[index[row["query_id"]], row["k"]-1] for row in results if row["phase"] == "dev_tuning"]
    overhead = int(statistics.median(dev_offsets))
    counts += overhead
    errors, ranking_mismatches = [], 0
    for row in results:
        errors.append(int(counts[index[row["query_id"]], row["k"]-1]) - api[(row["query_id"], row["arm"])])
        ranking_mismatches += row["record_ids"] != queries[row["query_id"]]["record_ids"][:row["k"]]
    validation = {
        "tokenizer_repo": metadata["repo"], "tokenizer_revision": metadata["revision"],
        "chat_overhead_calibrated_on_dev": overhead,
        "compared_api_requests": len(errors), "exact_token_count_matches": sum(value == 0 for value in errors),
        "max_absolute_token_error": max(abs(value) for value in errors),
        "mean_absolute_token_error": float(np.mean(np.abs(errors))),
        "prefix_ranking_mismatches": ranking_mismatches,
    }
    if validation["max_absolute_token_error"] > 2 or ranking_mismatches:
        raise RuntimeError("Token/ranking validation failed: " + json.dumps(validation))
    assert np.all(np.diff(counts, axis=1) >= 0)
    assert np.all(np.diff(recall, axis=1) >= 0)
    test_ids = partition["heldout_test_query_ids"]
    ii = np.array([index[qid] for qid in test_ids])
    tokens, recalls, precisions, scores = counts[ii], recall[ii], precision[ii], similarities[ii]
    n = len(ii)
    query_index = np.arange(n)
    settings = []
    arrays = {}

    def add(family, value, selected):
        selected = np.maximum(1, np.minimum(20, selected)).astype(int)
        columns = selected - 1
        cost = tokens[query_index, columns]
        effect = recalls[query_index, columns]
        relevant_precision = precisions[query_index, columns]
        key = f"{family}:{value:.8g}"
        row = {
            "family": family, "parameter": float(value), "mean_input_tokens": float(cost.mean()),
            "macro_recall": float(effect.mean()), "macro_precision": float(relevant_precision.mean()),
            "mean_documents": float(selected.mean()), "queries": n,
        }
        settings.append(row)
        arrays[key] = {"tokens": cost, "recall": effect, "k": selected}
        return row, key

    fixed = [add("fixed_k", k, np.full(n, k))[0] for k in range(1, 21)]
    # Add deterministic endpoints so curves share the same top-1/top-20 domain.
    for tau in sorted(set(config["absolute_similarity_grid"] + [-1.0, 1.01])):
        add("absolute_radius", tau, (scores >= tau).sum(axis=1))
    for gap in sorted(set(config["relative_gap_grid"] + [0.1, 0.2, 0.5, 2.0])):
        add("relative_gap", gap, (scores >= scores[:, :1] - gap).sum(axis=1))
    planners = {row["query_id"]: row for row in read_rows(args.run / "planners.jsonl")}
    predicted, predicted_key = add("llm_depth", 1, np.array([planners[qid]["k"] for qid in test_ids]))
    fx, fy = tokens.mean(axis=0), recalls.mean(axis=0)
    hull = upper_hull(fx, fy)
    for row in settings:
        row["fixed_k_mixture_recall_at_same_cost"] = float(np.interp(row["mean_input_tokens"], fx[hull], fy[hull]))
        row["recall_gain_vs_fixed_mixture"] = row["macro_recall"] - row["fixed_k_mixture_recall_at_same_cost"]
    frozen = []
    for family, value in [("absolute_radius", parameters["tau"]), ("relative_gap", parameters["delta"]), ("llm_depth", 1)]:
        row = min((row for row in settings if row["family"] == family), key=lambda row: abs(row["parameter"] - value))
        frozen.append(dict(row))
    boot = {row["family"]: [] for row in frozen}
    rng = np.random.default_rng(20261121)
    for _ in range(args.bootstrap):
        sample = rng.integers(0, n, size=n)
        bx, by = tokens[sample].mean(axis=0), recalls[sample].mean(axis=0)
        bh = upper_hull(bx, by)
        for row in frozen:
            values = arrays[f"{row['family']}:{row['parameter']:.8g}"]
            cost, effect = values["tokens"][sample].mean(), values["recall"][sample].mean()
            boot[row["family"]].append(float(effect - np.interp(cost, bx[bh], by[bh])))
    for row in frozen:
        row["matched_budget_gain_bootstrap_95pct"] = np.percentile(boot[row["family"]], [2.5, 97.5]).tolist()
        position = int(np.searchsorted(fx[hull], row["mean_input_tokens"]))
        low, high = hull[max(0, position - 1)], hull[min(len(hull)-1, position)]
        row["matched_fixed_k_low"] = int(low + 1)
        row["matched_fixed_k_high"] = int(high + 1)
    summary = {
        "queries": n, "ranking_pool": 20, "fixed_k_values": list(range(1,21)),
        "absolute_settings": sum(row["family"] == "absolute_radius" for row in settings),
        "relative_settings": sum(row["family"] == "relative_gap" for row in settings),
        "fixed_k_mixture_hull_k": (hull+1).tolist(), "validation": validation,
        "frozen_policy_points": frozen, "additional_inference_api_calls": 0,
        "notes": [
            "All curves use the same cached rankings and same 324 test queries; no new generation or judging.",
            "X is full generator input tokens, including system/question/chat overhead. Planner tokens and outputs excluded: this is selection-only, not total deployment cost.",
            "Y is macro recall of labelled relevant documents; it is not answer quality.",
            "The fixed-k upper concave envelope permits query-independent random mixtures and is an empirical descriptive comparator, not a deployment policy tuned on test.",
            "Previously dev-selected points are highlighted. Swept test curves are descriptive; no new operating point is selected from test.",
            "Added endpoint thresholds anchor the shared top-1/top-20 domain; all policies return at least one document.",
            "Bootstrap intervals resample paired queries and recompute the fixed-k envelope; they are exploratory and unadjusted.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out / "sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(settings[0]))
        writer.writeheader(); writer.writerows(settings)
    np.savez_compressed(args.out / "query-prefix-metrics.npz", input_tokens=tokens, recall=recalls, precision=precisions, query_ids=np.array(test_ids))

    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "axes.spines.top":False, "axes.spines.right":False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(15.8, 7.5), gridspec_kw={"width_ratios":[1.18,1]})
    for family in ["fixed_k", "absolute_radius", "relative_gap"]:
        group = sorted([row for row in settings if row["family"] == family], key=lambda row: row["mean_input_tokens"])
        x = [row["mean_input_tokens"] for row in group]
        y = [100*row["macro_recall"] for row in group]
        gain = [100*row["recall_gain_vs_fixed_mixture"] for row in group]
        left.plot(x, y, color=COLORS[family], lw=2.2, label=LABELS[family], marker="o" if family=="fixed_k" else None, ms=3.8, alpha=0.95)
        if family != "fixed_k":
            right.plot(x, gain, color=COLORS[family], lw=2.2, label=LABELS[family])
    left.plot(fx[hull], fy[hull]*100, color="#57616D", ls="--", lw=1.4, label="Fixed-k random-mixture envelope", zorder=5)
    for k in [1,2,3,5,10,20]:
        offset = (-22, 12) if k == 2 else (6, -15 if k == 10 else 7)
        left.annotate(f"k={k}", (fx[k-1],fy[k-1]*100), xytext=offset, textcoords="offset points", color=COLORS["fixed_k"], fontsize=9)
    for row in frozen:
        family = row["family"]
        color = COLORS.get(family, "#8C5AA4")
        marker = "*" if family=="llm_depth" else "D"
        size = 145 if family=="llm_depth" else 65
        left.scatter(row["mean_input_tokens"],100*row["macro_recall"],s=size,marker=marker,color=color,edgecolors="white",linewidths=0.9,zorder=8,label="Cached LLM-predicted k" if family=="llm_depth" else None)
        right.errorbar(row["mean_input_tokens"],100*row["recall_gain_vs_fixed_mixture"]
            ,yerr=np.array([[row["recall_gain_vs_fixed_mixture"]-row["matched_budget_gain_bootstrap_95pct"][0]],[row["matched_budget_gain_bootstrap_95pct"][1]-row["recall_gain_vs_fixed_mixture"]]])*100
            ,fmt=marker,color=color,ms=10 if family=="llm_depth" else 7,capsize=4,lw=1.5,zorder=8)
    for ax in [left,right]:
        ax.set_xlabel("Mean generator input tokens per query")
        ax.grid(True,alpha=0.20)
        ax.set_xlim(0,5000)
    left.set_ylabel("Relevant-document recall (%)")
    left.set_ylim(0,65)
    left.set_title("A  Evidence coverage vs. input budget",loc="left",weight="bold",pad=14)
    left.legend(loc="lower right",frameon=True,framealpha=0.96,fontsize=9)
    right.axhline(0,color="#57616D",ls="--",lw=1.4)
    right.set_ylim(-7.8, 1.2)
    right.set_ylabel("Recall gain vs. fixed-k mixture (percentage points)")
    right.set_title("B  Is adaptive stopping better at the same cost?",loc="left",weight="bold",pad=14)
    right.text(0.98,0.97,"Above zero: better evidence coverage\nBelow zero: worse evidence coverage",transform=right.transAxes,ha="right",va="top",fontsize=10,color="#444444")
    fig.suptitle("Retrieval-only frontier: the ranking stays fixed; the stopping rule changes",fontsize=18,weight="bold",y=0.975)
    fig.text(0.06,0.902,"324 held-out queries  |  24,000-document corpus  |  shared top-20 ranking  |  fixed k = 1-20",fontsize=11,color="#555555")
    fig.text(0.06,0.045,"Diamonds: thresholds previously selected on dev. Star: cached model-predicted k; planner overhead excluded.\nError bars: paired 95% bootstrap intervals at those operating points. Lines join discrete settings; no answer-generation calls.",fontsize=9.5,color="#555555",linespacing=1.6)
    fig.subplots_adjust(left=0.06,right=0.98,bottom=0.19,top=0.83,wspace=0.30)
    for extension in ["png","svg"]:
        fig.savefig(args.out / f"retrieval-frontier.{extension}",dpi=180,facecolor="white")
    plt.close(fig)
    # A larger-type, single-panel version is easier to read inline in chat.
    plt.rcParams.update({"font.size":12})
    fig, ax = plt.subplots(figsize=(10.8, 7.4))
    for family in ["fixed_k", "absolute_radius", "relative_gap"]:
        group = sorted([row for row in settings if row["family"] == family], key=lambda row: row["mean_input_tokens"])
        ax.plot([row["mean_input_tokens"] for row in group], [100*row["macro_recall"] for row in group],
                color=COLORS[family], lw=2.5, label=LABELS[family], marker="o" if family=="fixed_k" else None, ms=4)
    ax.plot(fx[hull], fy[hull]*100, color="#57616D", ls="--", lw=1.3, label="Fixed-k mixture envelope")
    for k in [1,2,3,5,10,20]:
        ax.annotate(f"k={k}", (fx[k-1],fy[k-1]*100), xytext=(-22,12) if k==2 else (6,-16 if k==10 else 8), textcoords="offset points", color=COLORS["fixed_k"], fontsize=10)
    for row in frozen:
        family=row["family"]
        ax.scatter(row["mean_input_tokens"],100*row["macro_recall"],s=165 if family=="llm_depth" else 80,
                   marker="*" if family=="llm_depth" else "D",color=COLORS.get(family,"#8C5AA4"),
                   edgecolors="white",linewidths=1,zorder=8,label="Cached LLM-predicted k" if family=="llm_depth" else None)
    ax.set(xlim=(0,5000),ylim=(0,65),xlabel="Mean generator input tokens per query",ylabel="Relevant-document recall (%)")
    ax.grid(True,alpha=0.20)
    ax.legend(loc="lower right",fontsize=10,framealpha=0.96)
    fig.suptitle("Retrieval-only: evidence coverage vs. input budget",fontsize=17,weight="bold",y=0.975)
    fig.text(0.10,0.91,"324 held-out queries  |  fixed ranking  |  higher and farther left is better",fontsize=11,color="#555555")
    fig.text(0.10,0.035,"Diamonds: dev-selected cutoffs. Star: cached model-predicted k (planner cost excluded).\nInput-token counting matched all 2,335 logged API requests exactly. No new inference calls.",fontsize=9.5,color="#555555",linespacing=1.6)
    fig.subplots_adjust(left=0.10,right=0.97,bottom=0.19,top=0.86)
    for extension in ["png","svg"]:
        fig.savefig(args.out / f"retrieval-frontier-main.{extension}",dpi=180,facecolor="white")
    plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
