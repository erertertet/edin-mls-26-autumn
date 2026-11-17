"""Render per-query answer diagnostics from the completed saved nine-policy run.

No model calls or retrieval are performed. Only aggregate plot data are exported;
query IDs, transcripts and answers remain in the private source run.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


GROUPS = [
    ("Fixed top-k", ["fixed5", "fixed10", "fixed20"],
     ["#91abc5", "#527da5", "#24476c"]),
    ("Absolute similarity", ["absolute_radius", "absolute_088", "absolute_087"],
     ["#e7a080", "#d65f2b", "#97401e"]),
    ("Relative score gap", ["relative_gap", "relative_002", "relative_003"],
     ["#80c7bd", "#16877d", "#07584f"]),
]
LABELS = {
    "fixed5": r"$k=5$", "fixed10": r"$k=10$", "fixed20": r"$k=20$",
    "absolute_radius": r"$\tau=0.8925$", "absolute_088": r"$\tau=0.88$",
    "absolute_087": r"$\tau=0.87$", "relative_gap": r"$\delta=0.009$",
    "relative_002": r"$\delta=0.02$", "relative_003": r"$\delta=0.03$",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_saved(run):
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in
            (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    require(summary["complete"] and summary["evaluation_queries"] == 324,
            "Expected the completed 324-query evaluation")
    expected = set(LABELS)
    require({row["arm"] for row in rows} == expected, "Unexpected policy coverage")
    require(len(rows) == summary["judged"] == summary["answers"] == 2916,
            "Expected 2,916 final judgements")
    require(len({(row["query_id"], row["arm"]) for row in rows}) == len(rows),
            "Duplicate query/policy judgement")
    by_arm = {arm: sorted([row for row in rows if row["arm"] == arm],
                          key=lambda row: row["query_id"]) for arm in LABELS}
    ids = [row["query_id"] for row in by_arm["fixed5"]]
    summaries = {row["arm"]: row for row in summary["by_arm"]}
    for arm, values in by_arm.items():
        require(len(values) == 324 and [row["query_id"] for row in values] == ids,
                "Policies must share the same 324 queries")
        for row in values:
            parts = [row["scores"][key] for key in ("correctness", "grounding", "completeness")]
            require(all(type(value) is int and 0 <= value <= 4 for value in parts),
                    "Invalid rubric component")
            require(abs(row["score"] - sum(parts) / 12) < 1e-12, "Score/rubric mismatch")
            require(type(row["k"]) is int and row["k"] == len(row["record_ids"]) >= 1,
                    "Document count mismatch")
            if arm.startswith("fixed"):
                require(row["k"] == int(arm[5:]), "Wrong fixed depth")
        for field, metric in (("score", "mean_score"), ("k", "mean_k")):
            require(abs(float(np.mean([row[field] for row in values])) - summaries[arm][metric]) < 1e-10,
                    "Plot data disagree with saved summary: " + arm + "/" + metric)
    return by_arm


def axes_style(ax):
    ax.set_ylim(-0.055, 1.055)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.grid(axis="y", alpha=.2, linewidth=.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, out, stem):
    for extension in ("pdf", "png"):
        fig.savefig(out / (stem + "." + extension), dpi=240, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def distribution(by_arm, out):
    fig, ax = plt.subplots(figsize=(6.0, 2.65))
    rng = np.random.default_rng(20261117)
    index = 0
    for name, arms, colors in GROUPS:
        for arm, color in zip(arms, colors):
            scores = np.array([row["score"] for row in by_arm[arm]])
            ax.scatter(index + rng.uniform(-.28, .28, len(scores)), scores,
                       s=7, alpha=.22, color=color, edgecolors="none", rasterized=True)
            ax.scatter(index, scores.mean(), s=38, marker="D", color="white",
                       edgecolors="#222222", linewidth=.9, zorder=5)
            index += 1
    for x in (2.5, 5.5):
        ax.axvline(x, color="#dddddd", linewidth=.7)
    axes_style(ax)
    ax.set(xlim=(-.6, 8.6), ylabel="Final judge score (0-1)")
    ax.set_xticks(range(9), [LABELS[arm] for arm in LABELS], rotation=25, ha="right")
    ax.set_title("324 answers per setting; diamonds show means", loc="left", fontsize=8)
    handles = [Line2D([], [], marker="o", linestyle="", color=colors[1],
                      label=name, markersize=4) for name, _, colors in GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.subplots_adjust(left=.105, right=.99, top=.88, bottom=.29)
    save(fig, out, "answer-score-distribution")


def score_documents(by_arm, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 3.0), sharex=True, sharey=True)
    # Coincident observations are aggregated per policy, not jittered. Marker
    # area is directly proportional to count; all observed counts are retained.
    area_per_answer = 2.0
    for ax, (name, arms, colors) in zip(axes, GROUPS):
        handles = []
        for arm, color, marker in zip(arms, colors, ("o", "s", "^")):
            counts = Counter((row["k"], round(row["score"] * 12)) for row in by_arm[arm])
            points = sorted(counts)
            ax.scatter([k for k, _ in points], [total / 12 for _, total in points],
                       s=[area_per_answer * counts[point] for point in points],
                       c=color, marker=marker, alpha=.48, edgecolors=color, linewidth=.35)
            handles.append(Line2D([], [], linestyle="", marker=marker, color=color,
                                  label=LABELS[arm], markersize=4))
        axes_style(ax)
        ax.set_ylim(-.15, 1.15)
        ax.set_xscale("log")
        ax.set_xlim(.65, 400)
        ax.set_xticks([1, 5, 20, 100, 300], ["1", "5", "20", "100", "300"])
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_title(name, loc="left", fontsize=8)
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -.26),
                  ncol=1, frameon=False, fontsize=6.5, borderaxespad=0, handletextpad=.4,
                  labelspacing=.2)
    axes[0].set_ylabel("Final judge score (0-1)")
    fig.text(.52, .265, "Retrieved documents (log scale; identical axes)", ha="center", fontsize=7)
    size_handles = [Line2D([], [], linestyle="", marker="o", color="#777777", alpha=.5,
                          markersize=np.sqrt(area_per_answer * count), label=str(count))
                    for count in (1, 10, 100)]
    fig.legend(handles=size_handles, title="Answers per bubble", loc="lower center",
               ncol=3, frameon=False, fontsize=6.5, title_fontsize=6.5,
               bbox_to_anchor=(.53, -.015), handletextpad=.5, columnspacing=1.2)
    fig.subplots_adjust(left=.105, right=.99, top=.90, bottom=.37, wspace=.12)
    save(fig, out, "answer-score-documents")


def nine_distributions(by_arm, out):
    """The selected preview 01: exact-score histograms on shared axes."""
    fig, axes = plt.subplots(3, 3, figsize=(6.0, 4.6), sharex=True, sharey=True)
    names = ("Top-k", "Absolute", "Relative")
    parameters = ("k = 5", "k = 10", "k = 20", "tau = 0.8925", "tau = 0.88",
                  "tau = 0.87", "delta = 0.009", "delta = 0.02", "delta = 0.03")
    for ri, (_, arms, colors) in enumerate(GROUPS):
        color = ("#315c85", "#cf612f", "#17877e")[ri]
        for ci, arm in enumerate(arms):
            ax = axes[ri, ci]
            scores = np.array([row["score"] for row in by_arm[arm]])
            counts = np.bincount(np.rint(scores * 12).astype(int), minlength=13)
            percentages = counts / len(scores) * 100
            ax.bar(np.arange(13) / 12, percentages, width=.062, color=color, alpha=.88)
            for index in (0, 6, 12):
                ax.text(index / 12, percentages[index] + 1.1, f"{percentages[index]:.1f}%",
                        ha="center", fontsize=6.5, color=color)
            ax.axvline(scores.mean(), color="#333333", lw=1, ls="--")
            ax.set_title(f"{names[ri]} | {parameters[ri * 3 + ci]}", loc="left", fontsize=7.5, pad=6)
            ax.text(.50, .85, f"mean {scores.mean():.3f}", transform=ax.transAxes,
                    ha="center", fontsize=6.5, color="#555555",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1.5))
            ax.set(xlim=(-.09, 1.09), ylim=(0, 65), xticks=[0, .5, 1], yticks=[0, 20, 40, 60])
            ax.tick_params(axis="both", labelsize=6.5)
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#dfe4e8", linewidth=.6)
            ax.set_axisbelow(True)
    fig.suptitle("Score distribution for each of the nine configurations", fontsize=10, x=.09, ha="left", y=.985)
    fig.text(.09, .943, "324 answers per panel | score steps of 1/12 | dashed line = mean",
             fontsize=7.5, color="#555555")
    fig.supxlabel("Final judge score", y=.025, fontsize=8.5)
    fig.supylabel("Answers within configuration (%)", x=.005, fontsize=8.5)
    fig.subplots_adjust(left=.09, right=.98, bottom=.105, top=.88, hspace=.40, wspace=.16)
    save(fig, out, "answer-score-distributions-nine")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nine-only", action="store_true", help="Render only the selected nine-panel histogram")
    args = parser.parse_args()
    by_arm = load_saved(args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42})
    nine_distributions(by_arm, args.out)
    if args.nine_only:
        print("Rendered nine score distributions; 2,916 validated saved judgements.")
        return
    distribution(by_arm, args.out)
    score_documents(by_arm, args.out)
    aggregate = []
    for arm, rows in by_arm.items():
        counts = Counter((row["k"], sum(row["scores"][key] for key in
                         ("correctness", "grounding", "completeness"))) for row in rows)
        aggregate.append({"arm": arm, "queries": len(rows),
                          "mean_score": float(np.mean([row["score"] for row in rows])),
                          "zero_scores": sum(row["score"] == 0 for row in rows),
                          "full_scores": sum(row["score"] == 1 for row in rows),
                          "mean_documents": float(np.mean([row["k"] for row in rows])),
                          "max_documents": max(row["k"] for row in rows),
                          "points": [{"documents": k, "score_sum": total, "score": total / 12,
                                      "answers": count} for (k, total), count in sorted(counts.items())]})
    output = {"source_run": args.run.name,
              "input_sha256": {name: hashlib.sha256((args.run / name).read_bytes()).hexdigest()
                               for name in ("results.jsonl", "summary.json")},
              "rendering": {"matplotlib": matplotlib.__version__, "numpy": np.__version__,
                            "distribution_horizontal_jitter_seed": 20261117,
                            "score_documents_area_points_squared_per_answer": 2.0},
              "notes": ["Saved final judgements only; 324 paired queries and 9 policies.",
                        "Score is the sum of three integer 0-4 rubric dimensions divided by 12.",
                        "Distribution jitter affects the categorical x coordinate only.",
                        "Document plot uses exact score/count coordinates, with area proportional to frequency.",
                        "Repeated queries across policies are not independent samples; plots are descriptive.",
                        "No ASR, encoder, retrieval, generation or judge calls."],
              "policies": aggregate}
    (args.out / "answer-distribution-data.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "rendered", "queries_per_policy": 324, "answers": 2916,
                      "outputs": ["answer-score-distribution", "answer-score-documents"]}))


if __name__ == "__main__":
    main()
