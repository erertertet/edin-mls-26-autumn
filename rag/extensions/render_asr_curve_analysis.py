"""Render saved ASR curve statistics as a report and standalone scientific figures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ARMS = ("glm", "whisper", "mai", "voxtral")
CONDITIONS = ("raw", "normalized")
FAMILIES = ("topk", "cutoff")
NAMES = {"glm": "GLM-ASR-Nano", "whisper": "Whisper large-v3",
         "mai": "MAI-Transcribe 2", "voxtral": "Voxtral Small Transcribe 3"}
COLORS = {"glm": "#0072B2", "whisper": "#E69F00", "mai": "#009E73", "voxtral": "#CC79A7"}
STYLE = {"raw": "--", "normalized": "-"}
CURVE_NUMBERS = ("parameter", "macro_recall", "macro_precision", "micro_precision",
                 "mean_tokens", "mean_documents", "mean_hits")
METRICS = ("achieved_recall", "macro_precision", "micro_precision", "mean_tokens", "mean_documents")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric(value, digits=3, scale=1, signed=False):
    if not finite(value):
        return "unsupported"
    return format(scale * value, ("+" if signed else "") + "." + str(digits) + "f")


def ci_bounds(item, metric):
    interval = item.get("intervals", {}).get(metric, {})
    low, high = interval.get("low"), interval.get("high")
    if finite(low) and finite(high) and low <= high:
        return low, high
    return None


def estimate(item, metric, scale=1, digits=3, signed=False):
    value = item.get("metrics", {}).get(metric)
    result = numeric(value, digits, scale, signed)
    bounds = ci_bounds(item, metric)
    if bounds is None:
        return result + (" [CI unavailable]" if finite(value) else "")
    return result + " [" + numeric(bounds[0], digits, scale, signed) + ", " + numeric(bounds[1], digits, scale, signed) + "]"


def policy(family):
    return "top-k" if family == "topk" else "absolute cutoff"


def read_curves(path):
    rows, seen = [], set()
    with path.open(encoding="utf-8", newline="") as stream:
        for source in csv.DictReader(stream):
            row = dict(source)
            for field in CURVE_NUMBERS:
                row[field] = float(row[field])
                require(finite(row[field]), "Nonfinite curve value: " + field)
            require(row["arm"] in ARMS and row["condition"] in CONDITIONS
                    and row["family"] in FAMILIES, "Unknown curve arm/condition/family")
            require(all(0 <= row[field] <= 1 for field in ("macro_recall", "macro_precision", "micro_precision"))
                    and row["mean_tokens"] >= 0 and row["mean_documents"] > 0,
                    "Invalid curve metric range")
            key = tuple(row[field] for field in ("cohort", "arm", "condition", "family", "parameter"))
            require(key not in seen, "Duplicate curve parameter row")
            seen.add(key)
            rows.append(row)
    require(rows, "No saved curve rows")
    return rows


def set_plot_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 10, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.facecolor": "white"})


def save_figure(fig, destination):
    paths = []
    for suffix in (".png", ".pdf"):
        path = destination.with_suffix(suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def model_legend(fig, y):
    handles = [Line2D([0], [0], color=COLORS[arm], lw=2, label=NAMES[arm]) for arm in ARMS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, y), ncol=4, fontsize=9)


def recall_token_plot(rows, output, full_range=False):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.3), sharex=True, sharey=True)
    fig.subplots_adjust(top=.83, bottom=.14, left=.09, right=.98, hspace=.32, wspace=.17)
    title = "Recall–document-token curves: full candidate range" if full_range else "Recall–document-token curves: operating range"
    fig.suptitle(title, y=.995, fontsize=13)
    model_legend(fig, .962)
    styles = [Line2D([0], [0], color=".25", ls=STYLE[c], lw=1.7, label=c.title()) for c in CONDITIONS]
    styles += [Line2D([0], [0], marker="o", color=".25", ls="none", label="fixed top-5"),
               Line2D([0], [0], marker="D", color=".25", ls="none", label="fixed cutoff 0.8925")]
    fig.legend(handles=styles, loc="upper center", bbox_to_anchor=(.5, .916), ncol=4, fontsize=9)
    omitted_zero = 0
    for row_index, (cohort, label) in enumerate((("all", "All provided queries (n=475)"),
                                               ("remaining_dev", "Remaining dev (n=117)"))):
        for column, family in enumerate(FAMILIES):
            ax = axes[row_index, column]
            ax.set_title(label + " — " + policy(family))
            for arm in ARMS:
                for condition in CONDITIONS:
                    current = sorted([r for r in rows if (r["cohort"], r["arm"], r["condition"], r["family"])
                                      == (cohort, arm, condition, family)],
                                     key=lambda r: (r["macro_recall"], r["mean_tokens"], r["parameter"]))
                    require(current, "Missing requested plot curve: " + str((cohort, arm, condition, family)))
                    plotted = current
                    if full_range:
                        plotted = [r for r in current if r["mean_tokens"] > 0]
                        omitted_zero += len(current) - len(plotted)
                    ax.plot([100 * r["macro_recall"] for r in plotted], [r["mean_tokens"] for r in plotted],
                            color=COLORS[arm], ls=STYLE[condition], lw=1.65 if condition == "normalized" else 1.2,
                            alpha=.95 if condition == "normalized" else .8)
                    fixed_parameter = 5 if family == "topk" else .8925
                    controls = [r for r in current if abs(r["parameter"] - fixed_parameter) <= 1e-12]
                    require(len(controls) == 1, "Missing/duplicate fixed-control curve point")
                    point = controls[0]
                    if not full_range or point["mean_tokens"] > 0:
                        ax.plot(100 * point["macro_recall"], point["mean_tokens"],
                                marker="o" if family == "topk" else "D", ms=4.5,
                                markeredgecolor=COLORS[arm], markeredgewidth=1,
                                markerfacecolor=COLORS[arm] if condition == "normalized" else "white",
                                ls="none", zorder=5)
            ax.set_xlim((0, 100) if full_range else (20, 70))
            if full_range:
                ax.set_yscale("log")
            else:
                ax.set_ylim(0, 10000)
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(True, which="major", color=".87", lw=.6)
            if row_index == 1:
                ax.set_xlabel("Macro labelled-evidence recall (%)")
            if column == 0:
                ax.set_ylabel("Mean document-text tokens" + (" (log scale)" if full_range else ""))
    footer = ("Observed descriptive curves; no simultaneous confidence bands. Raw: dashed / hollow marker; normalized: solid / filled marker.\n"
              "All provided data have been inspected previously. Operating-range limits crop the display, not the saved candidates.")
    if full_range:
        footer = ("Observed descriptive curves; no simultaneous confidence bands. Raw: dashed / hollow marker; normalized: solid / filled marker.\n"
                  "Full candidate range; logarithmic token axis. Zero-cost points omitted from log display: " + str(omitted_zero) + ".")
    fig.text(.5, .035, footer, ha="center", va="bottom", fontsize=8)
    return save_figure(fig, output), {"zero_cost_points_omitted": omitted_zero}


def normalization_plot(contrasts, output):
    points = [p for p in contrasts if p["panel"] == "matched" and p["kind"] == "normalization"]
    mapping = {(p["arm"], p["family"], float(p["target"])): p for p in points}
    require(len(mapping) == len(points), "Duplicate normalization contrast")
    targets = sorted({float(p["target"]) for p in points})
    require(targets, "No matched normalization contrasts")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True, sharey="col")
    fig.subplots_adjust(top=.84, bottom=.15, left=.09, right=.98, hspace=.30, wspace=.22)
    fig.suptitle("Normalization − raw at shared evaluation-curve recall", y=.995, fontsize=13)
    model_legend(fig, .960)
    style_handles = [Line2D([0], [0], color=".25", marker="o", ls="none", label="estimate with available 95% interval"),
                     Line2D([0], [0], color=".25", marker="o", markerfacecolor="white", ls="none", label="estimate; interval unavailable")]
    fig.legend(handles=style_handles, loc="upper center", bbox_to_anchor=(.5, .912), ncol=2, fontsize=9)
    missing_points = missing_intervals = 0
    for row_index, family in enumerate(FAMILIES):
        for column, (metric, scale, title) in enumerate((("mean_tokens", 1, "Change in mean document-text tokens"),
                                                        ("macro_precision", 100, "Change in macro precision (pp)"))):
            ax = axes[row_index, column]
            ax.set_title(policy(family) + " — " + title)
            ax.axhline(0, color=".35", lw=.85, ls=":")
            for model_index, arm in enumerate(ARMS):
                for target in targets:
                    item = mapping.get((arm, family, target))
                    require(item is not None, "Missing normalization contrast setting")
                    value = item.get("metrics", {}).get(metric)
                    if not finite(value):
                        missing_points += 1
                        continue
                    x = 100 * target + (model_index - 1.5) * .75
                    bounds = ci_bounds(item, metric)
                    if bounds is not None:
                        low, high = (scale * v for v in bounds)
                        middle = (low + high) / 2
                        # Draw interval independently: percentile intervals need
                        # not contain the observed point estimate.
                        ax.errorbar(x, middle, yerr=[[middle - low], [high - middle]], fmt="none",
                                    color=COLORS[arm], capsize=3, elinewidth=1.05, capthick=1.05, zorder=2)
                    else:
                        missing_intervals += 1
                    ax.plot(x, scale * value, marker="o", ms=4.6, color=COLORS[arm], ls="none",
                            markerfacecolor=COLORS[arm] if bounds is not None else "white", zorder=4)
            ax.set_xticks([100 * t for t in targets])
            ax.set_xlim(100 * min(targets) - 3, 100 * max(targets) + 3)
            ax.grid(True, axis="y", color=".87", lw=.6)
            ax.set_ylabel("Δ tokens" if metric == "mean_tokens" else "Δ macro precision (pp)")
            if row_index == 1:
                ax.set_xlabel("Shared evaluation-curve macro-recall target (%)")
    fig.text(.5, .035, "Descriptive recall-standardized contrasts, not deployed calibrated policies. Model offsets are for legibility only.\n"
             "Pointwise paired-query percentile intervals; no multiplicity adjustment. Unsupported points/intervals are listed in the report.",
             ha="center", va="bottom", fontsize=8)
    return save_figure(fig, output), {"unsupported_metric_points": missing_points,
                                      "finite_points_without_intervals": missing_intervals}


def point_table(points, calibration=False):
    header = "| ASR | Input | Policy | " + ("Calibration recall (%) | " if calibration else "")
    header += "Evaluation recall (%) [95% CI] | Macro precision (%) [95% CI] | Micro precision (%) [95% CI] | Mean tokens [95% CI] |"
    fields = 8 if calibration else 7
    lines = [header, "| " + " | ".join(["---"] * fields) + " |"]
    for point in sorted(points, key=lambda p: (FAMILIES.index(p["family"]), ARMS.index(p["arm"]), CONDITIONS.index(p["condition"]))):
        values = [NAMES[point["arm"]], point["condition"], policy(point["family"])]
        if calibration:
            values.append(numeric(point.get("calibration_recall"), scale=100))
        values += [estimate(point, metric, scale=100 if metric != "mean_tokens" else 1,
                            digits=1 if metric == "mean_tokens" else 3)
                   for metric in ("achieved_recall", "macro_precision", "micro_precision", "mean_tokens")]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def contrast_table(contrasts):
    lines = ["| Panel | Contrast | ASR | Input / family | Target recall (%) | Δ recall (pp) [95% CI] | Δ tokens [95% CI] | Δ macro precision (pp) [95% CI] | Δ micro precision (pp) [95% CI] |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in contrasts:
        context = "/".join(str(item[key]) for key in ("condition", "family") if item.get(key) is not None)
        values = [str(item["panel"]), str(item["kind"]), NAMES.get(item.get("arm"), str(item.get("arm", "all"))),
                  context or "—", numeric(item.get("target"), scale=100) if item.get("target") is not None else "—",
                  estimate(item, "achieved_recall", scale=100, signed=True),
                  estimate(item, "mean_tokens", digits=1, signed=True),
                  estimate(item, "macro_precision", scale=100, signed=True),
                  estimate(item, "micro_precision", scale=100, signed=True)]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def report_lines(data, figures):
    metadata = data["metadata"]
    cal_n, eval_n = metadata.get("calibration_queries", 350), metadata.get("evaluation_queries", 117)
    lines = ["# ASR recall–token curves, calibration transfer and paired intervals", "",
        "This follow-up reuses the saved four-model raw/normalized transcriptions and query vectors. "
        "It makes **zero new ASR, encoder or external API calls**. The fixed normalization rule and the original 0.8925 control are retained.", "",
        "## Two distinct estimands", "",
        "1. **Calibration transfer:** choose threshold/top-k mixture parameters using **" + str(cal_n)
        + " practice queries**, then apply those exact indices and mixture weights to the **" + str(eval_n)
        + " remaining dev queries**, excluding the eight pilot queries. A common calibration target does **not** guarantee equal achieved evaluation recall. "
        "Evaluation recall is reported explicitly; this is the transfer result for a deployable, calibration-chosen rule.",
        "2. **Descriptive evaluation-curve standardization:** evaluate the predeclared curve functional at a common recall target on the same remaining-dev evaluation queries. "
        "Its interpolation uses evaluation labels as part of a descriptive metric. It is **not** an independently selected deployment threshold, "
        "and it must not replace the frozen calibration-transfer result.", "",
        "Both analyses are exploratory: all 475 recordings' fixed-policy summaries have already been inspected; "
        "the normalization rule was designed after the original eight-query pilot and the old cutoff used public dev labels. "
        "The new calibration/evaluation query sets are disjoint for this calibration step, not a newly untouched holdout.", "",
        "## Metric and mixture definitions", "",
        "- Macro recall averages each original query's retrieved positive-qrel fraction. Macro precision averages each query's hits/returned-documents ratio.",
        "- Mean document-text tokens exclude question/system text, labels, separators and chat framing. They are not billed prompt tokens or an end-to-end cost measure.",
        "- A two-policy mixture uses **one query-independent probability** for every query. Expected recall, macro precision, hits, documents and token cost are mixed analytically. "
        "Micro precision is **mixed expected hits divided by mixed expected documents**, not a linear interpolation of the two micro-precision ratios.",
        "- Interpolation is between adjacent distinct-recall candidate points. Recall plateaus select lowest mean tokens, then mean documents, numeric parameter and original index, "
        "with absolute recall tolerance 1e-12. This is not a globally optimized convex-envelope policy. Targets outside observed support are unsupported; no extrapolation is performed.",
        "- Intervals use **" + str(metadata.get("bootstrap_replicates", "unavailable"))
        + " query-level bootstrap replicates**. Calibration and evaluation are resampled separately; within each set the same indices are shared by every model, input condition and policy. "
        "For calibration-transfer intervals, parameter selection is repeated within each replicate. Fixed 0.8925/top-5 controls are not tuned. "
        "Intervals are pointwise percentile intervals, unadjusted for multiple comparisons.",
        "- Matched-curve bootstrap replicates recompute curve means and interpolation. A panel/target uses a common complete support mask across all settings. "
        "Support counts and unavailable intervals are disclosed below; they are not silently counted as zero.", "",
        "## Execution metadata", "", "| Field | Value |", "| --- | --- |"]
    allowed = ("calibration_queries", "evaluation_queries", "bootstrap_replicates", "seed", "slurm_job_id",
               "source_commit", "git_head", "numpy_version", "scipy_version", "matplotlib_version")
    for key in allowed:
        if key in metadata:
            lines.append("| " + key + " | `" + str(metadata[key]) + "` |")
    lines += ["| rendering_slurm_job_id | `" + os.environ["SLURM_JOB_ID"] + "` |", "",
              "## Observed recall–token curves", "",
              "The operating-range plot shows macro recall 20–70% and mean document-text tokens 0–10,000. "
              "Axis limits crop the display only; the full candidate-range plot is provided separately. "
              "The all-475 panels are descriptive. The remaining-dev panels show the evaluation cohort. "
              "Neither line plot is a simultaneous confidence band.", "",
              "![Operating-range recall–token curves](figures/asr-recall-token-operating.png)", "",
              "[Operating-range PDF](figures/asr-recall-token-operating.pdf) · "
              "[Full-range PNG](figures/asr-recall-token-full.png) · [Full-range PDF](figures/asr-recall-token-full.pdf)", "",
              "## Fixed controls on the evaluation cohort", "",
              "The original top-5 and absolute cutoff 0.8925 are retained without retuning. "
              "Tables show the point estimate followed by its query-level 95% interval. 'CI unavailable' does not mean a zero-width interval.", ""]
    lines += point_table(data["fixed_controls"])
    lines += ["", "## Calibration transfer: frozen practice-selected parameters", "",
              "The same calibration recall target is used for every arm, but the evaluation recalls below may differ. "
              "Cost/precision differences at unequal achieved evaluation recall are **not** automatically matched-recall efficiency claims."]
    for target in sorted({float(p["target"]) for p in data["calibrated_points"]}):
        lines += ["", "### Calibration target: " + numeric(target, digits=1, scale=100) + "%", ""]
        lines += point_table([p for p in data["calibrated_points"] if float(p["target"]) == target], calibration=True)
    lines += ["", "### Calibration-selected policy parameters", "",
              "`high_weight` is the probability of applying the high-recall candidate to any query. "
              "The low/high values are the original numerical k or threshold parameters; 'high' refers to recall, not numerical parameter order.", "",
              "| ASR | Input | Policy | Target recall (%) | Low parameter | High parameter | High weight |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for point in data["calibrated_points"]:
        params = point.get("parameters", {})
        lines.append("| " + " | ".join([NAMES[point["arm"]], point["condition"], policy(point["family"]),
            numeric(point["target"], scale=100), numeric(params.get("low"), digits=6),
            numeric(params.get("high"), digits=6), numeric(params.get("high_weight"), digits=6)]) + " |")
    lines += ["", "## Descriptive comparisons at shared evaluation-curve recall", "",
              "These standardized points answer a curve-comparison question on the evaluation data. "
              "Their parameters are not claimed to have been independently calibrated for deployment."]
    for target in sorted({float(p["target"]) for p in data["matched_points"]}):
        lines += ["", "### Evaluation-curve target: " + numeric(target, digits=1, scale=100) + "%", ""]
        lines += point_table([p for p in data["matched_points"] if float(p["target"]) == target])
    normalizations = [p for p in data["contrasts"] if p["panel"] == "matched" and p["kind"] == "normalization"]
    lines += ["", "## Normalization contrasts at shared evaluation-curve recall", "",
              "Differences are **normalized minus raw** at the same descriptive evaluation-curve recall target. "
              "Negative token differences mean less document text; positive precision differences mean higher labelled precision. "
              "These signs are definitions, not a claim about the observed direction or statistical significance.", "",
              "![Paired normalization effects at shared recall](figures/asr-normalization-effects.png)", "",
              "[Standalone normalization-effects PDF](figures/asr-normalization-effects.pdf)", ""]
    lines += contrast_table(normalizations)
    lines += ["", "## Bootstrap support", "",
              "All settings within a panel/target share the same valid replicate set. "
              "When fewer than 95% of replicates are valid, intervals are suppressed. "
              "Even with an available interval, inference is conditional on the disclosed common support mask.", "",
              "| Panel | Target recall (%) | Valid replicates | Total replicates | Available intervals |",
              "| --- | ---: | ---: | ---: | --- |"]
    require(isinstance(data.get("support"), list) and data["support"], "Missing bootstrap support disclosure")
    for item in data["support"]:
        lines.append("| " + " | ".join([str(item["panel"]),
            numeric(item.get("target"), scale=100) if item.get("target") is not None else "—",
            str(item["valid_bootstraps"]), str(item["total_bootstraps"]),
            "yes" if item["intervals_available"] else "no"]) + " |")
    other = [p for p in data["contrasts"] if not (p["panel"] == "matched" and p["kind"] == "normalization")]
    lines += ["", "## Other saved paired contrasts", "",
              "The remaining fixed-policy, calibration-transfer and model-versus-GLM contrasts are retained below. "
              "The reported delta is always left_key minus right_key from the saved analysis. "
              "Check the accompanying recall difference before interpreting calibration-transfer token differences.", "",
              "<details>", "<summary>All other paired contrasts</summary>", ""]
    lines += contrast_table(other)
    lines += ["", "</details>", "", "## Interpretation limits", "",
              "The shared corpus, original qrels, pinned tokenizer and frozen normalization rule are unchanged. "
              "Normalization combines acronym, case, punctuation/hyphen and whitespace changes; it is not a component-specific causal ablation. "
              "Agreement after normalization does not establish that any transcript is correct, and common errors can remain.", "",
              "No new reference transcripts, WER experiment, repeated-ASR variance estimate, answer generation, judging or cross-model latency benchmark is included. "
              "Precision and recall concern the provided relevance labels, which may be incomplete. "
              "These exploratory paired intervals do not account for annotation bias, ASR service-revision variability or all earlier analyst choices. "
              "The figures and tables present saved measurements without asserting an effect direction or significance.", "",
              "## Artifacts", "",
              "This public note and its plots contain aggregate metrics only. Raw audio, transcript text and query IDs are not embedded. "
              "The full-range figures expose the saved candidate grid rather than extrapolating beyond it.", "",
              "| Figure | PNG | PDF |", "| --- | --- | --- |"]
    for stem, label in (("asr-recall-token-operating", "Operating-range recall–token curves"),
                        ("asr-recall-token-full", "Full-range recall–token curves"),
                        ("asr-normalization-effects", "Paired normalization effects")):
        lines.append("| " + label + " | [PNG](figures/" + stem + ".png) | [PDF](figures/" + stem + ".pdf) |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and socket.getfqdn().endswith(".inf.ed.ac.uk"),
            "Run only inside an Edinburgh Slurm allocation")
    os.umask(0o077)
    run, output = args.run.resolve(), args.output.resolve()
    require(output not in {(run / "analysis.json").resolve(), (run / "curves.csv").resolve()},
            "Markdown output would overwrite saved input evidence")
    data = read(run / "analysis.json")
    for key in ("metadata", "fixed_controls", "calibrated_points", "matched_points", "contrasts"):
        require(key in data, "Analysis schema missing " + key)
    require(len(data["fixed_controls"]) == 16 and len(data["calibrated_points"]) == len(data["matched_points"]) == 48,
            "Unexpected fixed/calibrated/matched point coverage")
    for key in ("new_asr_calls", "new_asr_inferences", "asr_calls", "new_encoder_calls", "new_encoder_inferences",
                "new_api_calls", "new_api_requests"):
        if key in data["metadata"]:
            require(data["metadata"][key] == 0, "Unexpected new inference/API activity")
    rows = read_curves(run / "curves.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    folder = output.parent / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    files, plot_checks = [], {}
    for full_range, stem in ((False, "asr-recall-token-operating"), (True, "asr-recall-token-full")):
        paths, checks = recall_token_plot(rows, folder / stem, full_range=full_range)
        files.extend(paths)
        plot_checks[stem] = checks
    paths, checks = normalization_plot(data["contrasts"], folder / "asr-normalization-effects")
    files.extend(paths)
    plot_checks["asr-normalization-effects"] = checks
    lines = report_lines(data, files)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"status": "rendered", "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "matplotlib_version": matplotlib.__version__, "curve_rows": len(rows),
                "input_sha256": {name: digest(run / name) for name in ("analysis.json", "curves.csv")},
                "plot_checks": plot_checks,
                "output_sha256": {str(path.relative_to(output.parent)): digest(path) for path in files + [output]}}
    (output.parent / "render-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
