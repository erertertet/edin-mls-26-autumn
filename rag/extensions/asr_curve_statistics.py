"""Pure query-weighted curve statistics and analytic, query-independent mixtures."""
from __future__ import annotations

import numbers

import numpy as np

PLATEAU_TOLERANCE = 1e-12
CURVE_FIELDS = ("macro_recall", "macro_precision", "micro_precision",
                "mean_tokens", "mean_documents", "mean_hits")
MIX_METRICS = ("achieved_recall", "macro_precision", "micro_precision",
               "mean_tokens", "mean_documents", "mean_hits")


def aggregate(hits, docs, tokens, gold, weights=None):
    """Return one metric vector per candidate policy.

    Rows are queries and columns are policies. ``weights`` are common query
    multiplicities/weights, e.g. bootstrap counts, rather than policy weights.
    Macro precision averages per-query ratios; micro precision divides pooled
    weighted hit and document totals. Gold denominators remain per-query.
    """
    hits, docs, tokens = (np.asarray(a, dtype=np.float64) for a in (hits, docs, tokens))
    gold = np.asarray(gold, dtype=np.float64)
    if hits.ndim != 2 or not all(a.shape == hits.shape for a in (docs, tokens)):
        raise ValueError("hits/docs/tokens must share a two-dimensional (query, policy) shape")
    n, policies = hits.shape
    if n < 1 or policies < 1 or gold.shape != (n,):
        raise ValueError("Need nonempty queries/policies and one gold denominator per query")
    if not all(np.isfinite(a).all() for a in (hits, docs, tokens, gold)):
        raise ValueError("Input metrics must be finite")
    if (np.any(gold <= 0) or np.any(docs <= 0) or np.any(tokens < 0)
            or np.any(hits < 0) or np.any(hits > docs) or np.any(hits > gold[:, None])):
        raise ValueError("Invalid positive denominators, hit counts or token costs")
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    if (weights.shape != (n,) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0):
        raise ValueError("Query weights must be finite, nonnegative and have positive total")
    mean_hits = np.average(hits, weights=weights, axis=0)
    mean_documents = np.average(docs, weights=weights, axis=0)
    return {
        "macro_recall": np.average(hits / gold[:, None], weights=weights, axis=0),
        "macro_precision": np.average(hits / docs, weights=weights, axis=0),
        "micro_precision": mean_hits / mean_documents,
        "mean_tokens": np.average(tokens, weights=weights, axis=0),
        "mean_documents": mean_documents,
        "mean_hits": mean_hits,
    }


def _curve_arrays(curve):
    if not isinstance(curve, dict) or any(key not in curve for key in CURVE_FIELDS):
        raise ValueError("Curve is missing aggregate metric arrays")
    arrays = {key: np.asarray(curve[key], dtype=np.float64) for key in CURVE_FIELDS}
    recall = arrays["macro_recall"]
    if recall.ndim != 1 or not recall.size:
        raise ValueError("Curve must contain nonempty one-dimensional metric arrays")
    if any(a.shape != recall.shape or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Curve metric arrays must share a finite one-dimensional shape")
    if any(np.any((arrays[key] < 0) | (arrays[key] > 1))
           for key in ("macro_recall", "macro_precision", "micro_precision")):
        raise ValueError("Recall and precision must be within [0, 1]")
    if (np.any(arrays["mean_documents"] <= 0) or np.any(arrays["mean_tokens"] < 0)
            or np.any(arrays["mean_hits"] < 0)
            or np.any(arrays["mean_hits"] > arrays["mean_documents"])):
        raise ValueError("Curve costs/counts have invalid denominators or ranges")
    if not np.allclose(arrays["micro_precision"], arrays["mean_hits"] / arrays["mean_documents"],
                       rtol=0, atol=PLATEAU_TOLERANCE):
        raise ValueError("Micro precision must equal pooled mean hits / mean documents")
    return arrays


def _unsupported():
    return dict(supported=False, low_index=None, high_index=None,
                high_weight=float("nan"), parameter_low=None, parameter_high=None,
                **{key: float("nan") for key in MIX_METRICS})


def _combine(curve, low, high, weight, parameter_low, parameter_high):
    def mix(field):
        return float((1 - weight) * curve[field][low] + weight * curve[field][high])
    mean_hits, mean_documents = mix("mean_hits"), mix("mean_documents")
    return {
        "supported": True, "low_index": int(low), "high_index": int(high),
        "high_weight": float(weight), "parameter_low": float(parameter_low),
        "parameter_high": float(parameter_high),
        "achieved_recall": mix("macro_recall"),
        "macro_precision": mix("macro_precision"),
        "micro_precision": mean_hits / mean_documents,
        "mean_tokens": mix("mean_tokens"), "mean_documents": mean_documents,
        "mean_hits": mean_hits,
    }


def mix_at_recall(curve, parameters, target):
    """Interpolate adjacent distinct-recall policies, without extrapolation.

    Sort by macro recall. Near-equal recall values form plateaus within absolute
    tolerance 1e-12 of the first value in that plateau. Select a representative
    by minimum tokens, then documents, numeric parameter and original index.
    The same mixture probability applies to every query. This is adjacent-point
    interpolation, not a globally optimized convex-envelope frontier.

    A target outside the original observed recall range is unsupported, even
    just outside an endpoint. A target inside a numerical plateau may differ
    from its chosen representative by <=1e-12; achieved_recall always reports
    the actual analytic result, rather than replacing it with the target.
    """
    arrays = _curve_arrays(curve)
    recall = arrays["macro_recall"]
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape != recall.shape or not np.isfinite(parameters).all():
        raise ValueError("Need one finite numeric parameter per curve point")
    if not isinstance(target, numbers.Real) or not np.isfinite(target):
        raise ValueError("Recall target must be finite and numeric")
    target = float(target)
    if target < float(recall.min()) or target > float(recall.max()):
        return _unsupported()
    order = np.argsort(recall, kind="stable")
    groups = []
    for index in order.tolist():
        if not groups or recall[index] - recall[groups[-1][0]] > PLATEAU_TOLERANCE:
            groups.append([index])
        else:
            groups[-1].append(index)
    selected = [min(group, key=lambda i: (arrays["mean_tokens"][i], arrays["mean_documents"][i],
                                        parameters[i], i)) for group in groups]
    near = [i for i in selected if abs(float(recall[i]) - target) <= PLATEAU_TOLERANCE]
    if near:
        index = min(near, key=lambda i: (abs(float(recall[i]) - target), arrays["mean_tokens"][i],
                                        arrays["mean_documents"][i], parameters[i], i))
        return _combine(arrays, index, index, 0.0, parameters[index], parameters[index])
    below = [i for i in selected if recall[i] < target]
    above = [i for i in selected if recall[i] > target]
    if not below or not above:
        return _unsupported()
    low = max(below, key=lambda i: recall[i])
    high = min(above, key=lambda i: recall[i])
    weight = (target - recall[low]) / (recall[high] - recall[low])
    return _combine(arrays, low, high, float(weight), parameters[low], parameters[high])


def mix_evaluate(curve, selection):
    """Apply calibration-selected indices/probability to evaluation means.

    Evaluation achieved recall is reported as observed, even when it differs
    from the calibration target. No evaluation-label recalibration occurs.
    Unsupported calibration selections propagate unsupported numeric values;
    malformed/incomplete supported selections raise ValueError.
    """
    arrays = _curve_arrays(curve)
    if not isinstance(selection, dict) or "supported" not in selection:
        raise ValueError("Missing calibration selection")
    if selection["supported"] is False:
        return _unsupported()
    required = ("low_index", "high_index", "high_weight", "parameter_low", "parameter_high")
    if selection["supported"] is not True or any(key not in selection for key in required):
        raise ValueError("Incomplete supported calibration selection")
    low, high = selection["low_index"], selection["high_index"]
    if (not isinstance(low, numbers.Integral) or isinstance(low, (bool, np.bool_))
            or not isinstance(high, numbers.Integral) or isinstance(high, (bool, np.bool_))
            or not 0 <= low < len(arrays["macro_recall"]) or not 0 <= high < len(arrays["macro_recall"])):
        raise ValueError("Calibration indices do not identify evaluation curve points")
    try:
        weight = float(selection["high_weight"])
        parameters = (float(selection["parameter_low"]), float(selection["parameter_high"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid mixture weight or parameter metadata") from exc
    if not np.isfinite(weight) or not 0 <= weight <= 1 or not np.isfinite(parameters).all():
        raise ValueError("Mixture weight must be in [0,1] and parameter metadata finite")
    return _combine(arrays, int(low), int(high), weight, *parameters)
