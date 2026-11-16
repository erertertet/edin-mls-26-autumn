"""Offline oracle bounds for stopping in cached embedding-ranked top-20 lists.

No API, ASR, embedding, or GPU work. Bounds allow randomization at a budget
boundary. They use test relevance labels deliberately and are not deployable.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import kron, eye, csr_matrix


def hull(x, y):
    """Increasing-cost upper concave hull; stop at cheapest maximum-recall point."""
    out = []
    for i in np.argsort(x, kind="stable"):
        if out and x[i] == x[out[-1]]:
            if y[i] <= y[out[-1]] + 1e-14:
                continue
            out.pop()
        while len(out) >= 2:
            a, b = out[-2:]
            cross = (x[b]-x[a]) * (y[i]-y[b]) - (y[b]-y[a]) * (x[i]-x[b])
            if cross < -1e-12:
                break
            out.pop()
        out.append(int(i))
    best = float(np.max(y))
    end = next(j for j, i in enumerate(out) if y[i] >= best - 1e-12)
    return np.array(out[:end+1], dtype=int)


def query_oracle(cost, recall, allowed=None):
    """Merge decreasing marginal slopes of individual query hulls (exact LP)."""
    n, width = cost.shape
    edges = []
    choices = []
    for q in range(n):
        valid = np.arange(width) if allowed is None else allowed[q]
        hh = valid[hull(cost[q, valid], recall[q, valid])]
        choices.append(hh.tolist())
        for a, b in zip(hh[:-1], hh[1:]):
            dc, dr = float(cost[q, b]-cost[q, a]), float(recall[q, b]-recall[q, a])
            assert dc > 0 and dr > 0
            edges.append((dr/dc, q, int(a), int(b), dc, dr))
    edges.sort(key=lambda e: (-e[0], e[1], e[2]))
    selected = np.zeros(n, dtype=int)
    x, y = [float(cost[:, 0].mean())], [float(recall[:, 0].mean())]
    for slope, q, a, b, dc, dr in edges:
        assert selected[q] == a
        selected[q] = b
        x.append(x[-1] + dc/n)
        y.append(y[-1] + dr/n)
    assert np.allclose(y[-1], recall[:, -1].mean())
    assert np.isclose(x[-1], cost[np.arange(n), selected].mean())
    return np.array(x), np.array(y), selected, edges


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    src = args.run / "retrieval-frontier"
    a = np.load(src / "query-prefix-metrics.npz")
    cost, recall = a["input_tokens"].astype(float), a["recall"]
    ids = a["query_ids"].tolist()
    allq = {q["query_id"]: q for q in [json.loads(s) for s in (args.run / "queries.jsonl").read_text().splitlines()]}
    scores = np.array([allq[q]["similarities"] for q in ids])
    assert np.all(np.diff(scores, axis=1) <= 0)
    assert np.all(np.diff(cost, axis=1) > 0)
    assert np.all(np.diff(recall, axis=1) >= 0)
    n, width = cost.shape
    fx, fy = cost.mean(axis=0), recall.mean(axis=0)
    fh = hull(fx, fy)
    ox, oy, chosen, edges = query_oracle(cost, recall)
    # Similarity threshold cannot split tied scores. Top-1 fallback remains legal.
    allowed = [np.array(sorted(set([0, width-1] + [k for k in range(width-1) if scores[q,k] > scores[q,k+1]]))) for q in range(n)]
    sx, sy, schosen, sedges = query_oracle(cost, recall, allowed)

    # Enumerate every distinct global threshold, rather than the old coarse grid.
    events = sorted([(float(scores[q,k]), q, k) for q in range(n) for k in range(1,width)], reverse=True)
    selected = np.zeros(n, dtype=int)
    tx, ty, thresholds = [float(fx[0])], [float(fy[0])], [float(scores.max()+1e-6)]
    pos = 0
    while pos < len(events):
        value = events[pos][0]
        updates = {}
        while pos < len(events) and events[pos][0] == value:
            _, q, k = events[pos]
            updates[q] = max(updates.get(q, 0), k)
            pos += 1
        dc, dr = 0.0, 0.0
        for q, k in updates.items():
            dc += cost[q,k]-cost[q,selected[q]]
            dr += recall[q,k]-recall[q,selected[q]]
            selected[q] = k
        tx.append(tx[-1]+dc/n); ty.append(ty[-1]+dr/n); thresholds.append(value)
    tx, ty = np.array(tx), np.array(ty)
    th = hull(tx, ty)
    assert np.allclose([tx[-1],ty[-1]], [fx[-1],fy[-1]])
    for i in np.linspace(0,len(tx)-1,17,dtype=int):
        kk = np.clip((scores >= thresholds[i]).sum(axis=1),1,width)-1
        assert np.allclose([cost[np.arange(n),kk].mean(),recall[np.arange(n),kk].mean()],[tx[i],ty[i]])

    old = json.loads((src / "summary.json").read_text())
    points = [{"name":f"fixed{k}","tokens":float(fx[k-1]),"recall":float(fy[k-1])} for k in [1,2,3,5,10,20]]
    points += [{"name":r["family"],"tokens":r["mean_input_tokens"],"recall":r["macro_recall"]} for r in old["frozen_policy_points"]]
    for point in points:
        budget = point["tokens"]
        point.update({"query_specific_topk_oracle":float(np.interp(budget,ox,oy)),
                      "query_specific_similarity_oracle":float(np.interp(budget,sx,sy)),
                      "shared_topk_envelope":float(np.interp(budget,fx[fh],fy[fh])),
                      "shared_similarity_envelope":float(np.interp(budget,tx[th],ty[th]))})
        point["oracle_gap_pp"] = 100*(point["query_specific_topk_oracle"]-point["recall"])
    # Independently verify three points using a general linear-program solver.
    eq = kron(eye(n,format="csr"),csr_matrix(np.ones((1,width))),format="csr")
    checks = []
    for budget in [float(fx[1]),float(fx[4]),float(fx[9])]:
        result = linprog(-recall.flatten()/n, A_ub=csr_matrix(cost.reshape(1,-1)/n), b_ub=[budget],
                         A_eq=eq,b_eq=np.ones(n),bounds=(0,None),method="highs")
        assert result.success, result.message
        predicted = float(np.interp(budget,ox,oy))
        assert abs(-result.fun-predicted) < 1e-8
        checks.append({"tokens":budget,"linear_program_recall":float(-result.fun),"curve_recall":predicted})
    grid = np.unique(np.concatenate([ox,sx]))
    max_difference = float(np.max(np.abs(np.interp(grid,ox,oy)-np.interp(grid,sx,sy))))
    summary = {"queries":n,"pool":width,"min_documents":1,"additional_api_calls":0,
               "pool_macro_recall_ceiling":float(fy[-1]),
               "top20_tokens":float(fx[-1]),"topk_oracle_saturation_tokens":float(ox[-1]),
               "similarity_oracle_saturation_tokens":float(sx[-1]),
               "oracle_mean_k_at_saturation":float((chosen+1).mean()),
               "global_similarity_settings":len(tx),"score_tie_pairs":int((np.diff(scores,axis=1)==0).sum()),
               "max_difference_between_query_specific_oracles":max_difference,
               "max_randomization_vs_feasible_step_gap_pp":100/n,
               "linear_program_validation":checks,"comparisons":points,
               "notes":["All 324 original retrieval-test queries; independent of the paused 101-query sufficiency run.",
                        "Primary per-query oracle can choose k(q) using qrels but must retain an embedding-ranking prefix of length 1..20.",
                        "Per-query similarity oracle uses tau(q), retaining score>=tau with top-1 fallback. It can choose the same prefixes except score ties.",
                        "Shared rule envelopes choose one global k or one global tau (or a query-independent mixture); these have less query-specific information.",
                        "Curves permit randomized budget-boundary choices. The query-specific oracle is the exact fractional optimum under a mean token budget, independently checked by linear programming.",
                        "Each vertex of the per-query oracle is a feasible deterministic allocation. Between adjacent vertices its lower endpoint is within 1/N recall of the fractional bound.",
                        "Recall is macro qrels document recall, not evidence sufficiency or answer correctness. Denominators include relevant documents outside top 20.",
                        "X is full generator input tokens, using previously API-validated tokenizer counts. Oracle decision cost is assumed zero; no planner, generation output or evaluation cost included.",
                        "This is a hindsight diagnostic using test relevance labels, not a deployable retriever or a newly validated policy."]}
    (args.out / "summary.json").write_text(json.dumps(summary,indent=2))
    curves = []
    for family,x,y in [("query_specific_topk",ox,oy),("query_specific_similarity",sx,sy),("shared_topk_envelope",fx[fh],fy[fh]),("shared_similarity_envelope",tx[th],ty[th])]:
        curves.extend({"family":family,"mean_input_tokens":float(xx),"macro_recall":float(yy)} for xx,yy in zip(x,y))
    with (args.out / "curves.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    with (args.out / "comparisons.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
    np.savez_compressed(args.out / "oracle-allocation.npz",query_ids=np.array(ids),topk_saturation_k=chosen+1,similarity_saturation_k=schosen+1)

    plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
    maxx = float(fx[-1])*1.045
    def draw(ax, zoom=False):
        xx=np.linspace(fx[0],maxx,1000)
        ax.plot(np.r_[ox,maxx],100*np.r_[oy,oy[-1]],color="#20262e",lw=2.8,label="Per-query top-k oracle")
        ax.plot(np.r_[sx,maxx],100*np.r_[sy,sy[-1]],color="#bc4c82",lw=2,ls=(0,(4,4)),label="Per-query similarity oracle (overlaps)")
        ax.plot(np.r_[fx[fh],maxx],100*np.r_[fy[fh],fy[fh][-1]],color="#245a81",lw=2.3,label="Best shared top-k / mixtures")
        ax.scatter(fx,fy*100,color="#245a81",s=20,zorder=5)
        ax.plot(np.r_[tx[th],maxx],100*np.r_[ty[th],ty[th][-1]],color="#d45c27",lw=2.2,label="Best shared similarity cutoff / mixtures")
        better=np.maximum(np.interp(xx,fx[fh],fy[fh]),np.interp(xx,tx[th],ty[th]))
        ax.fill_between(xx,100*better,100*np.interp(xx,ox,oy),color="#506775",alpha=.10,label="Headroom for per-query stopping")
        ax.axhline(100*fy[-1],color="#727980",ls=":",lw=1)
        for k in [1,3,5,10,20]:
            if not zoom or fx[k-1]<1600:
                ax.annotate(f"k={k}",(fx[k-1],100*fy[k-1]),xytext=(5,-17),textcoords="offset points",fontsize=10,color="#245a81")
        for row in old["frozen_policy_points"]:
            if row["family"]=="absolute_radius":
                ax.scatter(row["mean_input_tokens"],100*row["macro_recall"],marker="D",s=65,color="#d45c27",edgecolors="white",zorder=7,label="Existing dev-selected distance cutoff")
        ax.set(xlabel="Mean generator input tokens per query",ylabel="Relevant-document recall (%)",xlim=(250,1600 if zoom else maxx),ylim=(20,63))
        ax.grid(alpha=.18)
        if not zoom:
            ax.annotate(f"Top-20 pool ceiling: {100*fy[-1]:.2f}%",(maxx*.98,100*fy[-1]),xytext=(0,8),textcoords="offset points",ha="right",fontsize=10,color="#555")
        ax.legend(loc="lower right",fontsize=9,framealpha=.97)
    for zoom,name in [(False,"retrieval-oracle-frontier"),(True,"retrieval-oracle-zoom")]:
        fig,ax=plt.subplots(figsize=(11.8,7.6))
        draw(ax,zoom)
        fig.suptitle("Oracle headroom under the same embedding ranking",fontsize=18,weight="bold",y=.97)
        fig.text(.09,.905,"324 queries · fixed top-20 candidates · at least one document per query",fontsize=11,color="#555")
        fig.text(.09,.035,"Per-query k and similarity cutoffs have the same oracle when scores are distinct. Shared cutoffs are more restricted.\nLines allow randomized budget allocation. Recall uses qrels, not answer sufficiency. No new API calls; inference run remains paused.",fontsize=9.5,color="#555",linespacing=1.5)
        fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.86)
        for ext in ["png","svg"]:fig.savefig(args.out/f"{name}.{ext}",dpi=180,facecolor="white")
        plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
