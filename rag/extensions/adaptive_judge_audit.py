"""Repeat judging on a preselected test subset after the full run completes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import time

from dotenv import load_dotenv
from adaptive_retrieval_batch import ROOT, ARMS, Run, Judge, JUDGE_SYSTEM_PROMPT, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from pathlib import Path
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--budget", type=float, default=40)
    parser.add_argument("--model", default="mistralai/mistral-small-2603")
    parser.add_argument("--judge", default="google/gemini-2.5-flash")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if json.loads((args.out / "status.json").read_text())["state"] != "complete":
        raise RuntimeError("Finish the main experiment before the judge audit")
    run = Run(args)
    plan = json.loads((args.out / "judge-audit-plan.json").read_text())
    path = args.out / "judge-audit-results.jsonl"
    existing = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(row["query_id"], row["arm"], row["repeat"]) for row in existing}
    expected = len(plan["query_ids"]) * len(ARMS) * plan["additional_repeats"]

    def evaluate(qid, arm, repeat):
        original = run.results[(qid, arm)]
        gold = run.gold[original["source_split"]][qid]["gold_documents"]
        payload = {"question": original["transcript"], "candidate_answer": original["answer"], "gold_evidence": [{"record_id": row["record_id"], "text": row["text"]} for row in gold]}
        prompt = 'Return JSON exactly as {"correctness":0,"grounding":0,"completeness":0,"explanation":"brief reason"}.\n\nINPUT:\n' + json.dumps(payload, ensure_ascii=False)
        for attempt in range(3):
            response, record = run.call("heldout_judge_audit", qid, arm, "judge", [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
            try:
                scores = Judge.parse(response.choices[0].message.content or "")
                break
            except (ValueError, TypeError, KeyError):
                if attempt == 2:
                    raise
        row = {"query_id": qid, "arm": arm, "repeat": repeat, "id": record["id"], "scores": scores, "score": sum(scores[key] for key in ["correctness", "grounding", "completeness"]) / 12, "original_score": original["score"]}
        run.append(path.name, row)
        return row

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, qid, arm, repeat) for qid in plan["query_ids"] for arm in ARMS for repeat in range(2, plan["additional_repeats"] + 2) if (qid, arm, repeat) not in done]
        for future in as_completed(futures):
            existing.append(future.result())
            atomic_json(args.out / "judge-audit-status.json", {"completed": len(existing), "expected": expected, "elapsed_seconds": time.perf_counter() - started})
    run.reconcile()
    summary = {"queries": len(plan["query_ids"]), "additional_judgements": len(existing), "by_arm": []}
    for arm in ARMS:
        original, averaged, dispersions, changed = [], [], [], []
        for qid in plan["query_ids"]:
            base = run.results[(qid, arm)]["score"]
            values = [base] + [row["score"] for row in existing if row["query_id"] == qid and row["arm"] == arm]
            original.append(base); averaged.append(statistics.mean(values))
            dispersions.append(statistics.pstdev(values)); changed.append(len(set(values)) > 1)
        summary["by_arm"].append({"arm": arm, "original_mean_score": statistics.mean(original), "three_judge_mean_score": statistics.mean(averaged), "mean_within_answer_std": statistics.mean(dispersions), "answers_with_disagreeing_scores": sum(changed)})
    summary["reported_audit_api_usd"] = sum(float(row["usage"]["cost"]) for row in run.requests if row["phase"] == "heldout_judge_audit")
    atomic_json(args.out / "judge-audit-summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
