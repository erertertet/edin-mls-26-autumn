"""Nine operating points, including exploratory recall targets; generation then judging.

Uses saved transcripts and full-corpus rankings. No ASR, encoding or reranker.
The evaluation queries were inspected in previous studies: this is a follow-up.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import threading
import time

import numpy as np
from dotenv import load_dotenv

from adaptive_retrieval_batch import (
    Run, ROOT, read_rows, atomic_json, CourseData, SimpleContextBuilder,
    GENERATOR_SYSTEM, Judge, JUDGE_SYSTEM_PROMPT, build_public_gold,
)

EXPLORATORY = {
    'absolute_r40': ('absolute_radius', .8795, .40),
    'absolute_r50': ('absolute_radius', .8655, .50),
    'relative_r40': ('relative_gap', .0176, .40),
    'relative_r50': ('relative_gap', .027200000000000002, .50),
}
ARMS = ['fixed5', 'fixed10', 'fixed20', 'absolute_radius', 'relative_gap', *EXPLORATORY]
DIMS = ['correctness', 'grounding', 'completeness']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count(scores, arm, tau, delta, fixed_k=1):
    if arm.startswith('fixed'):
        return int(arm[5:]), False
    if arm == 'topk':
        return fixed_k, False
    if arm in EXPLORATORY:
        family, value, _ = EXPLORATORY[arm]
        threshold = value if family == 'absolute_radius' else float(scores[0]) - value
    else:
        assert arm in ['absolute_radius', 'relative_gap']
        threshold = tau if arm == 'absolute_radius' else float(scores[0]) - delta
    selected = int(np.searchsorted(-scores.astype(np.float64), -threshold, side='right'))
    return max(1, selected), selected == 0


class Followup(Run):
    # Reuse the established metered API transport and billing reconciliation only.
    def __init__(self, args):
        self.args = args
        self.out = args.out
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.local = threading.local()
        self.started = time.perf_counter()
        self.reserved = 0.0
        self.errors = []
        self.requests = read_rows(self.out / 'requests.jsonl')
        self.spent = sum(float(r['usage']['cost']) for r in self.requests)
        self.answers = {(r['query_id'], r['arm']): r for r in read_rows(self.out / 'answers.jsonl')}
        self.results = {(r['query_id'], r['arm']): r for r in read_rows(self.out / 'results.jsonl')}
        self.plan = json.loads((args.base / 'partition.json').read_text())
        self.dev = self.plan['tuning_query_ids']
        self.test = self.plan['heldout_test_query_ids']
        assert len(self.dev) == 125 and len(self.test) == 324 and not set(self.dev) & set(self.test)
        source = read_rows(args.base / 'queries.jsonl')
        self.queries = {r['query_id']: r for r in source if r['query_id'] in set(self.dev+self.test)}
        self.rankings = np.load(args.pool / 'rankings.npz')
        self.corpus_ids = self.rankings['corpus_ids'].tolist()
        rank_indices = self.rankings['indices']
        rank_scores = self.rankings['similarities']
        self.corpus_size = len(self.corpus_ids)
        assert self.corpus_size == 24000
        for i,qid in enumerate(self.rankings['query_ids'].tolist()):
            self.queries[qid]['indices'] = rank_indices[i]
            self.queries[qid]['similarities'] = rank_scores[i]
        assert len(self.queries) == 449
        assert set(self.queries) == set(self.dev + self.test)
        for q in self.queries.values():
            assert len(q['indices']) == len(np.unique(q['indices'])) == len(q['similarities']) == self.corpus_size
            assert np.isfinite(q['similarities']).all()
            assert np.all(np.diff(q['similarities']) <= 1e-6)
        self.data = CourseData(ROOT / 'data/rag_corpus')
        self.builder = SimpleContextBuilder()
        self.qrels = {s: self.data.load_qrels(s) for s in ['dev', 'practice']}
        self.gold = None
        self.total = len(self.test) * len(ARMS)
        config = {
            'generator': args.model, 'judge': args.judge, 'arms': ARMS,
            'api_workers': args.workers, 'budget_usd': args.budget,
            'pool_size': self.corpus_size, 'document_count_cap': None, 'min_documents': 1,
            'temperature_generator': 0.1, 'temperature_judge': 0,
            'max_tokens_generator': 2048, 'max_tokens_judge': 768,
            'reasoning_enabled': False, 'seed': 20261121,
            'objective': 'Original cutoffs: dev macro F1. Four exploratory points: nearest evaluation-curve recall to 40/50 percent, ties prefer fewer text tokens.',
            'exploratory_points': {k:list(v) for k,v in EXPLORATORY.items()},
            'selection_curve_sha256': sha(args.reference/'frontiers/sweep.csv'),
            'absolute_grid': np.linspace(.75, .99, 97).tolist(),
            'relative_grid': np.linspace(0, .08, 81).tolist(),
            'dev_ids': self.dev, 'evaluation_ids': self.test,
            'pool_sha256': sha(args.pool / 'rankings.npz'),
            'partition_sha256': sha(args.base / 'partition.json'),
            'runner_sha256': sha(Path(__file__)),
            'frozen_reference_sha256': sha(args.reference / 'frozen-parameters.json'),
            'fixed_k_display_values': [5,10,20],
            'expanded_control_note': 'Three fixed controls, two dev-selected cutoffs and four exploratory 40/50-percent-recall cutoffs. Fresh answers for all nine rows in one randomized batch.',
            'transport_sha256': sha(Path(__file__).with_name('adaptive_retrieval_batch.py')),
            'generator_system': GENERATOR_SYSTEM, 'judge_system': JUDGE_SYSTEM_PROMPT,
            'notes': [
                'Follow-up on previously inspected evaluation queries; not a pristine holdout.',
                'The four recall targets use previously viewed evaluation qrels, not dev-only tuning. Freeze all policies before generating new answers; no changes after judging.',
                'Cutoff parameters retain the earlier dev macro-F1 selection. Fixed k=5,10,20 are predeclared answer-evaluation controls.',
                'Cutoffs examine the entire corpus. There is no top-50 or other count cap and no silent context truncation.',
                'Independent fresh generation for each query/policy, including identical contexts.',
                'Bounded parallel workers; random query/policy order. All generation precedes all judging.',
                'Judge sees transcript, answer and gold evidence, never policy or selected IDs/scores.',
                'Latency is selection/context construction plus API service, excluding cached upstream stages and queueing.',
                'Logged usage and reserved in-flight budgets are guards, not a provider account spending cap.',
            ],
        }
        path = self.out / 'config.json'
        if path.exists():
            assert json.loads(path.read_text()) == config, 'Immutable config changed: create a new run'
        else:
            atomic_json(path, config)
        self.config = config

    def selected_ids(self, qid, k):
        return [self.corpus_ids[int(i)] for i in self.queries[qid]['indices'][:k]]

    def metrics(self, qid, ids, split):
        gold = set(self.qrels[split][qid])
        hits = len(set(ids) & gold)
        p, r = hits / len(ids), hits / len(gold)
        return dict(hits=hits, relevant_documents=len(gold), precision=p, recall=r,
                    retrieval_f1=2*p*r/(p+r) if p+r else 0)

    def tune(self):
        path = self.out / 'frozen-parameters.json'
        if path.exists():
            self.parameters = json.loads(path.read_text())
            return
        assert not self.answers and not self.results and not self.requests
        self.parameters = json.loads((self.args.reference / 'frozen-parameters.json').read_text())
        self.parameters['reference_run'] = str(self.args.reference)
        self.parameters['expansion_frozen_at'] = now()
        self.parameters['expansion_answers_at_freeze'] = 0
        import csv
        curve = list(csv.DictReader((self.args.reference/'frontiers/sweep.csv').open()))
        points = []
        for arm, (family, value, target) in EXPLORATORY.items():
            r = min((r for r in curve if r['arm']==family), key=lambda r:(abs(float(r['macro_recall'])-target),float(r['mean_evidence_tokens'])))
            assert float(r['parameter']) == value
            points.append(dict(arm=arm, family=family, threshold=value, target_recall=target,
                               observed_recall=float(r['macro_recall']), selection='nearest previously inspected evaluation-curve recall'))
        self.parameters['exploratory_points'] = points
        atomic_json(path,self.parameters)
        print('FROZEN',json.dumps(self.parameters),flush=True)

    def prepare(self):
        self.tune()
        targets = []
        for qid in self.test:
            q = self.queries[qid]
            for arm in ARMS:
                k, fallback = count(q['similarities'], arm, self.parameters['tau'], self.parameters['delta'], self.parameters['fixed_k'])
                ids = self.selected_ids(qid,k)
                targets.append(dict(query_id=qid, arm=arm, k=k, fallback=fallback, record_ids=ids,
                                    **self.metrics(qid, ids, 'practice')))
        atomic_json(self.out / 'retrieval-results.json', targets)
        atomic_json(self.out / 'validation.json', dict(dev_queries=len(self.dev), evaluation_queries=len(self.test),
                    candidate_pool_queries=len(self.queries), candidates_per_query=self.corpus_size,
                    unique_disjoint_partitions=True, sorted_finite_unique_candidates=True, expected_answers=self.total))
        self.status('prepared')

    def generate(self, qid, arm):
        q = self.queries[qid]
        start = time.perf_counter()
        k, fallback = count(q['similarities'], arm, self.parameters['tau'], self.parameters['delta'], self.parameters['fixed_k'])
        ids = self.selected_ids(qid,k)
        prompt = self.builder.build(q['transcript'], ids, self.data)['prompt']
        selection_ms = 1000*(time.perf_counter()-start)
        response, record = self.call('followup_evaluation', qid, arm, 'generator',
            [{'role':'system','content':GENERATOR_SYSTEM}, {'role':'user','content':prompt}])
        answer = (response.choices[0].message.content or '').strip()
        if not answer:
            raise ValueError('Empty generated answer')
        row = dict(query_id=qid, arm=arm, transcript=q['transcript'], answer=answer,
                   k=k, fallback=fallback, record_ids=ids, similarities=q['similarities'][:k].tolist(),
                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                   selection_context_ms=selection_ms, generation_ms=record['latency_ms'],
                   downstream_ms=selection_ms+record['latency_ms'],
                   input_tokens=record['usage']['prompt_tokens'], output_tokens=record['usage']['completion_tokens'],
                   request_id=record['id'], finish_reason=record['finish_reason'], completed_at=now(),
                   **self.metrics(qid, ids, 'practice'))
        with self.lock:
            self.append('answers.jsonl', row)
            self.answers[(qid, arm)] = row

    def judge_answer(self, qid, arm):
        answer = self.answers[(qid, arm)]
        payload = dict(question=answer['transcript'], candidate_answer=answer['answer'],
                       gold_evidence=[{'record_id':d['record_id'], 'text':d['text']} for d in self.gold[qid]['gold_documents']])
        prompt = 'Return JSON exactly as {"correctness":0,"grounding":0,"completeness":0,"explanation":"brief reason"}.\n\nINPUT:\n' + json.dumps(payload, ensure_ascii=False)
        for attempt in range(3):
            response, record = self.call('followup_evaluation', qid, arm, 'judge',
                [{'role':'system','content':JUDGE_SYSTEM_PROMPT}, {'role':'user','content':prompt}])
            raw = response.choices[0].message.content or ''
            self.append('judge-raw.jsonl', dict(query_id=qid, arm=arm, raw=raw, request_id=record['id'], parse_attempt=attempt+1))
            try:
                scores = Judge.parse(raw)
                break
            except (ValueError, KeyError, TypeError):
                if attempt == 2:
                    raise
        row = dict(**answer, scores=scores, score=sum(scores[d] for d in DIMS)/12,
                   judge_request_id=record['id'], judged_at=now())
        with self.lock:
            self.append('results.jsonl', row)
            self.results[(qid, arm)] = row

    def status(self, state):
        with self.lock:
            atomic_json(self.out / 'status.json', dict(state=state, answers=len(self.answers),
                        judged=len(self.results), expected_each=self.total, requests=len(self.requests),
                        logged_cost_usd=self.spent, reserved_usd=self.reserved,
                        errors_this_process=len(self.errors), updated_at=now()))

    def phase(self, name, operation, cache, seed):
        order = [(q,a) for q in self.test for a in ARMS]
        random.Random(seed).shuffle(order)
        self.status(name)
        # Retry missing tasks once after the first pass; persisted successes are never repeated.
        for cycle in range(2):
            todo = iter([k for k in order if k not in cache])
            with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
                pending = {}
                while True:
                    while len(pending) < self.args.workers and not self.stop.is_set():
                        key = next(todo, None)
                        if key is None:
                            break
                        pending[pool.submit(operation, *key)] = key
                    if not pending:
                        break
                    done, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
                    for future in done:
                        key = pending.pop(future)
                        try:
                            future.result()
                        except Exception as error:
                            row = dict(stage=name, query_id=key[0], arm=key[1], error_type=type(error).__name__, cycle=cycle)
                            self.errors.append(row)
                            self.append('failed-tasks.jsonl', row)
                            if len(self.errors) >= 10:
                                self.stop.set()
                    self.status(name)
                    print(json.dumps(dict(stage=name, completed=len(cache), total=self.total, cost=self.spent)), flush=True)
            if len(cache) == self.total or self.stop.is_set():
                break
        if len(cache) != self.total:
            self.status('incomplete_'+name)
            raise RuntimeError('Incomplete stage; inspect failure records and resume unchanged configuration')
        atomic_json(self.out / (name+'-complete.json'), dict(completed_at=now(), count=len(cache)))

    def execute(self):
        self.prepare()
        if self.args.stage == 'prepare':
            return
        preflight = json.loads((self.out/'context-preflight.json').read_text())
        assert preflight['all_fit'] and preflight['settings']==len(ARMS)
        assert preflight['config_sha256']==sha(self.out/'config.json')
        self.phase('generation', self.generate, self.answers, 20261121)
        assert len(self.answers) == self.total
        barrier = self.out / 'judge-start.json'
        if not barrier.exists():
            atomic_json(barrier, dict(started_at=now(), complete_answers=len(self.answers),
                                     answers_sha256=sha(self.out/'answers.jsonl'), blinded_policy=True))
        self.gold = build_public_gold(ROOT/'data/rag_corpus', 'practice')
        self.phase('judging', self.judge_answer, self.results, 20261122)
        self.status('reconciling')
        self.reconcile()
        self.summarise()
        self.status('complete')

    def summarise(self):
        billing = {r['id']:r['billing'] for r in read_rows(self.out/'billing.jsonl') if r.get('billing')}
        costs = Counter()
        for r in self.requests:
            c = billing.get(r['id'], {}).get('total_cost')
            costs[(r['query_id'],r['arm'],r['role'])] += float(r['usage']['cost'] if c is None else c)
        by_arm = []
        for arm in ARMS:
            rows = [self.results[(q,arm)] for q in self.test]
            entry = dict(arm=arm, queries=len(rows), micro_precision=sum(r['hits'] for r in rows)/sum(r['k'] for r in rows),
                         micro_recall=sum(r['hits'] for r in rows)/sum(r['relevant_documents'] for r in rows),
                         fallbacks=sum(r['fallback'] for r in rows), length_limited=sum(r['finish_reason']=='length' for r in rows))
            for f in ['k','precision','recall','retrieval_f1','score','input_tokens','output_tokens','generation_ms','selection_context_ms','downstream_ms']:
                entry['mean_'+f] = statistics.mean(r[f] for r in rows)
            for d in DIMS:
                entry['mean_'+d] = statistics.mean(r['scores'][d] for r in rows)
            for role in ['generator','judge']:
                entry[role+'_usd'] = sum(costs[(q,arm,role)] for q in self.test)
            entry['inference_usd_per_1000'] = entry['generator_usd']/len(rows)*1000
            entry['p95_downstream_ms'] = float(np.percentile([r['downstream_ms'] for r in rows],95))
            by_arm.append(entry)
        rng = np.random.default_rng(20261123)
        indices = rng.integers(0,len(self.test),(5000,len(self.test)))
        comparisons = []
        for i, arm in enumerate(ARMS):
            for baseline in ARMS[:i]:
                item = dict(arm=arm, baseline=baseline)
                for field in ['score','correctness','grounding','completeness','precision','recall','input_tokens','downstream_ms','inference_usd']:
                    def get(q,a):
                        r=self.results[(q,a)]
                        return costs[(q,a,'generator')] if field=='inference_usd' else r['scores'][field] if field in DIMS else r[field]
                    diff=np.array([get(q,arm)-get(q,baseline) for q in self.test])
                    item[field]=dict(difference=float(diff.mean()),bootstrap95=np.percentile(diff[indices].mean(axis=1),[2.5,97.5]).tolist())
                comparisons.append(item)
        atomic_json(self.out/'summary.json', dict(complete=True, evaluation_queries=len(self.test),
                    answers=len(self.answers), judged=len(self.results), api_requests=len(self.requests),
                    billed_requests=len(billing), total_api_usd=sum(costs.values()),
                    parameters=self.parameters, by_arm=by_arm, paired=comparisons,
                    notes=self.config['notes']+['Paired 5000-query-bootstrap intervals are exploratory and not multiplicity-adjusted; omit systematic judge error.']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','full'],default='full')
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--budget',type=float,default=10)
    p.add_argument('--model',default='mistralai/mistral-small-2603')
    p.add_argument('--judge',default='google/gemini-2.5-flash')
    args=p.parse_args()
    assert 1<=args.workers<=16 and args.budget>args.workers*.4
    load_dotenv(ROOT/'.env')
    Followup(args).execute()


if __name__=='__main__':
    main()
