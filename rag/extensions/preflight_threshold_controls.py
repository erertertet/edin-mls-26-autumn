"""Check every full prompt before the rounded-cutoff experiment spends API credit."""
import hashlib
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace
import urllib.request

from tokenizers import Tokenizer
from cutoff_threshold_controls import Followup, ARMS, atomic_json, ROOT, load_dotenv, sha


def main():
    base = Path('/home/zcm/ragexp')
    args = SimpleNamespace(base=base/'full-adaptive-20261115T215241Z',
        pool=base/'uncapped-rankings-20261116', reference=base/'cutoff-uncapped-20261116',
        out=base/'cutoff-threshold-controls-20261116', workers=12, budget=10,
        model='mistralai/mistral-small-2603', judge='google/gemini-2.5-flash', stage='prepare')
    run = Followup(args)
    run.prepare()
    meta = json.loads((args.base/'tokenizer-metadata.json').read_text())
    tokenizer = Tokenizer.from_file(meta['path'])
    rows = json.loads((args.out/'retrieval-results.json').read_text())
    for row in rows:
        prompt = run.builder.build(run.queries[row['query_id']]['transcript'], row['record_ids'], run.data)['prompt']
        row['prompt_tokens_predicted'] = len(tokenizer.encode(prompt, add_special_tokens=False).ids) + meta['system_tokens'] + 17
        row['prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
    atomic_json(args.out/'prompt-preflight-rows.json', rows)
    load_dotenv(ROOT/'.env')
    key = os.environ.get('ELM_API_KEY') or os.environ.get('OPENAI_API_KEY')
    request = urllib.request.Request('https://openrouter.ai/api/v1/credits', headers={'Authorization': 'Bearer '+key})
    with urllib.request.urlopen(request, timeout=30) as response:
        credits = json.load(response)['data']
    with urllib.request.urlopen('https://openrouter.ai/api/v1/models', timeout=30) as response:
        model = next(m for m in json.load(response)['data'] if m['id']==args.model)
    pricing = model['pricing']
    by_arm = []
    for arm in ARMS:
        rr = [r for r in rows if r['arm']==arm]
        by_arm.append(dict(arm=arm, mean_docs=statistics.mean(r['k'] for r in rr),
            max_docs=max(r['k'] for r in rr), macro_precision=statistics.mean(r['precision'] for r in rr),
            macro_recall=statistics.mean(r['recall'] for r in rr),
            mean_prompt=statistics.mean(r['prompt_tokens_predicted'] for r in rr),
            max_prompt=max(r['prompt_tokens_predicted'] for r in rr)))
    prompt_cost = sum(r['prompt_tokens_predicted'] for r in rows)*float(pricing['prompt'])
    result = dict(all_fit=all(r['prompt_tokens_predicted']+2048<=model['context_length'] for r in rows),
        settings=len(ARMS), queries=len(run.test), total_prompts=len(rows),
        config_sha256=sha(args.out/'config.json'), context_limit=model['context_length'],
        max_prompt_tokens=max(r['prompt_tokens_predicted'] for r in rows), tokenizer=meta,
        by_arm=by_arm, generator_pricing=pricing,
        remaining_credits=credits['total_credits']-credits['total_usage'],
        estimated_generation_usd_at_500_output_tokens=prompt_cost+len(rows)*500*float(pricing['completion']),
        generation_usd_at_output_cap=prompt_cost+len(rows)*2048*float(pricing['completion']))
    atomic_json(args.out/'context-preflight.json', result)
    assert result['all_fit'], 'Do not run generation: a prompt exceeds the model context limit'
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()
