"""Synthetic-only Vela diagnosis; reuse audited transport/lifecycle helpers."""
import asyncio
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

PREVIOUS = Path('/data/sdb/decision-model-eval/qwen3-diagnostic-20261017')
spec = importlib.util.spec_from_file_location('qwen_diagnostic_helpers', PREVIOUS / 'diagnostic.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
shared.OUT = shared.ROOT / 'vela-diagnostic-20261017'
OUT = shared.OUT
capacity, benchmark, launcher = shared.capacity, shared.benchmark, shared.launcher
from deployment_probe import native_choice_request, native_choice_result

MODEL = 'vela-2.0-0.8b'

INSPECT = r'''
import hashlib, importlib.metadata, json
from pathlib import Path
from types import SimpleNamespace
from vllm_srun.families.vela2.members import DecoderMember
root = Path('/usr/local/lib/python3.12/site-packages/vllm_srun')
selected = ('families/vela2/members.py', 'families/vela2/family.py',
            'scheduler/batching.py', 'scheduler/scheduler.py',
            'engines/native/engine.py', 'supervision/metrics.py', 'runtime.py')
sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(root.rglob('*.py'))}
results = {}
for name, same, packed in (('distinct_packed', False, True),
                           ('same_packed', True, True),
                           ('distinct_rows', False, False)):
    calls = []
    def tree(batch):
        calls.append({'prefixes': len(batch.prefixes), 'blocks': len(batch.blocks),
                      'layout': batch.layout})
        return SimpleNamespace(hidden=None)
    member = object.__new__(DecoderMember)
    member.engine_model = SimpleNamespace(tree=tree)
    member._read = lambda blocks, hidden: [None] * len(blocks)
    items = [SimpleNamespace(prefix=[1, 2, 3 if same else i + 3], ids=[1, 2, 3, 4],
             window=False, blocks=[SimpleNamespace(read=True, ids=[4])]) for i in range(5)]
    answer = member.run(items, packed=packed)
    assert len(answer) == 5
    results[name] = calls
assert len(results['distinct_packed']) == 5
assert len(results['same_packed']) == 1
assert len(results['distinct_rows']) == 1 and results['distinct_rows'][0]['prefixes'] == 5
print(json.dumps({'version': importlib.metadata.version('vllm-srun'),
    'source_sha256': sources, 'selected_source': {name: (root/name).read_text() for name in selected},
    'dispatch_without_inference': results, 'model_loads': 0, 'gpu_forwards': 0}))
'''

PARSER = '''import json,sys
from prometheus_client.parser import text_string_to_metric_families
out={}
for family in text_string_to_metric_families(sys.stdin.read()):
 for sample in family.samples:
  if sample.name.startswith('vllm_srun_') and not sample.name.endswith(('_bucket','_created')):
   out[sample.name]=out.get(sample.name,0)+sample.value
print(json.dumps(out))
'''


def read(path):
    return json.loads(path.read_text())


def pools():
    mixed = shared.make_jobs()
    for j in mixed:
        j['case_id'] = j['case_id'].replace('diagnostic-20261017', 'vela-mixed-20261017')
    fixed, identical = [], []
    base = mixed[2]['state']
    for name, target, same in (('fixed-distinct', fixed, False), ('fixed-shared', identical, True)):
        for i in range(60):
            state = {**base, 'document': base['document'] + f' Batch case {10 if same else i+10}.'}
            target.append({'case_id': f'vela-{name}-20261017:{i}', 'fragment_id': str(i),
                'state': state, 'input_sha256': benchmark.digest(state),
                'chars': len(state['query']) + len(state['document'])})
    return {'mixed-distinct': mixed, 'fixed-distinct': fixed, 'fixed-shared': identical}


def preserved():
    original = read(shared.BASE / 'audit-final-20261016.json')
    capacity.check_artifacts(original['artifact_sha256'])
    prior = read(PREVIOUS / 'diagnostic-audit.json')
    capacity.check_artifacts({str(PREVIOUS / k): v for k, v in prior['diagnostic_artifact_sha256'].items()})
    return len(original['artifact_sha256']), len(prior['diagnostic_artifact_sha256'])


async def metrics(http, headers):
    response = await http.get(shared.BASE_URL + '/metrics', headers=headers)
    response.raise_for_status()
    parsed = subprocess.run(['docker', 'exec', '-i', 'decision-acceptance-model',
        'python', '-B', '-c', PARSER], input=response.text, text=True, capture_output=True,
        check=True, timeout=60)
    return json.loads(parsed.stdout)


async def measure_pool(folder, jobs, key):
    family, _, served = launcher.MODELS[MODEL]
    transport = shared.httpx.AsyncHTTPTransport(retries=0,
        limits=shared.httpx.Limits(max_connections=20, max_keepalive_connections=20))
    async with shared.httpx.AsyncClient(transport=transport, timeout=300, trust_env=False) as http:
        headers = {'Authorization': 'Bearer ' + key}

        async def infer(job):
            payload = {'model': served, 'state': job['state'], 'questions': capacity.QUESTIONS}
            path, body = native_choice_request(family, payload)
            body['options'] = {'return_meta': True}
            meta = {'calls': 1, 'request_sha256': benchmark.digest(body)}
            try:
                response = await http.post(shared.BASE_URL + path, json=body, headers=headers)
                meta.update(http_status=response.status_code,
                    response_sha256=hashlib.sha256(response.content).hexdigest())
                response.raise_for_status()
                raw = response.json()
                choice, probabilities = native_choice_result(family, payload, raw)
                assert raw['meta']['device'].startswith('cuda')
                return {**meta, 'prediction': choice, 'probabilities': probabilities,
                        'usage': raw.get('usage'), 'meta': raw['meta']}
            except Exception as error:
                error.capacity_metadata = meta
                raise

        for n in (1, 5):
            for repeat in (1, 2):
                shared.status('warming', pool=folder.name, concurrency=n, repeat=repeat)
                records, block = await capacity.measure(jobs[:20], infer, n, 300)
                capacity.save_rows(folder / f'warmup-n{n}-r{repeat}.jsonl', records)
                capacity.save(folder / f'warmup-n{n}-r{repeat}.json', capacity.block_summary(records, block))
                assert all(r['status'] == 'valid' for r in records), 'Warmup failure'
        reports = []
        for repeat, n in ((1, 1), (1, 5), (2, 5), (2, 1)):
            shared.status('measuring', pool=folder.name, concurrency=n, repeat=repeat)
            before, cpu_before = await metrics(http, headers), shared.cgroup_stats()
            records, block = await capacity.measure(jobs, infer, n, 300)
            cpu_after, after = shared.cgroup_stats(), await metrics(http, headers)
            capacity.save_rows(folder / f'responses-n{n}-r{repeat}.jsonl', records)
            report = {**capacity.block_summary(records, block), 'repeat': repeat,
                'metrics_before': before, 'metrics_after': after,
                'server_metrics': shared.metric_delta(before, after),
                'cpu_before': cpu_before, 'cpu_after': cpu_after}
            capacity.save(folder / f'block-n{n}-r{repeat}.json', report)
            reports.append(report)
            assert all(r['status'] == 'valid' for r in records), 'Measurement failure'
        capacity.save(folder / 'summary.json', reports)


def main():
    os.umask(0o077)
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Diagnostic already started'
    key = launcher.read_key(shared.ENV_FILE)
    with (shared.ROOT / 'classification-20261015/classification.lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        counts = preserved()
        original = shared.owned_identity()
        final = read(shared.BASE / 'run-002' / shared.RESTORE / 'deployment.json')
        assert [original['path'], *original['args']] == final['command']
        assert original['image'] == final['image_id']
        frozen = read(shared.BASE / 'run-001' / MODEL / 'deployment.json')
        datasets = pools()
        capacity.save(OUT / 'protocol.json', {'at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Owner-approved independent synthetic diagnosis; no benchmark replay or runtime patch',
            'model': MODEL, 'image': frozen['image_id'], 'restore_model': shared.RESTORE,
            'original_service': original, 'script_sha256': capacity.sha(Path(__file__)),
            'reused_helper_sha256': capacity.sha(PREVIOUS / 'diagnostic.py'),
            'preserved_capacity_files': counts[0], 'preserved_qwen_diagnostic_files': counts[1],
            'prior_audit_sha256': capacity.sha(PREVIOUS / 'diagnostic-audit.json'),
            'pools': list(datasets), 'requests_per_block': 60, 'warmup_per_block': 20,
            'warmup_order': [[r,n] for n in (1,5) for r in (1,2)],
            'measurement_order': [[1,1],[1,5],[2,5],[2,1]],
            'request_metadata_only_option': {'return_meta': True}, 'retries': 0})
        for name, jobs in datasets.items():
            (OUT / name).mkdir()
            capacity.save_rows(OUT / name / 'inputs.jsonl', jobs)
        inspection = subprocess.run(['docker', 'run', '--rm', '-i', '--name', 'decision-vela-source-inspect',
            '--label', 'evaluation.decision.acceptance=true', '--network', 'none', '--read-only',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--entrypoint', 'python',
            frozen['image_id'], '-B', '-'], input=INSPECT, text=True, capture_output=True,
            check=True, timeout=120)
        capacity.save(OUT / 'runtime-source.json', json.loads(inspection.stdout))
        failure = None
        try:
            shared.status('loading', model=MODEL)
            launcher.launch(shared.args(MODEL))
            asyncio.run(shared.ready(key))
            live = shared.owned_identity()
            assert live['image'] == frozen['image_id']
            assert [live['path'], *live['args']] == frozen['command']
            capacity.save(OUT / 'live-identity.json', live)
            for name, jobs in datasets.items():
                folder = OUT / name
                with (folder / 'gpu-dmon.log').open('x') as stream:
                    monitor = subprocess.Popen(['nvidia-smi', 'dmon', '-s', 'puct', '-d', '1'],
                        stdout=stream, stderr=subprocess.STDOUT, text=True)
                    try:
                        asyncio.run(measure_pool(folder, jobs, key))
                    finally:
                        monitor.terminate()
                        monitor.wait(timeout=10)
                preserved()
            shared.engine_log(OUT, key)
        except BaseException as error:
            failure = {'type': type(error).__name__, 'message': str(error).replace(key, '<redacted>')[:1000]}
            capacity.save(OUT / 'failure.json', failure)
            if not (OUT / 'engine.log').exists():
                shared.engine_log(OUT, key)
        finally:
            shared.status('restoring', prior_failure=failure)
            launcher.launch(shared.args(shared.RESTORE))
            asyncio.run(shared.ready(key))
            restored = shared.owned_identity()
            assert [restored['path'], *restored['args']] == final['command']
            assert restored['image'] == original['image'] and restored['cpus'] == original['cpus']
            counts = preserved()
            capacity.save(OUT / 'restoration.json', {'at_utc': datetime.now(timezone.utc).isoformat(),
                'service': restored, 'health': 200, 'unauthorized': 401,
                'preserved_capacity_files': counts[0], 'preserved_qwen_diagnostic_files': counts[1],
                'inference_requests_for_restoration': 0})
            shared.status('failed_restored' if failure else 'complete_restored', failure=failure)
        if failure:
            raise RuntimeError('Vela diagnostic failed; original service restored')


if __name__ == '__main__':
    main()
