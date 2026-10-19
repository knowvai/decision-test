"""Authorized synthetic-only diagnostics with final-service restoration.

Uses frozen launch/request/measurement implementations, never benchmark data.
Run on the evaluation host with its existing Python environment and -B.
"""
import asyncio
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path('/data/sdb/decision-model-eval')
BASE = ROOT / 'capacity-20261016'
OUT = ROOT / 'qwen3-diagnostic-20261017'
SOURCE = BASE / 'source-r2'
sys.dont_write_bytecode = True
sys.path[:0] = [str(SOURCE / 'evaluation/runner/scripts'),
               str(SOURCE / 'deploy/docker/decision-eval')]
import httpx
import decision_capacity_preflight as capacity
import public_decision_benchmark as benchmark
import launch_reviewed as launcher

MODEL = 'qwen3-30b-a3b-instruct-2507'
RESTORE = 'qwen3.8-27b-thinking'
ENV_FILE = ROOT / 'source-decision-v3-r3-20261012/.env'
BASE_URL = 'http://127.0.0.1:30889'


def command(args):
    return subprocess.check_output(args, text=True).strip()


def status(stage, **fields):
    value = {'at_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage, **fields}
    benchmark.write_json(OUT / 'status.json', value)
    print(json.dumps(value), flush=True)


def args(model, tokens=32768):
    return SimpleNamespace(root=ROOT, model=model, env_file=ENV_FILE, native_port=30888,
                           public_port=30889, seqs=32, tokens=tokens,
                           enforce_eager=False, laya_max_concurrent=None)


def owned_identity():
    owned = command(['docker', 'ps', '--filter', 'label=evaluation.decision.acceptance=true',
                     '--format', '{{.Names}}']).splitlines()
    assert set(owned) == {'decision-acceptance-model', 'decision-acceptance-proxy'}
    value = json.loads(command(['docker', 'inspect', '--format',
        '{"image":{{json .Image}},"path":{{json .Path}},"args":{{json .Args}},'
        '"pid":{{.State.Pid}},"status":{{json .State.Status}},'
        '"cpus":{{.HostConfig.NanoCpus}}}', 'decision-acceptance-model']))
    assert value['status'] == 'running'
    return value


def cgroup_stats():
    pid = owned_identity()['pid']
    rows = Path(f'/proc/{pid}/cgroup').read_text().splitlines()
    unified = next((row.split(':', 2)[2] for row in rows if row.startswith('0::')), None)
    if unified is not None:
        folder = Path('/sys/fs/cgroup') / unified.lstrip('/')
        return {'cpu_max': (folder / 'cpu.max').read_text().strip(),
                **{k: int(v) for k, v in (line.split() for line in (folder / 'cpu.stat').read_text().splitlines())}}
    cpu = next(row.split(':', 2)[2] for row in rows if 'cpu' in row.split(':')[1].split(','))
    folder = Path('/sys/fs/cgroup/cpu,cpuacct') / cpu.lstrip('/')
    return {'cpu_max': (folder / 'cpu.cfs_quota_us').read_text().strip(),
            'usage_ns': int((folder / 'cpuacct.usage').read_text()),
            **{k: int(v) for k, v in (line.split() for line in (folder / 'cpu.stat').read_text().splitlines())}}


async def ready(key):
    deadline = time.monotonic() + 2400
    async with httpx.AsyncClient(trust_env=False, timeout=5) as http:
        while time.monotonic() < deadline:
            state = command(['docker', 'inspect', '--format', '{{.State.Status}}', 'decision-acceptance-model'])
            assert state == 'running', 'Runtime stopped during startup'
            try:
                response = await http.get(BASE_URL + '/health', headers={'Authorization': 'Bearer ' + key})
                if response.status_code == 200:
                    bad = await http.get(BASE_URL + '/health')
                    assert bad.status_code == 401
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(3)
    raise TimeoutError('Startup deadline exceeded')


def engine_log(folder, key):
    output = subprocess.run(['docker', 'logs', 'decision-acceptance-model'],
                            text=True, capture_output=True)
    safe = (output.stdout + output.stderr).replace(key, '<redacted>')
    with (folder / 'engine.log').open('x') as stream:
        stream.write(safe)


METRICS_PARSER = '''import json,sys
from prometheus_client.parser import text_string_to_metric_families
out={}
prefixes=("vllm:request_", "vllm:e2e_request_", "vllm:time_to_first_token_",
          "vllm:time_per_output_token_", "vllm:num_preemptions", "vllm:prompt_tokens",
          "vllm:generation_tokens", "vllm:num_requests_", "vllm:kv_cache_usage_perc")
for family in text_string_to_metric_families(sys.stdin.read()):
 for sample in family.samples:
  if sample.name.startswith(prefixes) and not sample.name.endswith(("_bucket","_created")):
   out[sample.name]=out.get(sample.name,0)+sample.value
print(json.dumps(out))
'''


async def metrics(http, headers):
    response = await http.get(BASE_URL + '/metrics', headers=headers)
    response.raise_for_status()
    result = subprocess.run(['docker', 'exec', '-i', 'decision-acceptance-model',
                             'python3', '-c', METRICS_PARSER], input=response.text,
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def metric_delta(before, after):
    delta = {k: v - before.get(k, 0) for k, v in after.items()
             if k.endswith(('_sum', '_count', '_total'))}
    means = {k[:-4]: value / delta[k[:-4] + '_count'] for k, value in delta.items()
             if k.endswith('_sum') and delta.get(k[:-4] + '_count', 0) > 0}
    return {'counters': delta, 'means': means,
            'gauges_after': {k: v for k, v in after.items() if k not in delta}}


def make_jobs():
    query = '\u7cfb\u7edf\u5347\u7ea7\u524d\u9700\u8981\u505a\u54ea\u4e9b\u51c6\u5907\uff1f'
    related = '\u7ef4\u62a4\u8bb0\u5f55\u8981\u6c42\u5148\u5907\u4efd\u914d\u7f6e\u548c\u6570\u636e\uff0c\u68c0\u67e5\u517c\u5bb9\u6027\uff0c\u786e\u8ba4\u56de\u9000\u6b65\u9aa4\u540e\u518d\u6267\u884c\u5347\u7ea7\u3002'
    unrelated = '\u6d3b\u52a8\u8bb0\u5f55\u5305\u542b\u4f1a\u573a\u5e03\u7f6e\u548c\u5ea7\u4f4d\u5b89\u6392\uff0c\u5fd7\u613f\u8005\u6309\u7167\u65f6\u95f4\u8868\u8fce\u63a5\u6765\u8bbf\u8005\u5e76\u5206\u53d1\u8d44\u6599\u3002'
    jobs = []
    for i in range(60):
        size = (64, 128, 256, 512, 1024)[i % 5]
        seed = related if i % 2 == 0 else unrelated
        document = f'Synthetic record {i}. ' + seed * max(1, size // len(seed))
        state = {'query': query, 'document': document}
        jobs.append({'case_id': f'diagnostic-20261017:{i}', 'fragment_id': str(i),
                     'state': state, 'input_sha256': benchmark.digest(state),
                     'chars': len(query) + len(document)})
    return jobs


async def measure_condition(folder, key, jobs):
    served = launcher.MODELS[MODEL][2]
    transport = httpx.AsyncHTTPTransport(retries=0, limits=httpx.Limits(max_connections=20, max_keepalive_connections=20))
    async with httpx.AsyncClient(transport=transport, timeout=300, trust_env=False) as http:
        headers = {'Authorization': 'Bearer ' + key}

        async def infer(job):
            body = capacity.llm_body(served, job['state'])
            meta = {'calls': 1, 'request_sha256': benchmark.digest(body)}
            try:
                response = await http.post(BASE_URL + '/v1/chat/completions', json=body, headers=headers)
                meta.update(http_status=response.status_code,
                            response_sha256=hashlib.sha256(response.content).hexdigest())
                response.raise_for_status()
                return {**meta, **capacity.llm_result(response.json(), served)}
            except Exception as error:
                error.capacity_metadata = meta
                raise

        for n in (1, 5):
            for repeat in (1, 2):
                status('warming', condition=folder.name, concurrency=n, repeat=repeat)
                rows, block = await capacity.measure(jobs[:20], infer, n, 300)
                capacity.save_rows(folder / f'warmup-n{n}-r{repeat}.jsonl', rows)
                capacity.save(folder / f'warmup-n{n}-r{repeat}.json', capacity.block_summary(rows, block))
                assert all(r['status'] == 'valid' for r in rows), 'Synthetic warmup failed'
        reports = []
        for index, (repeat, n) in enumerate(((1, 1), (1, 5), (2, 5), (2, 1))):
            status('measuring', condition=folder.name, concurrency=n, repeat=repeat)
            before = await metrics(http, headers)
            cpu_before = cgroup_stats()
            rows, block = await capacity.measure(jobs, infer, n, 300)
            cpu_after = cgroup_stats()
            after = await metrics(http, headers)
            capacity.save_rows(folder / f'responses-n{n}-r{repeat}.jsonl', rows)
            report = {**capacity.block_summary(rows, block), 'repeat': repeat,
                      'metrics_before': before, 'metrics_after': after,
                      'server_metrics': metric_delta(before, after),
                      'cpu_before': cpu_before, 'cpu_after': cpu_after}
            capacity.save(folder / f'block-n{n}-r{repeat}.json', report)
            reports.append(report)
            assert all(r['status'] == 'valid' for r in rows), 'Synthetic measurement failed'
        capacity.save(folder / 'summary.json', reports)


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Diagnostic already started'
    os_mode = 0o077
    import os
    os.umask(os_mode)
    audit = json.loads((BASE / 'audit-final-20261016.json').read_text())
    capacity.check_artifacts(audit['artifact_sha256'])
    original = owned_identity()
    final = json.loads((BASE / 'run-002' / RESTORE / 'deployment.json').read_text())
    assert [original['path'], *original['args']] == final['command'] and original['image'] == final['image_id']
    jobs = make_jobs()
    key = launcher.read_key(ENV_FILE)
    with (ROOT / 'classification-20261015/classification.lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        capacity.save(OUT / 'protocol.json', {
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Owner-approved independent synthetic diagnosis; not benchmark rerun',
            'script_sha256': capacity.sha(Path(__file__)), 'original_service': original,
            'frozen_audit_sha256': capacity.sha(BASE / 'audit-final-20261016.json'),
            'source_sha256': {str(p): capacity.sha(p) for p in (
                Path(capacity.__file__), Path(benchmark.__file__), Path(launcher.__file__))},
            'model': MODEL, 'restore_model': RESTORE, 'synthetic_jobs': 60,
            'conditions': ['baseline-cpu8-tokens32768', 'cpu16-tokens32768', 'cpu8-tokens2048'],
            'levels': [1, 5], 'measured_order': [[1, 1], [1, 5], [2, 5], [2, 1]],
            'max_tokens': 64, 'temperature': 0, 'retries': 0, 'startup_timeout': 2400,
            'original_benchmark_inputs_used': False})
        capacity.save_rows(OUT / 'synthetic-inputs.jsonl', jobs)
        failure = None
        try:
            for name, cpus, tokens, restart in (
                ('baseline-cpu8-tokens32768', 8, 32768, True),
                ('cpu16-tokens32768', 16, 32768, False),
                ('cpu8-tokens2048', 8, 2048, True)):
                folder = OUT / name
                folder.mkdir()
                status('loading' if restart else 'cpu_quota_change', condition=name)
                if restart:
                    launcher.launch(args(MODEL, tokens))
                    asyncio.run(ready(key))
                else:
                    command(['docker', 'update', '--cpus', str(cpus), 'decision-acceptance-model'])
                capacity.save(folder / 'deployment.json', json.loads((ROOT / 'acceptance-20261015/evidence' / (MODEL + '.deployment.json')).read_text()))
                capacity.save(folder / 'live_identity.json', owned_identity())
                with (folder / 'gpu-dmon.log').open('x') as stream:
                    monitor = subprocess.Popen(['nvidia-smi', 'dmon', '-s', 'put', '-d', '1'],
                                               stdout=stream, stderr=subprocess.STDOUT, text=True)
                    try:
                        asyncio.run(measure_condition(folder, key, jobs))
                    finally:
                        monitor.terminate()
                        monitor.wait(timeout=10)
                        engine_log(folder, key)
                capacity.check_artifacts(audit['artifact_sha256'])
                status('condition_complete', condition=name)
        except BaseException as error:
            failure = {'error_type': type(error).__name__}
            capacity.save(OUT / 'failure.json', failure)
        finally:
            status('restoring', prior_failure=failure)
            launcher.launch(args(RESTORE))
            asyncio.run(ready(key))
            restored = owned_identity()
            assert [restored['path'], *restored['args']] == final['command']
            assert restored['image'] == original['image'] and restored['cpus'] == original['cpus']
            capacity.check_artifacts(audit['artifact_sha256'])
            capacity.save(OUT / 'restoration.json', {'at_utc': datetime.now(timezone.utc).isoformat(),
                'service': restored, 'health': 200, 'unauthorized': 401,
                'frozen_artifacts_unchanged': True, 'model_inference_for_restoration': 0})
            status('failed_restored' if failure else 'complete_restored', failure=failure)
        if failure:
            raise RuntimeError('Diagnostic failed; original service restored')


if __name__ == '__main__':
    main()
