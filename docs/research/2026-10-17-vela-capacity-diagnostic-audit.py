"""Offline-only verification and aggregation of the Vela synthetic diagnosis."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
OUT = Path('/data/sdb/decision-model-eval/vela-diagnostic-20261017')
spec = importlib.util.spec_from_file_location('vela_diagnostic', OUT / 'diagnostic.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def gpu_summary(path):
    samples = {}
    names = []
    for line in path.read_text().splitlines():
        if line.startswith('# gpu'):
            names = line.split()[1:]
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split()
        if len(fields) != len(names):
            continue
        values = {k: float(v) for k, v in zip(names, fields) if v != '-'}
        samples.setdefault(str(int(values['gpu'])), []).append(values)
    return {gpu: {'samples': len(group), 'includes_warmup_and_between_block_gaps': True,
                  'metrics': {k: {'mean': statistics.mean(v), 'min': min(v), 'max': max(v)}
                              for k in ('pwr', 'sm', 'mem', 'pclk', 'mclk')
                              if (v := [row[k] for row in group if k in row])}}
            for gpu, group in samples.items()}


def summary(folder, n, inputs):
    records, blocks = [], []
    for repeat in (1, 2):
        batch = rows(folder / f'responses-n{n}-r{repeat}.jsonl')
        block = d.read(folder / f'block-n{n}-r{repeat}.json')
        assert len(batch) == block['requests'] == 60
        assert {r['case_id'] for r in batch} == set(inputs)
        assert block['peak_inflight'] == n
        active = peak = 0
        for _, change in sorted([(r['started_seconds'], 1) for r in batch] +
                                [(r['ended_seconds'], -1) for r in batch]):
            active += change
            peak = max(peak, active)
        assert active == 0 and peak == n
        for record in batch:
            job = inputs[record['case_id']]
            assert record['status'] == 'valid' and record['calls'] == 1 and record['http_status'] == 200
            assert record['input_sha256'] == d.benchmark.digest(job['state'])
            payload = {'model': d.launcher.MODELS[d.MODEL][2], 'state': job['state'],
                       'questions': d.capacity.QUESTIONS}
            _, body = d.native_choice_request('worker', payload)
            body['options'] = {'return_meta': True}
            assert record['request_sha256'] == d.benchmark.digest(body)
        records.extend(batch)
        blocks.append(block)
    counter, cpu = Counter(), Counter()
    for b in blocks:
        counter.update(b['server_metrics']['counters'])
        cpu.update({k: value - b['cpu_before'][k] for k, value in b['cpu_after'].items()
                    if isinstance(value, (float, int))})
    means = {k[:-4]: value / counter[k[:-4] + '_count'] for k, value in counter.items()
             if k.endswith('_sum') and counter[k[:-4] + '_count'] > 0}
    wall = sum(b['wall_seconds'] for b in blocks)
    assert counter['vllm_srun_request_duration_seconds_count'] == len(records)
    return {'requests': len(records), 'valid': len(records), 'failed': 0,
        'latency': d.benchmark.timing([r['elapsed_ms'] for r in records]),
        'queue_meta': d.benchmark.timing([r['meta']['queue_ms'] for r in records]),
        'compute_meta': d.benchmark.timing([r['meta']['compute_ms'] for r in records]),
        'rps': len(records) / wall, 'wall_seconds': wall, 'peak_inflight': n,
        'http_status': dict(Counter(str(r['http_status']) for r in records)),
        'input_tokens': dict(Counter(str(r['usage']['input_tokens']) for r in records)),
        'input_tokens_mean': statistics.mean(r['usage']['input_tokens'] for r in records),
        'identity': {k: sorted({r['meta'][k] for r in records})
                     for k in ('model_id', 'revision', 'dtype', 'numerics', 'engine', 'device', 'accelerator')},
        'server_counter_deltas': dict(counter), 'server_means': means,
        'cpu_deltas': dict(cpu), 'average_cpu_cores': cpu.get('usage_usec', 0) / 1e6 / wall,
        'rounds': [{k: b[k] for k in ('repeat', 'successful_latency_ms',
                                     'effective_requests_per_second', 'server_metrics')} for b in blocks]}


def main():
    assert d.read(OUT / 'status.json')['stage'] == 'complete_restored'
    protocol = d.read(OUT / 'protocol.json')
    assert d.capacity.sha(OUT / 'diagnostic.py') == protocol['script_sha256']
    assert d.capacity.sha(d.PREVIOUS / 'diagnostic-audit.json') == protocol['prior_audit_sha256']
    counts = d.preserved()
    result = {'at_utc': datetime.now(timezone.utc).isoformat(), 'protocol': protocol,
        'preserved_capacity_files': counts[0], 'preserved_qwen_diagnostic_files': counts[1],
        'restoration': d.read(OUT / 'restoration.json'),
        'completion_check': d.read(OUT / 'completion-check.json'), 'pools': {}}
    result['metric_semantics'] = {
        'queue_meta': 'Response meta.queue_ms is time before submission, NOT scheduler queue wait.',
        'compute_meta': 'Response meta.compute_ms is submission-to-result time, including scheduler queue.',
        'prometheus_queue': 'vllm_srun_queue_duration_seconds observes job enqueue to first scheduled batch.',
        'prometheus_forward': 'Scheduler batch wall time including readout; not necessarily one backbone call.',
        'gpu': 'One-second samples over each pool, not a kernel utilization or saturation proof.'}
    for name in protocol['pools']:
        folder = OUT / name
        jobs = rows(folder / 'inputs.jsonl')
        inputs = {j['case_id']: j for j in jobs}
        assert len(jobs) == len(inputs) == 60
        unique = len({j['input_sha256'] for j in jobs})
        assert unique == (1 if name == 'fixed-shared' else 60)
        for n in (1, 5):
            for repeat in (1, 2):
                warm = rows(folder / f'warmup-n{n}-r{repeat}.jsonl')
                assert len(warm) == 20 and all(r['status'] == 'valid' and r['calls'] == 1 for r in warm)
                assert {r['case_id'] for r in warm} == {j['case_id'] for j in jobs[:20]}
        levels = {str(n): summary(folder, n, inputs) for n in (1, 5)}
        for level in levels.values():
            level['mean_ratio_to_n1'] = level['latency']['mean_ms'] / levels['1']['latency']['mean_ms']
            level['rps_ratio_to_n1'] = level['rps'] / levels['1']['rps']
        result['pools'][name] = {'unique_input_hashes': unique, 'warmup_valid': 80, 'levels': levels,
                                'gpu': gpu_summary(folder / 'gpu-dmon.log')}
    inspection = d.read(OUT / 'runtime-source.json')
    for name, text in inspection['selected_source'].items():
        assert hashlib.sha256(text.encode()).hexdigest() == inspection['source_sha256'][name]
    result['dispatch_without_inference'] = inspection['dispatch_without_inference']
    result['runtime_version'] = inspection['version']
    result['runtime_source_sha256'] = inspection['source_sha256']
    result['live_identity'] = d.read(OUT / 'live-identity.json')
    result['measured_requests'], result['warmup_requests'] = 720, 240
    result['diagnostic_artifact_sha256'] = {str(p.relative_to(OUT)): d.capacity.sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'diagnostic-audit.json'}
    with (OUT / 'diagnostic-audit.json').open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({'audit_sha256': d.capacity.sha(OUT / 'diagnostic-audit.json'),
        'measurements': 720, 'warmup': 240, 'capacity_files_unchanged': counts[0],
        'qwen_diagnostic_files_unchanged': counts[1]}))


if __name__ == '__main__':
    main()
