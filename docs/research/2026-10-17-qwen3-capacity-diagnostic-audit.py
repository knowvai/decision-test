"""Offline-only audit of the authorized synthetic Qwen3 diagnostic."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path('/data/sdb/decision-model-eval')
OUT = ROOT / 'qwen3-diagnostic-20261017'
BASE = ROOT / 'capacity-20261016'
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE / 'source-r2/evaluation/runner/scripts'))
import decision_capacity_preflight as capacity
import public_decision_benchmark as benchmark


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def delta(before, after):
    return {k: after[k] - before[k] for k in before
            if isinstance(before[k], (float, int))}


def summarize(folder, n, expected):
    selected = []
    records = []
    for repeat in (1, 2):
        block = read(folder / f'block-n{n}-r{repeat}.json')
        batch = rows(folder / f'responses-n{n}-r{repeat}.jsonl')
        assert len(batch) == 60 and block['requests'] == 60
        assert block['peak_inflight'] == n
        assert {r['case_id']: r['input_sha256'] for r in batch} == expected
        assert all(r['calls'] == 1 and r['status'] == 'valid' and
                   r['http_status'] == 200 and r['finish_reason'] == 'stop' and
                   r['reasoning_chars'] == 0 for r in batch)
        for r in batch:
            assert len(r['request_sha256']) == 64
        events = sorted([(r['started_seconds'], 1) for r in batch] +
                        [(r['ended_seconds'], -1) for r in batch])
        live = peak = 0
        for _, change in events:
            live += change
            peak = max(peak, live)
        assert live == 0 and peak == n
        assert block['server_metrics']['counters']['vllm:e2e_request_latency_seconds_count'] == 60
        selected.append(block)
        records.extend(batch)
    counters = Counter()
    cpu = Counter()
    for block in selected:
        counters.update(block['server_metrics']['counters'])
        cpu.update(delta(block['cpu_before'], block['cpu_after']))
    wall = sum(b['wall_seconds'] for b in selected)
    means = {k[:-4]: value / counters[k[:-4] + '_count'] for k, value in counters.items()
             if k.endswith('_sum') and counters[k[:-4] + '_count'] > 0}
    return {
        'requests': len(records), 'valid': len(records), 'failed': 0,
        'http_status': dict(Counter(str(r['http_status']) for r in records)),
        'latency': benchmark.timing([r['elapsed_ms'] for r in records]),
        'rps': len(records) / wall, 'wall_seconds': wall, 'peak_inflight': n,
        'server_counter_deltas': dict(counters), 'server_means': means,
        'cpu_quota': selected[0]['cpu_before']['cpu_max'], 'cpu_deltas': dict(cpu),
        'average_cpu_cores': cpu.get('usage_usec', 0) / 1e6 / wall,
        'prompt_tokens_mean': statistics.mean(r['usage']['prompt_tokens'] for r in records),
        'completion_tokens_mean': statistics.mean(r['usage']['completion_tokens'] for r in records),
        'rounds': [{k: b[k] for k in ('repeat', 'wall_seconds', 'successful_latency_ms',
                                     'effective_requests_per_second', 'server_metrics')}
                   for b in selected],
    }


def gpu_summary(path):
    data = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith('#'):
            continue
        try:
            values = [float(x) if x != '-' else None for x in fields]
        except ValueError:
            continue
        data.setdefault(str(int(values[0])), []).append(values)
    names = ('gpu', 'power_w', 'gtemp_c', 'mtemp_c', 'sm_pct', 'mem_pct',
             'enc_pct', 'dec_pct', 'jpg_pct', 'ofa_pct', 'pcie_rx_mb_s', 'pcie_tx_mb_s')
    return {gpu: {'samples': len(samples), 'whole_condition_including_warmup': True,
                  'columns': {name: {'mean': statistics.mean(v), 'max': max(v)}
                              for i, name in enumerate(names[1:], 1)
                              if (v := [row[i] for row in samples
                                        if len(row) == len(names) and row[i] is not None])}}
            for gpu, samples in data.items()}


def engine_evidence(path):
    markers = ('Custom allreduce is disabled', "Using ['PYNCCL']", 'Using default MoE config',
               'Using TritonExperts MoE backend', 'GPU KV cache size:',
               'Graph capturing finished', 'Model loading took')
    return [{'line': number, 'text': line} for number, line in enumerate(path.read_text().splitlines(), 1)
            if any(marker in line for marker in markers)]


def main():
    assert read(OUT / 'status.json')['stage'] == 'complete_restored'
    protocol = read(OUT / 'protocol.json')
    assert sha(OUT / 'diagnostic.py') == protocol['script_sha256']
    audit = read(BASE / 'audit-final-20261016.json')
    assert sha(BASE / 'audit-final-20261016.json') == protocol['frozen_audit_sha256']
    capacity.check_artifacts(audit['artifact_sha256'])
    jobs = rows(OUT / 'synthetic-inputs.jsonl')
    assert len(jobs) == len({j['input_sha256'] for j in jobs}) == 60
    assert all(benchmark.digest(j['state']) == j['input_sha256'] for j in jobs)
    expected = {j['case_id']: j['input_sha256'] for j in jobs}
    states = {j['case_id']: j['state'] for j in jobs}
    result = {'at_utc': datetime.now(timezone.utc).isoformat(), 'scope': protocol['scope'],
              'protocol': protocol, 'conditions': {}, 'restoration': read(OUT / 'restoration.json'),
              'completion_check': read(OUT / 'completion-check.json'),
              'frozen_artifacts_verified': len(audit['artifact_sha256'])}
    identities = []
    request_hashes = {}
    for name in protocol['conditions']:
        folder = OUT / name
        identity = read(folder / 'live_identity.json')
        identities.append(identity)
        served = identity['args'][identity['args'].index('--served-model-name') + 1]
        warmups = []
        for n in (1, 5):
            for repeat in (1, 2):
                batch = rows(folder / f'warmup-n{n}-r{repeat}.jsonl')
                assert len(batch) == 20
                assert {r['case_id']: r['input_sha256'] for r in batch} == {
                    j['case_id']: j['input_sha256'] for j in jobs[:20]}
                warmups.extend(batch)
        assert len(warmups) == 80 and all(r['status'] == 'valid' and r['calls'] == 1 for r in warmups)
        for path in folder.glob('responses-*.jsonl'):
            for r in rows(path):
                assert r['request_sha256'] == benchmark.digest(capacity.llm_body(served, states[r['case_id']]))
                previous = request_hashes.setdefault(r['case_id'], r['request_sha256'])
                assert previous == r['request_sha256']
        levels = {str(n): summarize(folder, n, expected) for n in (1, 5)}
        for level in levels.values():
            level['mean_ratio_to_n1'] = level['latency']['mean_ms'] / levels['1']['latency']['mean_ms']
            level['rps_ratio_to_n1'] = level['rps'] / levels['1']['rps']
        result['conditions'][name] = {'identity': identity, 'warmup_valid': 80, 'levels': levels,
                                      'gpu': gpu_summary(folder / 'gpu-dmon.log'),
                                      'engine_evidence': engine_evidence(folder / 'engine.log')}
    assert identities[0]['cpus'] == identities[2]['cpus'] == 8_000_000_000
    assert identities[1]['cpus'] == 16_000_000_000
    assert identities[0]['args'] == identities[1]['args']
    changed = list(identities[0]['args'])
    changed[changed.index('--max-num-batched-tokens') + 1] = '2048'
    assert changed == identities[2]['args']
    assert len({i['image'] for i in identities}) == 1
    result['measured_requests'] = 720
    result['warmup_requests'] = 240
    result['hardware'] = read(OUT / 'hardware.json')
    clock_groups = {}
    for sample in rows(OUT / 'clock-samples.jsonl'):
        if sample['status']['stage'] != 'measuring':
            continue
        for raw in sample['gpu']:
            gpu = {k.strip(): v.strip() for k, v in raw.items()}
            key = f"n{sample['status']['concurrency']}-gpu{gpu['index']}"
            group = clock_groups.setdefault(key, {'sm_mhz': Counter(), 'events': Counter()})
            group['sm_mhz'][gpu['clocks.current.sm [MHz]']] += 1
            group['events'][gpu['clocks_event_reasons.active']] += 1
    result['clock_samples_tokens2048_only'] = clock_groups
    result['diagnostic_artifact_sha256'] = {str(p.relative_to(OUT)): sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'diagnostic-audit.json'}
    with (OUT / 'diagnostic-audit.json').open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({'audit': str(OUT / 'diagnostic-audit.json'),
                      'sha256': sha(OUT / 'diagnostic-audit.json'),
                      'measured_requests': 720, 'warmup_requests': 240,
                      'frozen_artifacts_verified': result['frozen_artifacts_verified']}))


if __name__ == '__main__':
    main()
