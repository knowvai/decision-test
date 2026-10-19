"""Offline, append-only audit of the completed capacity runs; never calls inference.

Run locally with --collect to stream this file to the existing test interpreter.
The remote and local JSON evidence files are both created exclusively.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


BASE = Path('/data/sdb/decision-model-eval/capacity-20261016')
ROOT = BASE.parent
OUTPUT = 'audit-final-20261016.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    return subprocess.check_output(args, text=True).strip()


def inventory(folder):
    return {str(p): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()
            and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def distribution(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    position = (len(values) - 1) * .95
    low, high = math.floor(position), math.ceil(position)
    return {'n': len(values), 'sum': sum(values), 'min': min(values), 'max': max(values),
            'mean': statistics.mean(values),
            'p95': values[low] + (values[high] - values[low]) * (position - low)}


def run_audit(script_sha):
    sys.dont_write_bytecode = True
    source = BASE / 'source-r2'
    service = source / 'deploy/docker/decision-eval'
    scripts = source / 'evaluation/runner/scripts'
    sys.path[:0] = [str(service), str(scripts)]
    import decision_capacity_preflight as capacity
    import public_decision_benchmark as benchmark
    from deployment_probe import native_choice_request
    from launch_reviewed import MODELS, CLASSIFICATION_PROFILES

    state = read(BASE / 'run-002/status.json')
    assert state['stage'] == 'complete' and state['models'] == state['completed']
    unit = dict(line.split('=', 1) for line in command([
        'systemctl', 'show', 'decision-capacity-20261016-r2b.service', '-p', 'MainPID',
        '-p', 'ExecMainStatus', '-p', 'ActiveState', '-p', 'SubState']).splitlines())
    assert unit == {'MainPID': '0', 'ExecMainStatus': '0',
                    'ActiveState': 'inactive', 'SubState': 'dead'}
    locks = command(['lslocks', '-n', '-o', 'PID,PATH'])
    assert '/classification-20261015/classification.lock' not in locks
    assert not (BASE / OUTPUT).exists(), 'Audit evidence already exists; do not overwrite'
    before = {**inventory(BASE / 'run-001'), **inventory(BASE / 'run-002')}
    p1, p2 = [read(BASE / run / 'protocol.json') for run in ('run-001', 'run-002')]
    original, prep = [read(BASE / name) for name in ('preparation.json', 'preparation-r2.json')]
    correction = read(BASE / 'preparation-r2-startup-correction.json')
    old_snapshot = {str(BASE / p): h for p, h in original['source_sha256'].items()}
    old_inventory = inventory(BASE / 'source')
    assert old_snapshot.keys() <= old_inventory.keys()
    original_sibling_baselines = {
        p: prep['source_sha256'][str(source / Path(p).relative_to(BASE / 'source'))]
        for p in old_inventory.keys() - old_snapshot.keys()}
    assert old_inventory == {**old_snapshot, **original_sibling_baselines}
    assert inventory(source) == prep['source_sha256']
    for manifest in (old_snapshot, prep['source_sha256'], prep['preserved_artifact_sha256'],
                     p1['source_sha256'], p1['preserved_artifact_sha256'],
                     p2['source_sha256'], p2['preserved_artifact_sha256']):
        capacity.check_artifacts(manifest)
    assert sha(BASE / 'runner.log') == prep['old_runner_log_sha256']
    assert sha(BASE / 'preparation-r2.json') == correction['preparation_sha256']
    assert sha(BASE / 'runner-r2.log') == correction['failed_start_log_sha256']
    assert correction['prior_start_inference_calls'] == 0
    assert correction['prior_start_created_output'] is False
    assert read(BASE / 'run-002/preservation-final.json') == {
        'unchanged': True, 'artifacts': len(p2['preserved_artifact_sha256'])}
    assert set(inventory(BASE / 'run-001')) <= p2['preserved_artifact_sha256'].keys()
    done = read(BASE / 'run-001/status.json')['completed']
    assert done == ['d3-edge', 'vela-2.0-0.3b', 'vela-2.0-0.8b']
    link, _ = capacity.capacity_predecessor(BASE / 'run-001', BASE / 'run-002', p2['models'], p2)
    assert link == p2['predecessor'] == prep['predecessor']
    assert p1['models'] == done + p2['models'] and len(p1['models']) == 16
    assert len({CLASSIFICATION_PROFILES.get(m, {}).get('base_model', m) for m in p1['models']}) == 15
    assert p2['deployment_overrides'] == {'laya-multilingual': {
        'http_admission_limit': 32, 'gpu_execution_unchanged': True}}
    baseline = functions(ROOT / 'classification-20261015/source-r8/evaluation/runner/scripts/public_decision_benchmark.py')
    assert len(baseline) == 39 and baseline == functions(scripts / 'public_decision_benchmark.py')
    audit_paths = [ROOT / 'classification-20261015' / name for name in (
        'audit-20261015/summary.json', 'qwen38-supplement-20261016/classification-audit.json',
        'qwen38-thinking-20261016/classification-audit.json')]
    audit_hashes = ['4accc5327368197a99b9aa04b9c48751026b3ecfa8ad1be9b20c4b21e7e60729',
                    '5156b7ffe6d406d5a8a34f8dcbdfbadd5e716ad8f4485274f444edf89a14dbc7',
                    'dbb8cc70448bfb6372e1f513d35947324bd7e700d6957d333d74d9ee7230e8cf']
    assert [sha(p) for p in audit_paths] == audit_hashes
    groups = [read(p)['artifact_sha256'] for p in audit_paths]
    union = {p: h for group in groups for p, h in group.items()}
    assert [len(g) for g in groups] == [98, 45, 66] and len(union) == 200
    capacity.check_artifacts(union)

    data_dir = BASE / 'dataset-final'
    data = read(data_dir / 'protocol.json')
    assert sha(data_dir / 'protocol.json') == p1['dataset_protocol_sha256'] == p2['dataset_protocol_sha256'] == prep['dataset_protocol_sha256']
    capacity.check_artifacts({str(data_dir / name): h for name, h in data['files'].items()})
    jobs = benchmark.read_jsonl(data_dir / 'inputs.private.jsonl')
    pools = benchmark.read_jsonl(data_dir / 'engineering-pools.private.jsonl')
    validation = benchmark.read_jsonl(data_dir / 'validation-pools.private.jsonl')
    rebuilt, excluded = capacity.workload(pools, 2000)
    assert rebuilt == jobs and excluded == read(data_dir / 'excluded.json')
    assert len(jobs) == 199 and len(excluded) == 27 and len(pools) == 12 and len(validation) == 20
    assert all(benchmark.digest(j['state']) == j['input_sha256'] and j['chars'] <= 2000 for j in jobs)
    assert not set(data['engineering_ids']) & set(data['validation_ids'])
    warm_ordered = sorted(jobs, key=lambda r: r['chars'])
    warm_jobs = [warm_ordered[i * (len(warm_ordered) - 1) // 19] for i in range(20)]
    expected_measure = [(repeat, n) for repeat, order in enumerate(capacity.ORDERS, 1) for n in order]
    expected_warm = [(repeat, n) for n in capacity.LEVELS for repeat in (1, 2)]
    log_states = {}
    for run, filename in [('run-001', 'runner.log'), ('run-002', 'runner-r2b.log')]:
        events = []
        for line in (BASE / filename).read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and 'stage' in event:
                events.append(event)
        log_states[run] = events

    def audit_block(rows, jobs_for_block, saved, family, served, profile):
        n = saved['concurrency']
        assert len(rows) == len(jobs_for_block) == saved['requests']
        assert saved['peak_inflight'] == n
        events = []
        for index, (row, job) in enumerate(zip(rows, jobs_for_block)):
            for key in ('case_id', 'fragment_id', 'input_sha256', 'chars'):
                assert row[key] == job[key]
            if family == 'llm':
                path, body = '/v1/chat/completions', capacity.llm_body(served, job['state'], profile)
                assert body['temperature'] == 0 and body['max_tokens'] == (4096 if profile else 64)
                assert body['response_format']['json_schema']['strict'] is True
            else:
                path, body = native_choice_request(family, {'model': served, 'state': job['state'],
                                                            'questions': capacity.QUESTIONS})
            assert row['request_sha256'] == benchmark.digest(body) and row['path'] == path
            assert row['calls'] == 1 and row['sequence'] == index
            assert row['status'] in ('valid', 'failed')
            assert 1 <= row['inflight_at_start'] <= n and row['elapsed_ms'] > 0
            assert row['started_seconds'] < row['ended_seconds'] <= saved['wall_seconds']
            assert abs(row['elapsed_ms'] - (row['ended_seconds'] - row['started_seconds']) * 1000) < 1
            events.extend([(row['started_seconds'], 1), (row['ended_seconds'], -1)])
            if row['status'] == 'valid':
                assert row['http_status'] == 200 and row['prediction'] in ('relevant', 'irrelevant')
                assert len(row['response_sha256']) == 64
                if family == 'llm':
                    assert row['finish_reason'] == 'stop'
                    assert bool(row['reasoning_chars']) == bool(profile)
                    usage = row['usage']
                    assert 0 < usage['completion_tokens'] <= body['max_tokens']
                    assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
                    reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
                    assert 0 <= reasoning <= usage['completion_tokens']
                    if profile:
                        assert reasoning > 0 and len(row['reasoning_sha256']) == 64
                    else:
                        assert reasoning == 0 and row['reasoning_sha256'] == benchmark.digest('')
                else:
                    assert set(row['probabilities']) == {'relevant', 'irrelevant'}
                    assert all(math.isfinite(v) and 0 <= v <= 1 for v in row['probabilities'].values())
                    for k, v in (row.get('usage') or {}).items():
                        if k in ('truncated', 'truncated_tokens', 'state_tokens_dropped', 'truncated_questions'):
                            assert not v
        active = peak = 0
        for _, change in sorted(events):
            active += change
            peak = max(peak, active)
        assert active == 0 and peak == n and max(r['inflight_at_start'] for r in rows) == n
        basic = {k: saved[k] for k in ('wall_seconds', 'peak_inflight', 'requests', 'concurrency', 'repeat') if k in saved}
        result = capacity.block_summary(rows, basic)
        assert saved == result
        for field, selected in [('successful_latency_ms', [r for r in rows if r['status'] == 'valid']),
                                ('all_attempt_latency_ms', rows)]:
            independently = distribution([r['elapsed_ms'] for r in selected])
            assert independently['n'] == result[field]['n']
            if selected:
                assert independently['mean'] == result[field]['mean_ms']
                assert independently['p95'] == result[field]['p95_ms']
        result.update(http_status_counts=dict(Counter(str(r.get('http_status')) for r in rows)),
                      timeout_count=sum('Timeout' in r.get('error_type', '') for r in rows),
                      reconstructed_peak_inflight=peak)
        return basic, result

    reports = []
    for model in p1['models']:
        run = 'run-001' if model in done else 'run-002'
        folder = BASE / run / model
        family, model_path, served = MODELS[model]
        profile = CLASSIFICATION_PROFILES.get(model)
        deployment = read(folder / 'deployment.json')
        assert (deployment['model'], deployment['family'], deployment['served_model']) == (model, family, served)
        src = BASE / ('source' if run == 'run-001' else 'source-r2') / 'deploy/docker/decision-eval'
        for name, h in deployment['source_sha256'].items():
            assert sha(src / name) == h
        for key in ('image_id', 'head_image_id'):
            if key in deployment:
                assert command(['docker', 'image', 'inspect', deployment[key], '--format', '{{.Id}}']) == deployment[key]
        scheduling = deployment['scheduling']
        if family in ('worker', 'llm', 'sglang', 'jev'):
            assert scheduling['max_sequences'] == 32 and scheduling['max_batched_tokens'] == 32768
        if family == 'laya':
            assert deployment['env']['LAYA_MAX_CONCURRENT'] == '32'
            assert scheduling['http_admission_limit'] == 32 and scheduling['single_request_executor_workers'] == 1
            assert scheduling['cross_request_batching'] is False
        if profile:
            assert deployment['classification_profile'] == profile
            assert '--reasoning-parser' in deployment['command']
        if family == 'llm' and served == 'Qwen3.8-27B':
            args = deployment['command']
            template = json.loads(args[args.index('--default-chat-template-kwargs') + 1])
            assert template == ({'enable_thinking': True, 'reasoning_effort': 'xhigh'} if profile else {'enable_thinking': False})
        assert read(folder / 'auth.json') == {'ready': 200, 'unauthorized': [401, 401, 401]}
        events = [e for e in log_states[run] if e.get('model') == model]
        assert [(e['repeat'], e['concurrency']) for e in events if e['stage'] == 'measuring'] == expected_measure
        assert [(e['repeat'], e['concurrency']) for e in events if e['stage'] == 'warmup'] == expected_warm
        assert sum(e['stage'] == 'loading' for e in events) == 1
        assert sum(e['stage'] == 'model_complete' for e in events) == 1
        assert {p.name for p in folder.glob('responses-*')} == {f'responses-n{n}-r{r}.jsonl' for r, n in expected_measure}
        assert {p.name for p in folder.glob('warmup-*.jsonl')} == {f'warmup-n{n}-r{r}.jsonl' for r, n in expected_warm}
        blocks, details, warm_details, all_rows, mtimes = [], [], [], [], []
        for kind, pairs, inputs in [('warmup', expected_warm, warm_jobs), ('responses', expected_measure, jobs)]:
            for repeat, n in pairs:
                path = folder / f'{kind}-n{n}-r{repeat}.jsonl'
                rows = benchmark.read_jsonl(path)
                block_path = folder / f'{"block" if kind == "responses" else "warmup"}-n{n}-r{repeat}.json'
                basic, result = audit_block(rows, inputs, read(block_path), family, served, profile)
                st = path.stat()
                result.update(file=path.name, bytes=st.st_size, mtime_utc=datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat())
                mtimes.append(st.st_mtime)
                if kind == 'responses':
                    blocks.append({**basic, 'records': rows})
                    details.append(result)
                    all_rows.extend(rows)
                else:
                    assert result['failed'] == 0
                    warm_details.append(result)
        assert mtimes == sorted(mtimes)
        summary = capacity.summarize(blocks)
        assert read(folder / 'summary.json') == {'model': model, **summary}
        reference = {b['repeat']: {r['sequence']: r for r in b['records']} for b in blocks if b['concurrency'] == 1}
        n1_blocks = {b['repeat']: b for b in details if b['concurrency'] == 1}
        for result, block in zip(details, blocks):
            paired = [(r, reference[block['repeat']][r['sequence']]) for r in block['records']]
            valid = [(a, b) for a, b in paired if a['status'] == b['status'] == 'valid']
            base = n1_blocks[block['repeat']]
            result.update(paired_attempts_with_n1=len(paired), paired_valid_with_n1=len(valid),
                          decision_changes_vs_n1=sum(a['prediction'] != b['prediction'] for a, b in valid))
            for metric in ('mean_ms', 'p95_ms'):
                result[metric + '_ratio_to_same_round_n1'] = result['successful_latency_ms'][metric] / base['successful_latency_ms'][metric]
            result['throughput_ratio_to_same_round_n1'] = result['effective_requests_per_second'] / base['effective_requests_per_second']
        n1_repeat = []
        for a, b in ((1, 2), (1, 3), (2, 3)):
            valid = [(reference[a][i], reference[b][i]) for i in range(199)
                     if reference[a][i]['status'] == reference[b][i]['status'] == 'valid']
            n1_repeat.append({'rounds': [a, b], 'attempts': 199, 'paired_valid': len(valid),
                              'changes': sum(x['prediction'] != y['prediction'] for x, y in valid)})
        usage = {}
        for n in capacity.LEVELS:
            rows = [r for b in blocks if b['concurrency'] == n for r in b['records']]
            usage[str(n)] = {'attempts': len(rows), 'http_status_counts': dict(Counter(str(r.get('http_status')) for r in rows)),
                            'timeout_count': sum('Timeout' in r.get('error_type', '') for r in rows)}
            for key in sorted({k for r in rows for k, v in (r.get('usage') or {}).items()
                               if type(v) in (int, float)}):
                usage[str(n)][key] = distribution([r['usage'][key] for r in rows if key in (r.get('usage') or {})])
            if family == 'llm':
                usage[str(n)].update(finish_reasons=dict(Counter(r.get('finish_reason') for r in rows)),
                    reasoning_chars=distribution([r['reasoning_chars'] for r in rows]),
                    reasoning_tokens=distribution([(r['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens', 0) for r in rows]))
            batches = [(r.get('execution') or {}).get('systemone_batch') for r in rows]
            batches = [b for b in batches if b]
            if batches:
                usage[str(n)]['response_batch_metadata'] = {
                    'records': len(batches), 'batch_sizes': dict(Counter(b['batch_size'] for b in batches)),
                    'cross_request_batching_flags': dict(Counter(str(b['cross_request_batching']) for b in batches)),
                    'queue_wait_ms': distribution([b['queue_wait_ms'] for b in batches]),
                    'note': 'Response-level observations, not a unique GPU batch count.'}
        reports.append({'model': model, 'run': run, 'family': family, 'served_model': served,
            'model_path': model_path, 'deployment_sha256': sha(folder / 'deployment.json'),
            'image_id': deployment['image_id'], 'tp': deployment['tp'], 'scheduling': scheduling,
            'deployment_command_sha256': benchmark.digest(deployment['command']),
            'source_sha256': deployment['source_sha256'], 'classification_profile': profile,
            'auth': read(folder / 'auth.json'), 'measured_requests': len(all_rows),
            'failed': sum(r['status'] != 'valid' for r in all_rows),
            'checks': {'full_ids_input_request_hashes': True, 'calls_exactly_one': True,
                       'no_replay': True, 'timing_and_peak_reconstructed': True,
                       'metrics_recomputed_equal_saved': True, 'load_order_matches_log': True,
                       'same_deployment_all_levels': True},
            'summary': summary, 'measurement_blocks': details, 'warmup_blocks': warm_details,
            'n1_all_round_pairs': n1_repeat, 'usage_by_concurrency': usage})

    assert sum(r['measured_requests'] for r in reports) == 38208
    prior_warm = [r for p in (BASE / 'run-001/laya-multilingual').glob('warmup-*.jsonl') for r in benchmark.read_jsonl(p)]
    assert len(prior_warm) == 140 and Counter(r['http_status'] for r in prior_warm) == {200: 137, 503: 3}
    assert not list((BASE / 'run-001/laya-multilingual').glob('responses-*'))
    live = read(BASE / 'recovery-live-check.json')
    assert live['laya_image_command_and_other_env_unchanged'] is True
    owned = command(['docker', 'ps', '--filter', 'label=evaluation.decision.acceptance=true', '--format', '{{.Names}}']).splitlines()
    assert set(owned) == {'decision-acceptance-model', 'decision-acceptance-proxy'}
    final = read(BASE / 'run-002/qwen3.8-27b-thinking/deployment.json')
    inspected = json.loads(command(['docker', 'inspect', '--format',
        '{"image":{{json .Image}},"path":{{json .Path}},"args":{{json .Args}},"status":{{json .State.Status}},"oom":{{json .State.OOMKilled}},"owned":{{json (index .Config.Labels "evaluation.decision.acceptance")}}}', 'decision-acceptance-model']))
    assert inspected['owned'] == 'true' and inspected['status'] == 'running' and not inspected['oom']
    assert inspected['image'] == final['image_id']
    assert [inspected['path'], *inspected['args']] == final['command']
    assert inventory(BASE / 'run-001') | inventory(BASE / 'run-002') == before
    capacity.check_artifacts(p2['preserved_artifact_sha256'])
    capacity.check_artifacts(prep['source_sha256'])
    evidence = {'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'audit_script_sha256': script_sha,
        'scope': 'Completed engineering HTTP capacity preflight only; offline audit, no new inference',
        'systemd': unit, 'completed_at_utc': state['updated_at_utc'], 'shared_lock_released': True,
        'configurations': 16, 'distinct_models': 15, 'measured_requests': 38208,
        'measured_failed': sum(r['failed'] for r in reports), 'accepted_warmup_requests': 2560,
        'accepted_warmup_failed': 0, 'new_inference_calls': 0,
        'dataset': {k: data[k] for k in ('source', 'requests', 'excluded_pairs', 'characters',
            'engineering_ids', 'validation_ids', 'formal_pools', 'isolation_checked', 'truncation')},
        'dataset_protocol_sha256': sha(data_dir / 'protocol.json'),
        'protocol_sha256': {run: sha(BASE / run / 'protocol.json') for run in ('run-001', 'run-002')},
        'recovery': {'predecessor': link, 'old_laya_measured': 0, 'old_laya_warmup_attempts': 140,
            'old_laya_warmup_http_status': {200: 137, 503: 3}, 'prior_completed_records': 7164,
            'unmeasured_suffix_records': 31044, 'prior_measured_replayed': 0,
            'startup_correction_no_inference': True, 'laya_admission_only': True},
        'preservation': {'unchanged': True, 'classification_audit_sha256': dict(zip(map(str, audit_paths), audit_hashes)),
            'classification_artifact_groups': [98, 45, 66], 'classification_artifact_union': 200,
            'run002_preservation_manifest_count': len(p2['preserved_artifact_sha256']),
            'source_snapshot_files': {'source': len(old_snapshot), 'source-r2': len(prep['source_sha256'])},
            'original_additional_files_matching_recovery_sibling_baseline': len(original_sibling_baselines),
            'classification_functions_ast_equal': 39, 'capacity_measurement_functions_ast_equal': 7,
            'original_runner_log_unchanged': True, 'preinference_failed_log_unchanged': True,
            'run001_all_files_unchanged': True, 'audit_inputs_unchanged_during_audit': True},
        'reports': reports,
        'retained_service': {'model': 'qwen3.8-27b-thinking', 'owned_containers': owned,
            'status': inspected['status'], 'image_id': inspected['image'], 'command_matches_deployment': True,
            'gpu': command(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv,noheader'])},
        'limitations': ['199 predeclared short/medium engineering pairs; not formal relevance accuracy',
            '15 distinct models and 16 configurations; Qwen3.8 modes share weights',
            'HTTP inflight is not GPU batch size or GPU parallelism',
            'No frozen 20 percent latency pass line; no global optimal deployment claim',
            'Closed-loop finite workload, not open-loop production capacity or an SLA',
            'No continuous GPU resource or unique GPU batch telemetry across all runs',
            'Raw response bodies not retained; response hashes and frozen validator metadata are audited',
            'Model weight provenance inherited from prior acquisition/deployment evidence; large weights not rehashed'],
        'artifact_sha256': {**before, **old_inventory, **prep['source_sha256'],
            **p2['preserved_artifact_sha256'], **inventory(data_dir),
            **{str(BASE / name): sha(BASE / name) for name in ('preparation.json', 'preparation-r2.json',
                'preparation-r2-startup-correction.json', 'recovery-live-check.json',
                'runner.log', 'runner-r2.log', 'runner-r2b.log')}}}
    capacity.save(BASE / OUTPUT, evidence)
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == '__main__':
    if sys.argv[1:] == ['--collect']:
        script = Path(__file__).resolve()
        local = script.with_suffix('.json')
        assert not local.exists(), 'Local evidence exists; do not overwrite'
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            'root@test-host.invalid', str(ROOT / 'classification-20261015/venv/bin/python'),
            '-B', '-', '--remote', sha(script)], input=script.read_text(), text=True,
            capture_output=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(result.returncode)
        evidence = json.loads(result.stdout)
        with local.open('x') as stream:
            json.dump(evidence, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(json.dumps({k: evidence[k] for k in ('audited_at_utc', 'measured_requests', 'measured_failed',
            'accepted_warmup_requests', 'preservation')}, ensure_ascii=False))
        print('Local evidence:', local)
    elif len(sys.argv) == 3 and sys.argv[1] == '--remote':
        run_audit(sys.argv[2])
    else:
        raise SystemExit('Use --collect locally; remote mode is for the streamed offline audit only.')
