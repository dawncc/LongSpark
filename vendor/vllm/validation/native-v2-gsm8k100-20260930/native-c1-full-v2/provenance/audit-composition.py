"""Compose the frozen native runner audit with independent source/warmup/log checks."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_warmup(directory, plan, fixture):
    max_prompt = max(len(row['input_ids']) for row in fixture[:plan['limit']])
    cap = plan['max_num_seqs']
    expected_max = min(plan['max_num_batched_tokens'], plan['max_model_len'],
                       max_prompt if cap == 1 else plan['max_num_batched_tokens'])
    require(plan['prefill_shape_warmup']['max_width'] == expected_max, 'Incorrect prefill max width')
    widths = list(range(9, expected_max + 1))
    batches = list(range(1, cap + 1))
    result = {}
    for method in ('vanilla', 'longspark'):
        path = directory / method / 'prefill-warmup.json'
        rows = read(path)
        require(isinstance(rows, list) and len(rows) == 1, method + ': expected TP1 warmup report')
        row = rows[0]
        require(row['state_unchanged'] is True, method + ': warmup mutated state')
        if method == 'vanilla':
            require(row['status'] == 'not_applicable' and row['updates'] == 0
                    and row['widths'] == [] and row['batches'] == [], 'Invalid AR warmup evidence')
        else:
            require(row['status'] == 'completed', 'LongSpark prefill warmup incomplete')
            require(row['max_width'] == expected_max and row['widths'] == widths,
                    'Incomplete LongSpark prefill width coverage')
            require(row['batches'] == batches and row['updates'] == len(widths) * len(batches),
                    'Incomplete LongSpark prefill batch coverage')
            require(row['comparison'] == 'bitwise torch.equal(uint8 views)', 'Missing bitwise state comparison')
            require(row['dtype'] == 'torch.bfloat16' and row['device'].startswith('cuda'), 'Wrong warmup device/dtype')
            require(row['start_position'] == 1 and row['valid_count'] == 0, 'Warmup is not a no-op update')
            require(row['slot_policy'] == 'private positive slots: max_num_reqs + arange(batch)',
                    'Warmup private-slot evidence missing')
            require(row['input_layout'] == 'contiguous [B,L,width,KVH,D]; boundary hidden [B,L,H]',
                    'Warmup input strides differ from real gathered tensors')
            require(row['wall_seconds'] > 0, 'Missing warmup timing')
        result[method] = dict(report_sha256=sha(path), evidence=row)
    return dict(status='passed', max_prompt=max_prompt, expected_max_width=expected_max,
                expected_widths=widths, expected_batches=batches, methods=result)


def audit_logs(directory, plan):
    methods = {}
    for method in ('vanilla', 'longspark'):
        path = directory / (method + '.log')
        lines = path.read_text().splitlines()
        warm = [i for i, line in enumerate(lines) if re.match(method + r'/warmup: ', line)]
        measured = [i for i, line in enumerate(lines) if re.match(method + r'/measurement: ', line)]
        require(len(warm) == plan['warmup_count'] and len(warm) > 0, method + ': missing warmup result lines')
        require(len(measured) == plan['limit'], method + ': missing measurement result lines')
        start, end = warm[-1] + 1, measured[-1]
        require(start <= measured[0], method + ': invalid warmup/measurement ordering')
        activated = [dict(line=i + 1, text=line) for i, line in enumerate(lines[:start])
                     if 'Kernel JIT monitor activated' in line and 'mode=warn' in line]
        require(len(activated) == 1, method + ': JIT monitoring activation evidence missing')
        marker = method + ': prefill state shape warmup through width ' + str(plan['prefill_shape_warmup']['max_width'])
        shape_lines = [i for i, line in enumerate(lines) if marker in line]
        require(len(shape_lines) == 1 and shape_lines[0] < warm[0], method + ': shape warmup not before requests')
        candidates = [dict(line=i + 1, text=lines[i]) for i in range(start, end + 1)
                      if re.search(r'JIT|compil|Capturing.*[Gg]raph', lines[i])]
        require(not candidates, method + ': compilation occurred during measurement: ' + repr(candidates))
        methods[method] = dict(log_sha256=sha(path), last_warmup_record_line=warm[-1] + 1,
                              measurement_window_start_line=start + 1,
                              first_measurement_record_line=measured[0] + 1,
                              last_measurement_record_line=end + 1,
                              jit_monitor_activation_lines=activated,
                              prefill_shape_warmup_marker_line=shape_lines[0] + 1,
                              measurement_compilation_events=candidates)
    return dict(status='passed', measurement_jit=False, steady_state_no_measured_jit=True,
                window_definition='After the last warmup result, including first-request execution before its result, through the final measurement result inclusive.',
                scope='No compile/JIT/capture events reported in these log windows by the active JIT monitor; native graph counts are independently checked.',
                methods=methods)


def audit_sources(root, directory, remote):
    plan = read(directory / 'plan.json')
    expected = plan['integration_source_hashes']
    require(len(expected) == 16 and not any(Path(p).name.startswith('._') for p in expected),
            'Expected 16 genuine source/manifest entries and no metadata sidecars')
    rows = []
    for path, expected_sha in expected.items():
        actual = sha(root / path)
        require(actual == expected_sha, 'Local source hash mismatch: ' + path)
        rows.append(dict(path=path, sha256=actual, verified=True))
    checks = {}
    for filename in ('manifest.json', 'completed.json', 'vanilla/manifest.json', 'longspark/manifest.json'):
        obj = read(directory / filename)
        require(obj['integration_source_hashes'] == expected, filename + ': source hash map mismatch')
        checks[filename] = dict(source_hash_map_matches_plan=True)
        if '/' in filename:
            require(obj['integration_sources_verified_before_load'] is True, filename + ': no pre-load verification')
            checks[filename]['verified_before_model_load'] = True
    sources = remote['sources']
    require(sources['status'] == 'passed' and sources['plan_sha256'] == sha(directory / 'plan.json'),
            'Remote post-run source audit/plan mismatch')
    require({row['path']:row['sha256'] for row in sources['files']} == expected
            and all(row['verified'] for row in sources['files']), 'Remote post-run sources mismatch')
    patch = remote['patch']
    manifest_path = root / 'vendor/vllm/patches/native-v2-custom-class.manifest.json'
    manifest = read(manifest_path)
    require(patch['status'] == 'passed' and patch['manifest_sha256'] == sha(manifest_path), 'Remote patch manifest mismatch')
    require(patch['patch_id'] == manifest['patch_id'], 'Wrong deployed patch ID')
    require(len(patch['files']) == 5 and all(row['verified'] for row in patch['files']), 'Incomplete deployed patch verification')
    require({row['path']:row['sha256'] for row in patch['files']} ==
            {row['path']:row['after_sha256'] for row in manifest['files']}, 'Deployed patch SHA mismatch')
    return dict(status='passed', source_files=rows, source_count=16, metadata_sidecars=0,
                remote_post_run_source_count=len(sources['files']), manifest_checks=checks), patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--remote-evidence', required=True, type=Path)
    parser.add_argument('--expect-limit', type=int, default=100)
    args = parser.parse_args()
    script = args.root / 'vendor/vllm/scripts/audit_native_v2_gsm8k.py'
    spec = importlib.util.spec_from_file_location('native_base_audit', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.audit_directory(args.directory, input_path=args.input, expect_limit=args.expect_limit)
    plan = read(args.directory / 'plan.json')
    fixture = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    report['prefill_shape_warmup_audit'] = audit_warmup(args.directory, plan, fixture)
    report['compilation_log_audit'] = audit_logs(args.directory, plan)
    report['source_integrity'], report['deployed_patch_integrity'] = audit_sources(
        args.root, args.directory, read(args.remote_evidence))
    report['audited_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['audit_tool_sha256'] = sha(script)
    report['audit_composition_sha256'] = sha(Path(__file__))
    report['warnings'] = []
    output = args.directory / 'audit-full.json'
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], report=str(output), parity=report['token_parity'],
                          measured_jit=False, source_count=16, warmup_widths=len(report['prefill_shape_warmup_audit']['expected_widths'])), indent=2))


if __name__ == '__main__':
    main()
