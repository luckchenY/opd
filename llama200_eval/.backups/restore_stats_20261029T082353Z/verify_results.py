#!/usr/bin/env python3
"""Stream a completed benchmark's artifacts and independently verify its totals."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


EXPECTED = {
    'aime': {'aime24': {'default': 30}, 'aime25': {'default': 30}},
    'olympiad': {'olympiad_bench': {'OE_TO_maths_en_COMP': 674,
                                   'OE_TO_maths_zh_CEE': 1240,
                                   'OE_TO_maths_zh_COMP': 408}},
    'gpqa': {'gpqa_diamond': {'default': 198}},
}
PROTOCOL = {'aime': (8192, .6), 'olympiad': (8192, 1.), 'gpqa': (32768, .6)}


def preserve_gpqa_choices(self, record):
    """Same preprocessing as compat_plugins; ordinary square brackets stay."""
    import random
    def clean(text):
        return ' ' if text is None else text.strip().replace(' [title]', '. ').replace('  ', ' ')
    choices = [clean(record[key]) for key in
               ['Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3', 'Correct Answer']]
    correct = clean(record['Correct Answer'])
    assert choices.count(correct) == 1, 'GPQA correct answer collides with a distractor'
    random.shuffle(choices)
    return {'choices': choices, 'answer': chr(65 + choices.index(correct))}


def load_task_config(native):
    path = Path(native) / 'configs' / 'task_config.yaml'
    import yaml
    return yaml.safe_load(path.read_text(encoding='utf-8')), path


def validate_task_config(config, stage, model):
    assert config['datasets'] == list(EXPECTED[stage]), 'Unexpected benchmark selection'
    allowed_models = {str(model), Path(model).name}
    assert config.get('model') in allowed_models, 'Unexpected model alias/path'
    if config.get('model_id') is not None:
        assert config['model_id'] in allowed_models, 'Unexpected model alias/path'
    assert config['repeats'] == 4, 'This evaluator requires four responses per question'
    assert config.get('seed') == 42, 'Expected the fixed dataset seed42'
    assert config.get('limit') is None, 'Partial evaluation cannot be reported as the complete benchmark'
    generation = config['generation_config']
    max_tokens, temperature = PROTOCOL[stage]
    assert generation.get('max_tokens') == max_tokens, 'Unexpected output token limit'
    assert generation.get('temperature') == temperature, 'Unexpected sampling temperature'
    assert generation.get('top_p') == .95, 'Unexpected top_p'
    assert generation.get('extra_body', {}).get('top_k') == 50, 'Unexpected top_k'
    for dataset, subsets in EXPECTED[stage].items():
        arguments = config['dataset_args'][dataset]
        assert arguments['aggregation'] == 'mean_and_pass_at_k'
        assert arguments.get('subset_list') == list(subsets), 'Unexpected subset selection/order'
    return int(generation['max_tokens'])


def rebuild_gpqa_samples(config):
    """Rebuild all792 per-index choices from this run's local data/config."""
    import random
    import numpy as np
    import pyarrow.parquet as pq
    from evalscope.api.dataset import DatasetDict
    from evalscope.api.dataset.loader import DictDataLoader
    from evalscope.api.registry import get_benchmark
    from evalscope.benchmarks.gpqa.gpqa_adapter import GPQAAdapter
    from evalscope.config import TaskConfig
    assert isinstance(config['seed'], int) and config['repeats'] == 4
    assert config['datasets'] == ['gpqa_diamond']
    args = config['dataset_args']['gpqa_diamond']
    assert args['subset_list'] == ['default'] and args['few_shot_num'] == 0
    assert args.get('shuffle') is False and args.get('shuffle_choices') is False
    local = Path(args.get('local_path') or args['dataset_id']).expanduser().resolve(strict=True)
    data = [local] if local.is_file() else list(local.glob('train-*.parquet'))
    assert len(data) == 1, 'Expected one local GPQA parquet; network downloads are not used'
    raw = pq.read_table(data[0]).to_pylist()
    assert len(raw) == 198
    old = GPQAAdapter._process_input
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        GPQAAdapter._process_input = preserve_gpqa_choices
        adapter = get_benchmark('gpqa_diamond', TaskConfig.from_dict(config))
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        samples = DictDataLoader(raw, sample_fields=adapter.record_to_sample,
            filter_func=adapter.sample_filter, repeats=4, shuffle=False,
            shuffle_choices=False, seed=seed, auto_id=True).load()
        adapter.test_dataset = DatasetDict({'default': samples})
        adapter.fewshot_dataset = None
        adapter._post_process_samples()
        assert len(samples) == 792
        result = {}
        for index, sample in enumerate(samples):
            assert sample.id == index and sample.group_id == index // 4
            inputs = [m.model_dump(mode='json') if hasattr(m, 'model_dump') else m for m in sample.input]
            correct = sample.metadata['correct_answer']
            clean = correct.strip().replace(' [title]', '. ').replace('  ', ' ')
            assert sample.choices.count(clean) == 1
            assert sample.choices[ord(sample.target) - 65] == clean
            identity = {'question': raw[index // 4]['Question'], 'correct_answer': correct,
                        'incorrect_answers': sorted(sample.metadata['incorrect_answers']),
                        'choice_set': sorted(sample.choices)}
            result[('default', index)] = {'target': sample.target,
                'input_sha': message_identities(inputs)[0], 'metadata': sample.metadata,
                'group_identity': canonical_sha(identity)}
        return result, {'path': str(data[0]), 'sha256': file_sha(data[0]), 'questions': len(raw), 'seed': seed}
    finally:
        GPQAAdapter._process_input = old
        random.setstate(py_state)
        np.random.set_state(np_state)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def content_identity(content):
    """Retain text/reasoning while normalizing equivalent ContentText forms."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{'type': 'text', 'value': content}]
    assert isinstance(content, list), 'Unexpected message content schema'
    result = []
    for block in content:
        if isinstance(block, str):
            kind, value = 'text', block
        else:
            assert isinstance(block, dict), 'Unexpected structured content block'
            kind = block.get('type')
            if kind == 'text':
                value = block['text']
            elif kind in {'reasoning', 'thinking'}:
                kind = 'reasoning'
                value = block.get('reasoning', block.get('thinking', block.get('text')))
                assert isinstance(value, str), 'Missing reasoning content'
            else:
                result.append({'type': 'structured', 'value': block})
                continue
        assert isinstance(value, str)
        if result and result[-1]['type'] == kind:
            result[-1]['value'] += value
        else:
            result.append({'type': kind, 'value': value})
    return result


def semantic_message(message, default_role=None):
    role = message.get('role', default_role)
    assert role is not None and 'content' in message
    content = content_identity(message['content'])
    reasoning = message.get('reasoning_content')
    if reasoning:
        item = {'type': 'reasoning', 'value': reasoning}
        if not content or content[0] != item:
            content = [item] + content
    result = {'role': role, 'content': content}
    for key in ['tool_call_id', 'tool_calls', 'name']:
        if message.get(key) is not None:
            result[key] = message[key]
    return result


def message_identities(messages):
    """Separate semantic input from generated output and runtime metadata.

    EvalScope serializes the generated assistant turn inside messages and does
    not repeat target in prediction records. Message IDs/timing are not input.
    Retain earlier assistant turns if a future benchmark uses few-shot input.
    """
    assert isinstance(messages, list) and messages
    semantic = [semantic_message(m) for m in messages]
    inputs = semantic[:-1] if semantic[-1]['role'] == 'assistant' else semantic
    assert inputs and any(m['role'] == 'user' for m in inputs)
    return canonical_sha(inputs), canonical_sha(semantic)


def pass_k(values, k):
    n, c = len(values), int(sum(values))
    return 1.0 if n - c < k else 1 - math.prod((n - c - j) / (n - j) for j in range(k))


def iter_records(paths, dataset):
    for path in sorted(paths):
        subset = path.stem[len(dataset) + 1:]
        with path.open('rb') as handle:
            for line in handle:
                if line.strip():
                    yield subset, json.loads(line)


def check_score(actual, expected, description):
    if float(actual) != float(expected):
        raise ValueError(f'{description}: reported={actual}, independently_recomputed={expected}')


def verify_dataset(native, dataset, expected, model, max_tokens, normalize, gpqa_samples=None):
    reports = list((native / 'reports').glob(f'*/{dataset}.json'))
    if len(reports) != 1:
        raise ValueError(f'{dataset}: expected exactly one report, got {len(reports)}')
    report = json.loads(reports[0].read_text(encoding='utf-8'))
    assert report['dataset_name'] == dataset
    assert report['model_name'] == Path(model).name
    reviews = list((native / 'reviews').glob(f'*/{dataset}_*.jsonl'))
    predictions = list((native / 'predictions').glob(f'*/{dataset}_*.jsonl'))
    assert reviews and predictions, f'{dataset}: predictions and reviews must both be preserved'
    records = {}
    groups = defaultdict(lambda: defaultdict(list))
    identities = {}
    invalid = 0
    if dataset == 'gpqa_diamond':
        assert gpqa_samples is not None, 'GPQA requires independently rebuilt per-index identities'
    for subset, row in iter_records(reviews, dataset):
        assert subset in expected, f'{dataset}: unexpected subset {subset}'
        key = (subset, int(row['index']))
        assert key not in records, f'{dataset}: duplicate review {key}'
        score = row['sample_score']['score']
        acc = float(score['value']['acc'])
        assert acc in (0.0, 1.0)
        extracted = str(score.get('extracted_prediction') or '').strip()
        target = row['target']
        if dataset == 'gpqa_diamond':
            target_letter = str(target).strip().upper()
            assert target_letter in {'A', 'B', 'C', 'D'}
            independently_correct = float(extracted.upper() == target_letter)
            check_score(acc, independently_correct, f'{dataset} sample {key}')
            invalid += extracted.upper() not in {'A', 'B', 'C', 'D'}
        elif dataset in {'aime24', 'aime25'}:
            # AIME24's targets include \\boxed{25}; the official grader also
            # accepts equivalent LaTeX/numeric forms. Use an independent
            # integer comparison for the simple case, official rule otherwise.
            target_text = str(target).strip()
            boxed = re.fullmatch(r'\\boxed\s*\{\s*([0-9]+)\s*\}', target_text)
            numeric_target = boxed.group(1) if boxed else target_text
            if re.fullmatch(r'\d+', numeric_target) and re.fullmatch(r'[+]?\d+(?:\.0+)?', extracted):
                independently_correct = float(float(extracted) == int(numeric_target))
            else:
                from evalscope.benchmarks.aime.grader import grade_answer
                try:
                    independently_correct = float(bool(grade_answer(extracted, target_text)))
                except Exception:
                    independently_correct = 0.0
            check_score(acc, independently_correct, f'{dataset} sample {key}')
            invalid += not bool(extracted)
        group = str(row['sample_score']['group_id'])
        assert int(group) == int(row['index']) // 4, f'{dataset}: repeat-group index mismatch'
        input_sha, conversation_sha = message_identities(row['messages'])
        identity = (canonical_sha(target), input_sha)
        if dataset == 'gpqa_diamond':
            source = gpqa_samples[key]
            assert row['sample_score']['sample_id'] == key[1], 'GPQA review sample_id mismatch'
            assert target == source['target'], f'GPQA shuffled target differs from seeded source: {key}'
            assert input_sha == source['input_sha'], f'GPQA prompt differs from seeded source: {key}'
            # ReviewResult serializes source metadata inside SampleScore,
            # whereas ModelResult predictions use top-level metadata.
            # Validate every present source location, so a conflicting top-level
            # copy cannot be silently ignored in favor of the nested one.
            metadata_copies = []
            if row['sample_score'].get('sample_metadata') is not None:
                metadata_copies.append(('sample_score.sample_metadata', row['sample_score']['sample_metadata']))
            if row.get('metadata') is not None:
                metadata_copies.append(('metadata', row['metadata']))
            assert metadata_copies, f'GPQA review source metadata missing: {key}'
            for location, metadata in metadata_copies:
                assert isinstance(metadata, dict), f'GPQA review source metadata invalid: {key}/{location}'
                for field in ['correct_answer', 'incorrect_answers']:
                    assert metadata.get(field) == source['metadata'][field], f'GPQA review source metadata differs: {key}/{location}/{field}'
            # The original question and answer semantics are group-invariant;
            # displayed A-D order and therefore the target letter are not.
            identity = (canonical_sha(target), source['group_identity'])
        group_key = (subset, group)
        if group_key in identities:
            group_identity = identity[1] if dataset == 'gpqa_diamond' else identity
            assert identities[group_key] == group_identity, f'{dataset}: different source questions/answers in one repeat group'
        identities[group_key] = identity[1] if dataset == 'gpqa_diamond' else identity
        groups[subset][group].append(acc)
        records[key] = {'target_sha': identity[0], 'input_sha': input_sha,
                        'conversation_sha': conversation_sha}
    for subset, n in expected.items():
        assert len(groups[subset]) == n, f'{dataset}/{subset}: expected {n} questions, got {len(groups[subset])}'
        assert all(len(values) == 4 for values in groups[subset].values()), f'{dataset}/{subset}: incomplete repeat groups'
        assert {i for s, i in records if s == subset} == set(range(n * 4)), f'{dataset}/{subset}: sample index coverage mismatch'
    assert len(records) == sum(expected.values()) * 4
    predicted = set()
    truncated = 0
    output_tokens = 0
    for subset, row in iter_records(predictions, dataset):
        key = (subset, int(row['index']))
        assert key in records and key not in predicted, f'{dataset}: duplicate or unmatched prediction {key}'
        predicted.add(key)
        input_sha, conversation_sha = message_identities(row['messages'])
        assert records[key]['input_sha'] == input_sha, f'{dataset}: prediction/review input mismatch {key}'
        assert records[key]['conversation_sha'] == conversation_sha, f'{dataset}: prediction/review generated turn mismatch {key}'
        if row.get('target') is not None:
            assert records[key]['target_sha'] == canonical_sha(row['target']), f'{dataset}: prediction/review target mismatch {key}'
        if row.get('model') is not None:
            assert row['model'] in {model, Path(model).name}, f'{dataset}: unexpected prediction model'
        if dataset == 'gpqa_diamond':
            source = gpqa_samples[key]
            assert isinstance(row.get('metadata'), dict), f'GPQA prediction source metadata missing: {key}'
            for field in ['correct_answer', 'incorrect_answers']:
                assert row['metadata'].get(field) == source['metadata'][field], f'GPQA prediction source metadata differs: {key}/{field}'
        output = row['model_output']
        assert not output.get('error') and len(output.get('choices', [])) == 1, f'{dataset}: failed model output {key}'
        choice = output['choices'][0]
        assert choice.get('message') is not None
        assert row['messages'][-1].get('role') == 'assistant', f'{dataset}: missing persisted generated turn {key}'
        assert semantic_message(choice['message'], 'assistant') == semantic_message(row['messages'][-1]), f'{dataset}: model_output differs from persisted generated turn {key}'
        tokens = float(output['usage']['output_tokens'])
        assert 0 <= tokens <= max_tokens, f'{dataset}: output token count exceeds configured limit'
        output_tokens += tokens
        truncated += choice.get('stop_reason') in {'max_tokens', 'length'} or choice.get('finish_reason') == 'length'
    assert predicted == set(records), f'{dataset}: incomplete predictions'

    subsets = {}
    for subset, mapping in groups.items():
        values = list(mapping.values())
        raw = {'mean_acc': sum(sum(v) for v in values) / (4 * len(values))}
        raw.update({f'mean_acc_pass@{k}': sum(pass_k(v, k) for v in values) / len(values) for k in range(1, 5)})
        subsets[subset] = {'questions': len(values), 'samples': len(values) * 4,
                           'metrics_raw': raw, 'metrics_normalized': {k: normalize(float(v)) for k, v in raw.items()}}
    metric_names = {m['name'] for m in report['metrics']}
    assert {'mean_acc', 'mean_acc_pass@4'} <= metric_names
    verified_metrics = {}
    for metric in report['metrics']:
        name = metric['name']
        if name not in {'mean_acc', *(f'mean_acc_pass@{k}' for k in range(1, 5))}:
            continue
        category_values = []
        found = set()
        for category in metric['categories']:
            weighted = []
            for sub in category['subsets']:
                if sub.get('is_aggregate'):
                    continue
                subset = sub['name']
                assert subset in subsets and subset not in found
                found.add(subset)
                check_score(sub['num'], subsets[subset]['samples'], f'{dataset}/{subset} sample count')
                expected_score = subsets[subset]['metrics_normalized'][name]
                check_score(sub['score'], expected_score, f'{dataset}/{subset} {name}')
                weighted.append((expected_score, sub['num']))
            count = sum(n for _, n in weighted)
            assert count and category['num'] == count
            expected_category = normalize(float(sum(v * n for v, n in weighted) / count))
            check_score(category['score'], expected_category, f'{dataset} category {name}')
            category_values.append((expected_category, count))
        assert found == set(expected)
        count = sum(n for _, n in category_values)
        assert metric['num'] == count == len(records)
        expected_metric = normalize(float(sum(v * n for v, n in category_values) / count))
        check_score(metric['score'], expected_metric, f'{dataset} overall {name}')
        verified_metrics[name] = expected_metric
    check_score(report['score'], verified_metrics['mean_acc'], f'{dataset} report primary score')
    total_questions = sum(expected.values())
    raw_micro = {name: sum(sub['metrics_raw'][name] * sub['questions'] for sub in subsets.values()) / total_questions
                 for name in verified_metrics}
    return {'questions': total_questions, 'samples': len(records), 'subsets': subsets,
            'metrics_raw_micro': raw_micro, 'reported_metrics_verified': verified_metrics,
            'max_tokens_rate': truncated / len(records), 'mean_output_tokens': output_tokens / len(records),
            'invalid_answer_rate': invalid / len(records) if dataset != 'olympiad_bench' else None,
            'report_path': str(reports[0]), 'report_sha256': hashlib.sha256(reports[0].read_bytes()).hexdigest(),
            'score_comparison': 'Per-subset scores, followed by EvalScope hierarchical four-decimal normalization',
            'gpqa_per_index_choices_verified': dataset == 'gpqa_diamond',
            'gpqa_group_basis': 'Original question and correct/distractor semantics; per-repeat A-D permutations allowed' if dataset == 'gpqa_diamond' else None,
            'math_judge_note': 'Olympiad individual mathematical grades use the official EvalScope rule judge; aggregation and grouping independently recomputed.'}


def verify_output(stage, output, model, normalize=None):
    if normalize is None:
        from evalscope.report.report import normalize_score
        normalize = normalize_score
    out = Path(output).resolve(strict=True)
    native_paths = [p for p in (out / 'native').glob('*') if p.is_dir()]
    assert len(native_paths) == 1, f'Expected one EvalScope run under {out}/native, got {len(native_paths)}'
    native = native_paths[0]
    config, config_path = load_task_config(native)
    max_tokens = validate_task_config(config, stage, model)
    gpqa_samples, data_evidence = rebuild_gpqa_samples(config) if stage == 'gpqa' else (None, None)
    datasets = {dataset: verify_dataset(native, dataset, subsets, model,
                                        max_tokens, normalize, gpqa_samples)
                for dataset, subsets in EXPECTED[stage].items()}
    generation = config['generation_config']
    result = {'status': 'succeeded', 'stage': stage, 'model': str(model),
              'repeats': 4, 'seed': config['seed'], 'max_new_tokens': max_tokens,
              'temperature': generation.get('temperature'), 'top_p': generation.get('top_p'),
              'top_k': generation.get('extra_body', {}).get('top_k'),
              'task_config_path': str(config_path), 'task_config_sha256': file_sha(config_path),
              'source_data': data_evidence, 'datasets': datasets,
              'all_predictions_and_scores_verified': True}
    target = out / 'verification.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    temporary.replace(target)
    return result, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=EXPECTED, required=True)
    parser.add_argument('--output', required=True, type=Path, help='Stage directory passed to swift --eval_output_dir')
    parser.add_argument('--model', required=True, help='Effective model directory used by swift')
    args = parser.parse_args()
    result, target = verify_output(args.stage, args.output, args.model)
    for dataset, values in result['datasets'].items():
        metrics = values['reported_metrics_verified']
        print(f"{dataset}: {values['questions']} questions, {values['samples']} responses; "
              f"mean_acc={metrics['mean_acc']:.4f}, pass@4={metrics['mean_acc_pass@4']:.4f}")
    print(f'Verified: {target}', flush=True)


if __name__ == '__main__':
    main()
