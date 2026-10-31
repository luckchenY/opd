#!/usr/bin/env python3
"""Prepare an isolated Llama200 tokenizer view and check local evaluation data.

Only the view is written. Model weights, original model files and datasets are
never changed. Successful CLI output is exactly the absolute model-view path.
"""
import argparse
from collections.abc import Mapping
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

STAGES = ('aime', 'olympiad', 'gpqa')
OLYMPIAD_SUBSETS = {'OE_TO_maths_en_COMP': 674, 'OE_TO_maths_zh_CEE': 1240,
                    'OE_TO_maths_zh_COMP': 408}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dataset_args(stage):
    """One source for both CPU data checks and Swift's dataset arguments."""
    key = {'aime': 'AIME_DATA_DIR', 'olympiad': 'OLYMPIAD_DATA_DIR',
           'gpqa': 'GPQA_DATA_DIR'}[stage]
    require(bool(os.environ.get(key)), f'{key} is required; use eval_llama200.sh or set it explicitly')
    root = Path(os.environ[key]).expanduser().resolve()
    if stage == 'aime':
        return {name: {'local_path': str(root / f'{name}_local'),
                       'aggregation': 'mean_and_pass_at_k'} for name in ['aime24', 'aime25']}
    if stage == 'olympiad':
        return {'olympiad_bench': {'local_path': str(root),
                'aggregation': 'mean_and_pass_at_k', 'subset_list': list(OLYMPIAD_SUBSETS)}}
    return {'gpqa_diamond': {'local_path': str(root / 'gpqa_diamond_local'),
            'aggregation': 'mean_and_pass_at_k', 'few_shot_num': 0, 'subset_list': ['default']}}


def expected_data_files(stage):
    args = dataset_args(stage)
    if stage == 'aime':
        return [(Path(args['aime24']['local_path']) / 'test-00000-of-00001.parquet', 30),
                (Path(args['aime25']['local_path']) / 'test.jsonl', 30)]
    if stage == 'olympiad':
        root = Path(args['olympiad_bench']['local_path']) / 'OlympiadBench'
        return [(root / subset / f'{subset}.parquet', count)
                for subset, count in OLYMPIAD_SUBSETS.items()]
    return [(Path(args['gpqa_diamond']['local_path']) / 'train-00000-of-00001.parquet', 198)]


def prepare_view(model, view):
    model, view = Path(model).expanduser(), Path(view).expanduser()
    require(model.is_absolute() and view.is_absolute(), 'Use absolute local model and view paths; HF identifiers are unsupported')
    source = model.resolve(strict=True)
    require(source.is_dir() and source.name == view.name == 'Llama200', 'Source and view must retain the model basename Llama200')
    require(not view.is_symlink(), 'Model view directory cannot be a symlink')
    destination = view.resolve()
    require(not destination.is_relative_to(source), 'Tokenizer view must be outside the original model directory')
    config = read_json(source / 'config.json')
    require(config.get('model_type') == 'llama' and 'LlamaForCausalLM' in config.get('architectures', []), 'Expected the local Llama200 model')
    require(config.get('rope_scaling', {}).get('factor') == 8.0 and config['rope_scaling'].get('rope_type') == 'llama3', 'Expected original Llama3 RoPE factor8; do not alter the model configuration')
    index = read_json(source / 'model.safetensors.index.json')
    shards = set(index['weight_map'].values())
    require(len(shards) == 4 and all(Path(name).name == name and name.endswith('.safetensors') for name in shards), 'Expected four local safetensors shards')
    names = shards | {'config.json', 'generation_config.json', 'model.safetensors.index.json',
                     'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'special_tokens_map.json', '.gitattributes'}
    require(len(names) == 12, 'Unexpected Llama200 model layout')
    for name in names:
        path = source / name
        require(path.is_file() and not path.is_symlink(), f'Original model file must be regular: {path}')
    original = read_json(source / 'tokenizer_config.json')
    require(original.get('tokenizer_class') == 'LlamaTokenizerFast', 'Unexpected original tokenizer class')
    adapted = {**original, 'tokenizer_class': 'PreTrainedTokenizerFast'}
    if destination.exists():
        require(destination.is_dir() and {p.name for p in destination.iterdir()} <= names, 'View contains unrelated files; refusing to overwrite them')
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(names):
        target = destination / name
        if name == 'tokenizer_config.json':
            if target.exists() or target.is_symlink():
                require(not target.is_symlink() and target.is_file() and read_json(target) == adapted,
                        'Existing tokenizer view differs; no files were overwritten')
            else:
                with target.open('x', encoding='utf-8') as handle:
                    json.dump(adapted, handle, ensure_ascii=False, indent=2)
                    handle.write('\n')
        elif target.exists() or target.is_symlink():
            require(target.is_symlink() and target.resolve(strict=True) == source / name, f'View must link the original file: {target}')
        else:
            target.symlink_to(source / name)
    require(sum(p.is_symlink() for p in destination.iterdir()) == 11, 'Expected eleven exact model symlinks')
    require(read_json(destination / 'config.json') == config, 'Model configuration changed')
    return destination


def check_data(stage):
    import pyarrow.parquet as parquet
    stages = STAGES if stage == 'all' else [stage]
    for name in stages:
        for path, expected in expected_data_files(name):
            require(path.is_file(), f'Missing local {name} dataset: {path}')
            if path.suffix == '.parquet':
                actual = parquet.ParquetFile(path).metadata.num_rows
            else:
                with path.open(encoding='utf-8') as handle:
                    actual = sum(1 for line in handle if line.strip() and isinstance(json.loads(line), dict))
            require(actual == expected, f'Dataset row count differs: {path}: {actual}, expected {expected}')


def check_tokenizer(model, view):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    from swift.model import get_model_processor
    from swift.template import get_template
    from vllm.tokenizers import get_tokenizer
    source = Path(model).resolve(strict=True)
    raw = Tokenizer.from_file(str(source / 'tokenizer.json'))
    tokenizer = AutoTokenizer.from_pretrained(str(view), use_fast=True, local_files_only=True, trust_remote_code=False)
    _, processor = get_model_processor(str(view), model_type='llama', load_model=False)
    loaders = {'AutoTokenizer': tokenizer, 'Swift': getattr(processor, 'tokenizer', processor),
               'vLLM': get_tokenizer(str(view), tokenizer_mode='auto', trust_remote_code=False)}
    native = read_json(source / 'tokenizer.json')
    config = read_json(source / 'config.json')
    for name, loaded_tokenizer in loaders.items():
        require(loaded_tokenizer.is_fast and loaded_tokenizer.get_vocab() == raw.get_vocab(), f'{name}: vocabulary differs from original JSON')
        loaded = json.loads(loaded_tokenizer.backend_tokenizer.to_str())
        require(native['decoder']['type'] == 'ByteLevel' and loaded['decoder'] == native['decoder']
                and loaded['pre_tokenizer'] == native['pre_tokenizer'], f'{name}: native ByteLevel tokenizer was replaced')
        for text in ['Okay, let us try.\nThis is a test.', '科学测试 α + β² = √2.\n[Fe(CN)6]  two spaces.', ' leading\tspace, café, e\u0301, 🙂. ']:
            ids = loaded_tokenizer.encode(text, add_special_tokens=False)
            require(ids == raw.encode(text, add_special_tokens=False).ids, f'{name}: IDs differ from original JSON')
            require(loaded_tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text,
                    f'{name}: spaces or Unicode lost during decoding')
        require(loaded_tokenizer.bos_token_id == config['bos_token_id'] == 128000
                and loaded_tokenizer.eos_token_id == config['eos_token_id'] == 128001, f'{name}: BOS or EOS mismatch')
    messages = [{'role': 'user', 'content': 'Compute 1 + 1.\n请作答。'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    require(prompt.endswith('<｜Assistant｜><think>\n'), 'Expected the native DeepSeek-R1 reasoning prompt')
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ids = encoded['input_ids'] if isinstance(encoded, Mapping) else encoded
    require(ids == raw.encode(prompt, add_special_tokens=False).ids, 'Native chat prompt token IDs differ')
    template = get_template(processor, template_type='deepseek_r1', enable_thinking=True)
    template.set_mode('transformers')
    require(template.encode({'messages': messages})['input_ids'] == ids, 'Swift deepseek_r1 prompt differs from native Jinja')
    require(processor.model_info.max_model_len == config['max_position_embeddings']
            and all(processor.model_info.rope_scaling[k] == v for k, v in config['rope_scaling'].items()),
            'Loader changed native context or RoPE configuration')
    if 'torch' in sys.modules:
        require(not sys.modules['torch'].cuda.is_initialized(), 'CPU preflight unexpectedly initialized CUDA')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'dataset-args'])
    parser.add_argument('--model')
    parser.add_argument('--view')
    parser.add_argument('--stage', required=True, choices=[*STAGES, 'all'])
    args = parser.parse_args()
    if args.command == 'dataset-args' and args.stage == 'all':
        parser.error('dataset-args requires one stage: aime, olympiad or gpqa')
    if args.command == 'prepare' and (not args.model or not args.view):
        parser.error('prepare requires --model and --view')
    try:
        if args.command == 'dataset-args':
            print(json.dumps(dataset_args(args.stage)))
            return 0
        with redirect_stdout(sys.stderr):
            check_data(args.stage)
            view = prepare_view(args.model, args.view)
            check_tokenizer(args.model, view)
        print(f'[ok] {args.stage}: local data, isolated tokenizer and original RoPE factor8 verified on CPU', file=sys.stderr)
        print(view)
    except Exception as exc:
        print(f'[error] {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
