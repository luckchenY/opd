# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ms-swift (ModelScope SWIFT) is a production-grade LLM fine-tuning, inference, and deployment framework. It supports 600+ text LLMs and 400+ multimodal models with training paradigms including SFT, DPO, GRPO, KTO, CPO, SimPO, ORPO, and GKD. Package name is `ms_swift`, installed as `ms-swift`.

## Common Commands

### Install
```bash
pip install -e .                          # Editable install from source
pip install -r requirements/framework.txt # Core dependencies
pip install -r requirements/tests.txt     # Dev/test dependencies
```

### Lint & Format
```bash
pre-commit run --all-files    # Run all pre-commit hooks (flake8, isort, yapf, etc.)
```

Style rules: line length 120, yapf (PEP8-based), isort, flake8 (ignores F401,F403,F405,F821,W503,E251,W504,E126).

### Test
```bash
python tests/run.py --subprocess                    # Run all tests with subprocess isolation
python tests/run.py --case_pattern "test_sft*"      # Run tests matching a pattern
```

Tests use Python's `unittest` framework. Test files are organized under `tests/` by feature area (train/, infer/, deploy/, export/, eval/, tuners/, models/).

### Build
```bash
make whl    # Build wheel
make docs   # Build documentation
```

## CLI Entry Points

The main CLI is `swift` with subcommands routed through `swift/cli/main.py`:

| Command | Purpose | Arguments Class |
|---------|---------|-----------------|
| `swift sft` | Supervised fine-tuning | `SftArguments` |
| `swift pt` | Pre-training | `PretrainArguments` |
| `swift rlhf` | RLHF training (DPO, GRPO, etc.) | `RLHFArguments` |
| `swift infer` | Inference | `InferArguments` |
| `swift deploy` | Model deployment (OpenAI-compatible API) | `DeployArguments` |
| `swift eval` | Evaluation | `EvalArguments` |
| `swift export` | Model export/quantization | `ExportArguments` |
| `swift web-ui` | Gradio web interface | `WebUIArguments` |
| `swift sample` / `swift rollout` | Sampling for GRPO | `SamplingArguments` / `RolloutArguments` |
| `swift merge-lora` | Merge LoRA weights into base model | - |

There's also a `megatron` CLI for Megatron-LM distributed training.

## Architecture

### Command Flow
```
swift <cmd> [args] → cli/main.py:cli_main() → ROUTE_MAPPING lookup →
  cli/<cmd>.py → pipelines/<area>/<cmd>.py (SwiftPipeline subclass) → run()
```

### Key Layers

**Arguments** (`swift/arguments/`): Hierarchical dataclasses. `BaseArguments` inherits from multiple mixins (ModelArguments, DataArguments, TemplateArguments, QuantizeArguments, etc.). `SftArguments` and `RLHFArguments` extend it for their respective training types.

**Pipelines** (`swift/pipelines/`): Orchestration layer. `SwiftPipeline` base class with abstract `run()`. Concrete implementations: `SwiftSft`, `SwiftPretrain`, `SwiftRLHF`, `SwiftInfer`, `SwiftDeploy`, `SwiftEval`. Training pipelines use `TunerMixin` for model loading and tuner setup.

**Models** (`swift/model/`): Registry-based model loading. `register.py` registers 600+ architectures with metadata (context length, template type, capabilities). `patcher.py` applies model-specific modifications.

**Templates** (`swift/template/`): Chat format templates per model family. Converts prompts into model-specific token sequences. Registered via decorator pattern.

**Trainers** (`swift/trainers/`): Extends HuggingFace Trainer. `TrainerFactory` in `trainer_factory.py` dynamically selects the right trainer class (Seq2SeqTrainer, EmbeddingTrainer, RerankerTrainer, etc.). `mixin.py` contains core training logic (~60KB).

**Tuners** (`swift/tuners/`): Fine-tuning methods (LoRA, QLoRA, DoRA, Adapter, LongLoRA, etc.). Registry via `SWIFT_MAPPING`. Integrates with PEFT library.

**Datasets** (`swift/dataset/`): 150+ registered datasets. `loader.py` handles ModelScope/HuggingFace/local loading. Preprocessors in `preprocessor/` handle text, image, audio, video.

**Inference Engines** (`swift/infer_engine/`): Four backends sharing `InferEngine` base: Transformers, vLLM, SGLang, LmDeploy.

**RLHF Trainers** (`swift/rlhf_trainers/`): Specialized trainers for alignment methods (DPO, KTO, CPO, SimPO, ORPO, GRPO, GKD). Built on top of the `trl` library.

### Patterns
- **Registry/decorator pattern** throughout: models, templates, datasets, tuners, losses all use registration decorators.
- **Lazy module loading** (`_LazyModule` in `__init__.py`) for efficient imports.
- **Mixin composition**: Arguments and pipelines compose behavior through multiple mixins rather than deep inheritance.

## Key Dependencies
- transformers (>=4.33), torch (>=2.0), modelscope (>=1.23)
- peft (>=0.11,<0.19), trl (>=0.15,<0.30), datasets (>=3.0,<4.0)
- Optional: vllm, sglang, lmdeploy, deepspeed, flash_attn, ray
