# Llama200 三项评测

统一入口：`/data/chenyang/opdswift/llama200_eval/eval_llama200.sh`。默认使用现有 `swift` 环境、原模型 `/data/chenyang/models/Llama200` 和 GPU `1,4,5,7`。

```bash
cd /data/chenyang/opdswift/llama200_eval

# 先做 CPU 检查并显示实际命令，不启动模型推理。
bash eval_llama200.sh all --dry-run

# 顺序运行三项；也可以将 all 换为 aime、olympiad 或 gpqa。
bash eval_llama200.sh all
```

每次运行使用新目录：`/data/chenyang/opdswift/llama200_eval/test_results/日期_时间_进程号/{aime,olympiad,gpqa}/`。其中 `swift.log` 是运行日志，`command.sh` 保存实际命令，`native/` 保留 EvalScope 的完整预测、评分和报告，`verification.json` 是 CPU 核验结果。Swift 的辅助汇总也保存在本阶段的 `result/` 下，不受启动终端的工作目录影响。只有当前阶段成功并通过核验，才进入下一阶段。启动时会显示本次结果目录。

从其他工作目录运行也可以，脚本按自身位置查找辅助文件、兼容视图和结果目录。移动整个 `llama200_eval` 文件夹时，这些位置会自动跟着移动，不依赖原来的 `examples/train` 目录。

路径集中在入口顶部配置。默认 `PROJECT_DIR` 是 `llama200_eval` 的父目录，当前为 `/data/chenyang/opdswift`：

| 变量 | 当前默认位置 |
|---|---|
| `AIME_DATA_DIR` | `/data/chenyang/opdswift/data/data/aime_datasets` |
| `OLYMPIAD_DATA_DIR` | `/data/chenyang/opdswift/data/eval_datasets/olympiad_bench` |
| `GPQA_DATA_DIR` | `/data/chenyang/opdswift/data/data/gpqa_datasets` |

`support.py` 根据这些变量统一生成数据预检和 Swift 的数据参数，以后改数据位置只需改入口顶部或设置环境变量，无需再修改两份路径。AIME 根目录下应有 `aime24_local` 和 `aime25_local`；GPQA 根目录下应有 `gpqa_diamond_local`；Olympiad 根目录下应有 `OlympiadBench`。

可指定输出目录和 GPU，例如：

```bash
CUDA_VISIBLE_DEVICES=1,4,5,7 \
OUTPUT_DIR=/data/chenyang/opdswift/llama200_eval/test_results/my_new_run \
bash eval_llama200.sh gpqa
```

`OUTPUT_DIR` 必须尚不存在，避免混入旧结果。其他可选环境变量有 `MODEL_PATH`（这份本地 Llama200 模型）、`PYTHON`、`PROJECT_DIR`、上述三个数据目录、`PORT`（默认 18191）和 `HARD_TIMEOUT`（每项默认 172800 秒，即 48 小时）。路径建议使用绝对路径。GPU 列表长度决定 DP，TP 固定为 1；下表的客户端并发保持固定，不随卡数缩放。

| 阶段 | 数据 | 每题次数 | 输出上限 | 温度 | 客户端并发 | 每卡 max_num_seqs |
|---|---|---:|---:|---:|---:|---:|
| AIME | AIME24、AIME25，各 30 题 | 4 | 8192 | 0.6 | 16 | 32 |
| Olympiad | en_COMP 674、zh_CEE 1240、zh_COMP 408 题 | 4 | 8192 | 1.0 | 256 | 16 |
| GPQA | Diamond 198 题，零样本 | 4 | 32768 | 0.6 | 16 | 32 |

共同设置：seed=42、top_p=0.95、top_k=50、BF16、显存比例 0.7、`llama` 模型类型和 `deepseek_r1` thinking 模板。客户端并发包含排队请求；`max_num_seqs` 是上限，不表示实际同时活跃的数量。GPQA 的 32k 输出需要更多 KV 缓存，因此单独保留其资源设置。

文件分工：

- `eval_llama200.sh`：集中路径配置、三项参数、共用启动和退出处理。
- `support.py`：共用数据参数生成、本地数据及 CPU tokenizer 检查，建立 `.compat_model/Llama200`。
- `compat_plugins.py`：GPQA 保留方括号，以及 vLLM 0.23.0 的 dense DP 路由兼容处理。
- `verify_results.py`：检查预测与评分的对应关系、题数与索引、每题四条分组，并按 EvalScope 的层级四位小数归一化复算指标。

兼容视图通过软链接读取原模型，只有 `tokenizer_config.json` 的副本改为 `PreTrainedTokenizerFast`，避免 transformers 5.12 错误加载 ByteLevel tokenizer。权重与原模型文件不改。上下文只设置 `--vllm_max_model_len 40960`，保留模型的原 RoPE 配置。

整理时删除了重复启动代码、无用的单档长度循环、AIME26 配置、旧模型注释，以及绑定历史缓存和控制目录的一次性恢复流程。脚本遇到失败会停止并保留结果，不自动全量重跑或复用缓存。退出只清理本次创建的进程组，不清理其他任务。

评分沿用原 EvalScope 规则。

移植到其他机器时，复制本目录中的四个代码文件和本说明，并准备模型、数据及评测环境；`.compat_model` 中是绝对软链接，应由目标机器重新建立。当前已验证的核心版本为 ms-swift 4.3.1、EvalScope 1.8.1、vLLM 0.23.0、transformers 5.12.1。只同步 OPD 项目不会自动包含这套 Python 环境。`PYTHON`、模型路径和 GPU 编号仍需与目标机器匹配，先用 `--dry-run` 检查。

