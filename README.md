# Qwen3-8B 量化评测自动化流程

Qwen3-8B 模型从转换（SpinQuant）→ 量化（RTN/GPTQ）→ 评测的完整自动化流程。

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行流程

**方式1：使用默认配置**
运行pipline中文件夹内的yaml配置
```bash
python run_pipeline.py
```

**方式2：指定配置文件**
```bash
python run_pipeline.py \
  --trans-config pipline/1-trans/spinquant-r1r2.yaml \
  --quant-config pipline/2-quant/rtn-w8a8-dynamic.yaml \
  --eval-config pipline/3-eval/mmlu_gsm8k.yaml
```

**方式3：运行示例配置**
```bash
./run_example.sh
```

## 评测配置

`pipline/3-eval/mmlu_gsm8k.yaml`:
```yaml
dataset_name: mmlu_gsm8k_128
base_model: ./data/models/Qwen/Qwen3-8B  # 模型路径
gpu_resources:
  devices: [0,1,2,3]
  pipeline_parallel_size: 1  # PP=1时每个任务用1个GPU，PP=2时用2个GPU
```

## 目录结构

```
pipline/
├── 1-trans/          # 转换配置（SpinQuant等）
├── 2-quant/          # 量化配置（RTN/GPTQ）
│   └── example/      # 示例配置
└── 3-eval/           # 评测配置（MMLU/GSM8K）

data/                 # 模型和数据集文件
results/              # CSV结果汇总
logs/                 # 临时脚本和日志
outputs/              # 评测报告
```

## 输出结果

```bash
# CSV结果
results/results_{dataset_name}.csv

# 评测报告
outputs/{dataset_name}/reports/{model_name}/
```

## 特性

- ✅ 自动跳过已存在的模型和结果
- ✅ 量化和评测任务并行执行，一键完成复杂量化测评流程
- ✅ 临时py脚本保存到logs目录，可debug
- ✅ GPU自动并行分配（按pipeline_parallel_size, R4与tensor_parallel不兼容）