# 量化测试实验区

**说明**：这里是简化的量化测试流程，用于实验和验证。

## 用途

- **快速实验**：用于测试新的量化配置、recipe 或模型变换方法
- **手动验证**：每一步需要手动执行，方便调试和观察中间结果
- **适配验证**：在将配置添加到 `pipline` 进行自动化测试之前，先在这里验证

## 使用流程

1. **修改配置**：编辑对应的 Python 脚本或 YAML 配置文件
2. **手动执行**：按顺序执行各个步骤
3. **验证结果**：检查输出是否符合预期
4. **集成到 pipeline**：验证通过后，将配置复制到 `pipline/` 目录进行自动化测试

## 测试脚本

| 脚本 | 功能 |
|------|------|
| `1-trans.py` | 模型转换（如 SpinQuant 预平滑） |
| `2-quant.py` | 模型量化（GPTQ/RTN 等） |
| `3-eval.py` | 模型评测（MMLU/GSM8K） |
| `serve_float.sh` | 启动 FP16 模型服务（基线对比） |

## 实验数据

### 主流程评测
| 模型版本 | MMLU | GSM8K | 备注 |
| :--- | :---: | :---: | :--- |
| Qwen3-8B (Base) | 0.8321 | 0.8809 | 原始 FP16 模型 |
| Qwen3-8B-R1R2 (Smooth) | 0.8428 | 0.8887 | FP32 预平滑处理，等价变换 |
| Qwen3-8B-R1R2-gptq-W8A8-static | 0.7598 | 0.7891 | W8A8 静态量化，有精度损失 |

#### 处理细节
1. **转换阶段 (1-trans)**：FP32 权重 + 等价变换
   - 将模型从 FP16 转换为 FP32
   - 通过旋转平滑和权重等效变换处理激活值
   - 理论上数学等价，无精度损失
   - 甚至可能因激活数值稳定性提升而略有改善

2. **量化阶段 (2-quant)**：W8A8 静态量化
   - 权重量化：8-bit 整数，损失较小
   - 激活量化：8-bit 静态量化，损失较大
   - 这是精度损失的主要来源

### 消融实验与分析
通过以下额外测评，可以发现影响量化精度的关键因素：

- **激活量化是瓶颈**：在 Weight-only (W8) 模式下，无论 RTN 还是 GPTQ 损失均极小，激活量化误差主因是 down_proj 层激活的 8-bit 静态量化造成的
- **动态量化显著提升**：将激活量化由 `static per-tensor` 转为 `dynamic per-token` 后，精度接近 Weight-only 效果
- **关键敏感层**：`down_proj` 层对量化误差最为敏感。排除该层的权重与激活量化后，模型指标几乎恢复至无损水平

| 模型版本 | MMLU | GSM8K |
| :--- | :---: | :---: |
| Qwen3-8B-R1R2-gptq-W8-weightonly | 0.8301 | 0.8672 |
| Qwen3-8B-R1R2-rtn-W8-weightonly | 0.8301 | 0.8691 |
| Qwen3-8B-R1R2-gptq-W8A8-dynamic | 0.8359 | 0.8652 |
| Qwen3-8B-R1R2-gptq-ignore-down-W8A8-static | 0.8438 | 0.8770 |

## 额外实验

`extra/` 目录包含更多实验脚本：
- `2-quant-weightonly-gptq.py` - GPTQ 权重量化
- `2-quant-weightonly-rtn.py` - RTN 权重量化
- `2-quant-dynamic.py` - 动态激活量化
- `2-quant-ignore-down.py` - 忽略 down_proj 层
- `3-eval-extra.py` - 额外评测脚本
- `clean_model.py` - 清理模型配置（移除 quantization_params防止vllm载入报错）

## 自动化测试

实验验证通过后，可以使用 `run_pipeline.py` 进行自动化批量测试，配置文件位于 `pipline/` 目录。




