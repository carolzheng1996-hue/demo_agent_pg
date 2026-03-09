# loongflow_DGagent

`loongflow_DGagent` 是在现有 `5_DGagent` 之外新增的一套 LoongFlow 风格数据分析智能体实现，不会覆盖原有 `5_DGagent` 代码。

## 目标

这一版重点参考 LoongFlow `ml_agent` 的设计，做了三类增强：

- 细化数据科学阶段：新增 `feature_engineering`、`split_strategy`、`preprocess`、`evaluator`
- 引入 evaluator 驱动的 iteration 闭环：建模任务不只训练一次，自动做多轮参数/模型尝试，最多 10 轮
- 增加明确的执行工件结构：每轮都会输出独立目录和 `plan.json` / `summary.md`

## 目录结构

- `main.py`: CLI 入口
- `orchestrator.py`: 主智能体与迭代控制器
- `agent_loop.py`: subagent 执行循环
- `state.py`: 全局状态
- `task_manager.py`: 任务清单
- `teams.py`: team 到 subagent 的映射
- `LLM/`: 复用现有项目 LLM 接入代码，使用 `get_llm().invoke(...)`
- `tools/`: 分析、评估、沙箱、工件输出工具
- `subagents/`: 各阶段子智能体

## 子智能体阶段

### 非建模前置阶段

- `data_reading`
- `data_analysis`
- `feature_engineering`
- `split_strategy`
- `preprocess`

### 建模迭代阶段

- `model_selection`
- `model_training`
- `model_integration`
- `evaluator`
- `summary`

## 调用链

### 统计分析任务

```text
User Query
  -> DGOrchestrator
  -> data_reading
  -> data_analysis
  -> feature_engineering
  -> summary
```

### 建模任务

```text
User Query
  -> DGOrchestrator
  -> data_reading
  -> data_analysis
  -> feature_engineering
  -> split_strategy
  -> preprocess
  -> iteration loop (max 10)
       -> model_selection
       -> model_training
       -> model_integration
       -> evaluator
       -> summary
```

## 人工审批与自动优化并存

这一版保留了 `data_reading` 的人工审批机制：

- 先生成数据读取/标准化代码
- 用户确认后再执行
- 只有审批通过，后续自动优化流程才会继续

这意味着：

- 自动化发生在审批后的稳定建模阶段
- 不会绕过人工直接执行未经审查的 LLM 代码

## Iteration 机制

建模任务下：

- `evaluator` 会比较当前轮的最佳单模型和集成结果
- 如果仍有改进空间，则进入下一轮
- 默认最多 3 轮，可通过 `--max-iterations` 提高
- 最大不超过 10 轮

每轮都会记录：

- 候选模型与参数
- 训练指标
- 集成结果
- evaluator 判断
- 该轮 summary

## 工件目录结构

输出目录参考 LoongFlow 风格：

```text
loongflow_DGagent/output/
  <task-id>/
    final_summary.md
    iteration_history.json
    <iteration-id>/
      plan.json
      data_reading/
      data_analysis/
      feature_engineering/
      split_strategy/
      preprocess/
      model_selection/
      model_training/
      model_integration/
      evaluator/
      summary/
      summary.md
```

## 使用方式

### 1. 仅生成读取代码并等待审批

```bash
python run_loongflow_dgagent.py \
  --query "针对 ETTh1 数据做统计特性分析" \
  --dataset-path data/ETTh1.csv
```

### 2. 审批后执行统计分析

```bash
python run_loongflow_dgagent.py \
  --query "针对 ETTh1 数据做统计特性分析" \
  --dataset-path data/ETTh1.csv \
  --approve-generated-code
```

### 3. 执行多轮建模优化

```bash
python run_loongflow_dgagent.py \
  --query "针对 ETTh1 数据构建一个时序预测模型" \
  --dataset-path data/ETTh1.csv \
  --approve-generated-code \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --test-ratio 0.2 \
  --max-iterations 5
```

## 与 `5_DGagent` 的关系

- `5_DGagent` 保持原样不动
- `loongflow_DGagent` 是一套新的试验性增强版本
- 如果后续验证效果稳定，再考虑是否把其中一部分能力回迁到原架构
