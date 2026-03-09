# 仓库功能与 LoongFlow 对比简版

## 1. 本仓库是做什么的

本仓库是一个面向时间序列数据分析与预测建模的多智能体代码仓。

它包含两套主要实现：

### 根目录主流程

面向基础时序任务：

- 数据读取
- 统计特性分析
- 模型选择
- 模型训练
- 结果集成
- 总结报告

核心调用链：

```text
User Query
  -> Orchestrator
  -> agent_loop
  -> data_reading
  -> data_analysis
  -> model_selection
  -> model_training
  -> result_integration
  -> summary
```

### `5_DGagent`

这是本仓库里更完整的数据分析智能体版本，特点是：

- 有主智能体做任务规划
- 有多个 subagent 分工执行
- 有全局 state
- 有 teams 机制
- 支持人工确认后再执行 LLM 生成代码

核心调用链：

```text
User Query
  -> DGOrchestrator
  -> teams + subagent plan
  -> agent_loop
  -> data_reading
  -> data_analysis
  -> model_selection
  -> model_training
  -> model_integration
  -> summary
```

## 2. LoongFlow `ml_agent` 的核心架构

LoongFlow `ml_agent` 不是简单的单轮 agent，而是一套面向机器学习任务的演化式框架。

其核心模块包括：

- planner
- evocoder
- executor
- evaluator
- summary
- database / checkpoint

典型执行链：

```text
task input
  -> planner
  -> code generation
  -> executor
  -> evaluator
  -> summary
  -> next iteration
```

它的重点不是“一次完成流程”，而是：

- 生成候选方案
- 执行候选方案
- 评估结果
- 保留更优方案
- 继续迭代优化

## 3. LoongFlow 是否有主智能体

有。

但它的主角色更像“主控运行器”，而不是普通的对话式 orchestrator。

可以把 `ml_evolve_agent.py` 对应的顶层 `MLAgent` 看作 LoongFlow `ml_agent` 的主智能体或主控层。

它负责：

- 初始化任务
- 注册 planner / executor / summary / evaluator
- 控制 iteration
- 管理日志、数据库、checkpoint 和输出目录

所以结论是：

- LoongFlow 有主智能体
- 但这个主智能体更偏 evolution controller
- `5_DGagent` 的主智能体更偏 task orchestrator

## 4. `5_DGagent` 可以借鉴 LoongFlow 的地方

### 4.1 evaluator 闭环

当前 `5_DGagent` 更像一次性流程，后续可以引入：

```text
model_selection
  -> model_training
  -> evaluator
  -> 不满足目标则重新选择/训练
```

意义：

- 不再只跑一轮
- 可自动优化模型和参数

### 4.2 iteration 机制

LoongFlow 每轮都有独立输出。

`5_DGagent` 可以增加：

- iteration id
- 每轮输出目录
- 每轮模型结果记录
- 每轮总结

意义：

- 可复盘
- 可恢复
- 可追踪

### 4.3 候选方案池

LoongFlow 强调多候选解。

`5_DGagent` 可以借鉴：

- 保存多组模型候选
- 保存多组参数
- 做 top-k 比较
- 对 top-k 再做集成

### 4.4 更细粒度的 ML stage

目前 `5_DGagent` 可以进一步拆出：

- preprocess
- feature_engineering
- split_strategy
- train_and_predict
- evaluator
- ensemble_strategy

意义：

- 结构更清晰
- 优化更容易
- 更接近真实 ML pipeline

## 5. 两者最大的区别

### LoongFlow `ml_agent`

更强在：

- 自动试错
- 评估驱动优化
- 演化搜索
- 适合 Kaggle / MLE-Bench / AutoML

### `5_DGagent`

更强在：

- 流程可控
- 子智能体职责清晰
- 状态共享明确
- 支持人工审批
- 更适合业务型数据分析场景

## 6. 一句话总结

- LoongFlow 更像“自动优化型 ML agent”
- `5_DGagent` 更像“可控编排型数据分析 agent”

如果后续要增强 `5_DGagent`，最值得借鉴 LoongFlow 的是：

- evaluator
- iteration
- 候选方案池
- 更细粒度阶段拆分

而不一定要整体照搬它的演化框架。
