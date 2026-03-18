# Lessons

- 一次性数据产物不要放在 `iteration_*` 目录下。像 `data_reading`、`data_formatter`、`split_strategy` 这类只需执行一次的阶段，应统一写入 `output/<task-id>/data/`，避免多轮迭代重复生成和路径歧义。
- 不要把 `final_summary.md` 的生成绑死在 `summary` 步骤出现在 plan 里。任务级总结属于收尾行为，应该在 orchestrator 结束时统一兜底生成。
- 汇总类输出不要直接 `json.dumps(state_payload)`。state 里经常会有 `Timestamp`、numpy 标量等对象，summary 场景应统一使用可回退到字符串的序列化方式。
