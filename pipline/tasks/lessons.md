# Lessons

- 一次性数据产物不要放在 `iteration_*` 目录下。像 `data_reading`、`data_formatter`、`split_strategy` 这类只需执行一次的阶段，应统一写入 `output/<task-id>/data/`，避免多轮迭代重复生成和路径歧义。
- 不要把 `final_summary.md` 的生成绑死在 `summary` 步骤出现在 plan 里。任务级总结属于收尾行为，应该在 orchestrator 结束时统一兜底生成。
- 汇总类输出不要直接 `json.dumps(state_payload)`。state 里经常会有 `Timestamp`、numpy 标量等对象，summary 场景应统一使用可回退到字符串的序列化方式。
- 当用户明确要求把运行态文件一并提交时，不要按默认习惯排除 `.state/`、`output/`、示例数据目录；应按用户要求强制纳入版本控制。
- 默认参数不要散落在 CLI、Web 和子模块里各写一套。像切分比例、窗口长度、标准化策略这类运行参数，应统一放在配置文件中，再写入 state 供全链路消费。
- 面向 agent 开发时，优先提供单一 JSON 运行配置入口。让 CLI、Web、state 初始化都先读取同一份 `config_all.json`，比只在 Python 常量里维护默认值更稳定，也更方便自动化改参与复现。
- 建立统一配置入口后，要继续清理各 subagent 里的旧硬编码 fallback。否则表面上接了 `config_all.json`，实际运行仍可能偷偷回退到另一套默认值，导致配置失真。
- `summary` 既可能是数据处理任务的收尾，也可能是建模任务的收尾。plan 生成不能因为 `end_stage=summary` 就默认经过所有模型阶段，必须结合任务意图选择阶段链。
- 配置、CLI 和前端表单必须三方联动。若某个字段只在特定切分方式或阶段范围下有效，前端应折叠它，CLI 应提供可逆覆盖项，后端则只保留一套实际生效逻辑。
- 当输出工件结构调整后，任务详情页的读取逻辑必须同步调整。若后端已取消每阶段 `result.json`，前端/接口层就不能继续假设这些文件存在，而应改读任务级 plan、summary 和真实数据工件。
- 清理默认值时不要只改 subagent。像 orchestrator、Web payload 归一化、任务详情恢复逻辑这类外围层也要一起扫描，否则仍会残留隐式回退路径。
- 对 DS 这类序列型中间产物，长度补齐和 NaN 填补应放在“进入 split/feature/model 之前的统一入口”处理，而不是散落到下游多个步骤。这样更容易保证切分、特征工程和训练看到的是同一份清洗后数据。
- 新增运行参数时，必须同时检查四处：`config_all.json`、CLI、Web payload 归一化、前端表单。只改配置和 subagent 不够，最终会形成“后端支持、前端不可配”或“文档写了、命令行不能传”的断层。
- 只要某个 `start_stage` 在 CLI/UI 中暴露，就必须验证它的 bootstrap 能否独立恢复所需输入。否则应当收回该入口，或补齐任务根目录到中间工件的恢复逻辑。
