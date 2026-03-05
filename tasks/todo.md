# Todo

- [x] 按指定结构创建根模块与目录
- [x] 实现 GlobalState 共享状态总线（含 JSON 持久化）
- [x] 实现 TaskManager（任务追踪 + JSON 持久化）
- [x] 实现 19 个 tools 模块函数
- [x] 实现 6 个 subagent 并注册到 SUBAGENT_REGISTRY
- [x] 实现 orchestrator 意图识别与计划编排
- [x] 实现 agent_loop + run_subagent 核心循环
- [x] 实现 main.py CLI 入口
- [x] 更新 requirements 与 README
- [x] 更新 todo review

## 2026-03-05 OpenAI 接入改造

- [x] 新增统一 LLM 客户端，支持 OpenAI API Key / Base URL / Model
- [x] 将 orchestrator 意图识别接入 LLM（无 key 时回退规则）
- [x] 将模型选择与总结报告接入 LLM（无 key 时回退原逻辑）
- [x] 更新 CLI 参数与 README，说明如何以 OpenAI 模式运行
- [x] 更新 todo review，记录本次验证结论

## 2026-03-05 第三方平台 API 兼容修正

- [x] 支持 `API_KEY/API_BASE` 与 `OPENAI_API_BASE` 等常见环境变量别名
- [x] CLI 增加 `--api-base` 参数别名，兼容用户口径
- [x] LLM 启用条件改为“有 key 即可”，不强依赖 provider 值
- [x] 更新 README 为 OpenAI 兼容接口说明
- [x] 更新 lessons，记录本次纠正规则

## 2026-03-05 .env 配置改造

- [x] 引入 `.env` 自动加载机制（项目根目录）
- [x] 增加 `.env.example` 模板并忽略真实 `.env`
- [x] 更新 README，改为 `.env` 配置示例
- [x] 更新 lessons，记录本次纠正规则
- [x] 更新 todo review，记录改造结论

# Review

- 目录结构已按要求创建。
- 主流程采用：`orchestrator -> agent_loop -> subagents`。
- `GlobalState` 与 `TaskManager` 均支持 JSON 持久化到 `.tasks/`。
- 19 个 tools 已按分组实现，6 个 subagent 已接入注册表。
- 未执行 Python 脚本验证（按仓库约定由你执行）。
- 新增 `llm_client.py`，统一封装 OpenAI 文本/JSON 调用与异常回退。
- `orchestrator`、`model_selection`、`summary` 已接入 LLM，缺省自动回退本地规则。
- `main.py` 新增 `--api-provider/--api-key/--api-base-url/--api-model` 参数。
- `config.py` 支持 `OPENAI_API_KEY/OPENAI_BASE_URL` 以及 CLI 覆盖。
- `README.md` 和 `requirements.txt` 已更新。
- 未运行 Python 级验证（遵循仓库约定，需你本地执行验证命令）。
- 兼容第三方平台口径：支持 `API_KEY/API_BASE`、`--api-base`，并放宽 provider 约束。
- 已支持启动时自动读取项目根目录 `.env`（`config.py`）。
- 新增 `.env.example` 模板与 `.gitignore`（忽略真实 `.env`）。
- 新增 `.env` 占位文件，直接替换 key/base/model 即可运行。
