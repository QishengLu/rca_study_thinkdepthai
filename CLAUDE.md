# ThinkDepth.ai Deep Research（RCA 适配）

把 ThinkDepth.ai 的 Deep Research 框架用于微服务根因分析（RCA）：LLM 通过 SQL 查询单个 case 的遥测表，输出 CausalGraph JSON。

- 上游：https://github.com/thinkdepthai/Deep_Research @ 499c007，MIT（见 `LICENSE`）。
- 本目录新增：`agent_runner.py`、`model_factory.py`、`src/rca_tools.py`、`tests/`、`.env.*.example`、`uv.lock`。
- 相对上游的改动：`src/prompts.py` 增加 `rca_think_prompt`；`src/utils.py` 在没有 Tavily key 时不报错；`pyproject.toml` 增加 `duckdb`、`pytz`、`python-dotenv`；`src/research_agent.py` 增加 RCA 工具并换了模型名。`research_agent*`、`multi_agent_supervisor` 是上游的 Web 研究代码，harness 不运行。

## Harness 如何驱动

`python agent_runner.py [--verbose] [--log-file PATH]`，每个 case 一个进程。配置在 `harness/configs/agents/`：`thinkdepthai-qwen.yaml`（`prompts: rca_tables.yaml`）；`thinkdepthai-cmp.yaml` 是框架对比实验（父仓库 `docs/framework_comparison.md`）：对比用 prompt `rca_compare.yaml`，每个请求经过 harness 的录制网关，不设轮次或工具次数上限。模型由 `RCA_MODEL` 传入。

- stdin：JSON，含 `question`、`system_prompt`、`user_prompt`、`compress_system_prompt`、`compress_user_prompt`、`data_dir`。
- stdout 最后一行：JSON，含 `output`（CausalGraph JSON 字符串，无效时原样保留）、`graph_valid`、`trajectory`（OpenAI 消息格式）、`usage`、`run_status`、`stop_reason`（`finish`、`context_length`、`recursion_limit`）、`run_error`。harness 把其他键放进 meta。`trajectory` 只含图节点产生的消息，不含初始 user 消息。
- 流程：`START -> analyst_call -> (tool_node -> analyst_call)* -> compress_research -> END`，与上游 `research_agent.py` 拓扑相同，递归上限 300。
- 服务端因上下文长度拒绝请求（HTTP 400，vLLM 的 context length 报错）不是基础设施失败：Analyst 循环在那里结束，synthesis 用已收集的消息照常运行（`run_status=completed`、`stop_reason=context_length`）；synthesis 也被拒绝时，run 以 `failed`、空输出结束。递归上限同样记为失败的 run。退出码都是 0，harness 不重试。
- 每个请求带 `X-Cairn-Purpose` 头（Analyst 为 `agent`，synthesis 为 `synthesis`），网关据此区分辅助调用；其他端点忽略它。
- 工具：`think_tool`、`list_tables_in_directory`、`get_schema`、`query_parquet_files`，没有 `tavily_search`。system prompt 是 payload 的 `system_prompt` 加 `rca_think_prompt`。

## 环境

```bash
uv sync --frozen --python 3.12
cp .env.qwen.example .env.qwen      # 填 OPENAI_API_KEY；Claude 模型用 .env.sonnet.example
```

- 凭证文件：`RCA_ENV_FILE` 优先；否则 `RCA_MODEL` 以 `claude` 开头读 `.env.sonnet`，其他读 `.env.qwen`；文件不存在时退回 `.env`。已有的环境变量不被覆盖。
- 依赖 `_rca_shared`（`events`、`usage_tracker`）：默认在上一级目录；单独使用本仓库时，把其父目录设为 `RCA_SHARED_PARENT`。
- 其他变量：`RCA_EVENTS_DIR`（事件日志，实时写入该目录的 `events.jsonl`）、`RCA_MAX_RETRIES`（OpenAI 分支，默认 6）、`RCA_LLM_EXTRA_BODY`（JSON，透传）、`RCA_FORCE_ANTHROPIC=1`。
- 输出上限：`RCA_MAX_TOKENS`（Analyst，默认 32768）、`RCA_COMPRESS_MAX_TOKENS`（synthesis，默认 32000）。在 32768 长度的 vLLM 上必须调小（对比实验都是 4096）。
- 采样：`RCA_TEMPERATURE`（默认 0）、`RCA_TOP_P`、`RCA_TOP_K`、`RCA_PRESENCE_PENALTY`，未设置时不发送；`top_p`、`top_k` 放在 extra body 里（当前 langchain-openai 不发送 `ChatOpenAI(top_p=...)`）。

## 数据集与表

`data_dir` 是 prepared 沙箱的 `tables/`。`<name>.parquet` 或 `<name>/*.parquet` 都算一张表，分片按列名合并；模型只看到表名。框架不内置表名单，表由 `list_tables_in_directory` 发现。适配都在数据准备阶段完成，本目录不转换数据。不支持 cloud_opsbench：它回放录制的工具输出，没有 SQL 表。

| 数据集 | case 数 | 表 |
|---|---:|---|
| rcabench_500、rcabench、openrca2 | 500 / 1422 / 455 | `normal_*`、`abnormal_*` × logs、traces、metrics、metrics_histogram、metrics_sum，共 10 张 |
| openrca_bank | 136 | log_service、metric_app、metric_container、trace_span |
| openrca_market_cloudbed1、cloudbed2 | 70 / 78 | log_proxy、log_service、trace_span、metric_container、metric_mesh、metric_node、metric_runtime、metric_service |
| openrca_telecom | 51 | trace_span、metric_app、metric_container、metric_middleware、metric_node、metric_service |
| aiops2025 | 400 | logs、traces、apm_pod、apm_service、apm_namespace、infra_node、infra_pod、infra_tidb、infra_other（均为分片目录） |
| rca100、rca100_v1 | 103 / 57 | metrics、logs、traces、events、alerts、topology_entities、topology_edges |
| rcaeval | 735 | re1：metrics（长表：time、service、metric、value）；re2、re3：另有 logs，ob、tt 子集再加 traces（1 个 re2 tt case 没有 logs） |

- **基线**：只有 rcabench_500、rcabench、openrca2 有 `normal_*` 表，其他数据集用事件前的时段作对比。
- **OpenRCA**：每题只给问题日期（跨午夜时两天）的 CSV 转成的 Parquet，列名不变，时间为 UTC+8。**AIOps2025**：给该 UTC+8 自然日的遥测。
- **RCA100 列**：沿用原始列，只有 `topology.json` 被转成两张 `topology_*` 表。`traces.startTime`、`duration` 是 VARCHAR 存的纳秒值，比较前要 cast；`logs._time_` 是 RFC 3339 字符串（+08:00）；`metrics.time` 是微秒 BIGINT；`events` 的内容在 `eventId` 的 JSON 里。
- **RCA100 缺口**：`traces`、`logs` 是分段采样，大约每 5 分钟一段。告警窗口内没有 traces：t103；没有 logs：rca100 的 t085、t094。rca100_v1 中 `k8s.node`、`k8s.deployment`、`k8s.namespace`、`k8s.cluster` 指标的 `entity_name` 全为空，节点名只能从 logs、traces 或 events 取得。

## Smoke

假模型，不需要 key 和网络。检查图结构、路由、工具集、stdout 键集合和轨迹形状。不带参数时用临时 parquet，也可以传真实 case 的 tables 目录。

```bash
.venv/bin/python tests/smoke_fake_llm.py [tables目录 ...]
```

## 注意

- `data_dir` 下的每个 parquet 都会暴露给模型。不要指向原始 case 目录：例如 RCABench 的 `converted/` 含 `conclusion.parquet`，会被当成表。
- 工具结果超过约 5000 token（按 3 字符/token 估算）时，返回错误和建议的 LIMIT，不返回截断数据。`query_parquet_files` 的 `limit` 默认 10，在工具层截断，与 SQL 里的 LIMIT 无关。
- 含点号的列名在视图里改成下划线（`a.b` -> `a_b`）；`__tag__:_node_name_` 这类列名要加双引号。
