# hydrology_semantic_query

该场景通过 Cube REST API 加载公开语义目录并执行 `SemanticQuery`。现有水文语义模型位于 `cube/model/hydrology_model`，水文监测模型位于 `cube/model/hydrology_monitor_model`。

## 代码结构

场景代码按职责组织：

```text
hydrology_semantic_query/
├── models.py        # 目录、查询、结果和展示模型
├── runtime.py       # 服务依赖、状态、请求、错误和步骤事件
├── client.py        # Cube HTTP 客户端及 meta 目录解析
├── node.py          # 初始化、问题改写、ReAct 和最终结果节点
├── report.py        # 展示规划、渲染、报告和输出
├── tools/           # LangChain 工具、目录检索与语义查询实现
│   ├── tools.py
│   ├── common.py
│   ├── run_semantic_query.py
│   └── search_semantic_catalog.py
├── cube/            # Cube 部署、模型和脚本
├── graph.py         # LangGraph 节点与边的拓扑装配
└── tests/           # 场景行为、部署、集成和架构回归测试
```

依赖方向固定为 `graph/node → tools/tools.py → tools 内的查询实现 → scenario models/runtime`。工具实现不依赖场景图或节点；`graph.py` 只负责拓扑和递归限制。

连接外部 MySQL 的本地 Cube 可通过 `cube/start/start.sh` 一键启动，详见 [cube/start/README.md](cube/start/README.md)。

## Cube 基础模型生成

`cube/scripts/generate_cube_models.py` 使用 SQLAlchemy 读取 MySQL 中的表、View、字段、主键和外键，生成仅供审阅的 Cube 草稿。草稿固定输出至 `cube/generated/`，不会改写或加载当前 `cube/model/` 中的治理模型。

脚本优先使用 `--database-url`：

```bash
poetry run python -m app.agents.scenarios.hydrology_semantic_query.cube.scripts.generate_cube_models \
  --database-url 'mysql+pymysql://root:replace_me@127.0.0.1:3306/hydrology_local'\
  --table device_info \
  --table device_x_value\
  --table label_sensor\
  --table label\
  --table warn_state_info\
  --table water_warn_sensor_set\
  --table multifactor_sensor
```

也可以使用 `CUBEJS_DB_HOST`、`CUBEJS_DB_PORT`、`CUBEJS_DB_NAME`、`CUBEJS_DB_USER`、`CUBEJS_DB_PASS` 环境变量；当 Cube 变量未设置时，兼容 `MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_DATABASE`、`MYSQL_USER`、`MYSQL_PASSWORD`。

脚本不读取本地 Cube 模型范围，必须至少传入一个 `--table`。扩展数据库时先显式生成并审核草稿，再把需要自然语言查询的稳定业务粒度迁入正式 Cube；生成基础 Cube 不会同时生成 View。

反复传入 `--table` 可仅生成指定对象，显式指定时不应用排除模式：

```bash
poetry run python -m app.agents.scenarios.hydrology_semantic_query.cube.scripts.generate_cube_models \
  --table virtual_hydrological_monitoring \
  --table virtual_water_pumping
```

`cube/generated/model/cubes/` 包含每个数据库对象对应的私有 Cube，只自动生成 Dimension、`count` 和安全的单字段外键 Join。`cube/generated/join_candidates.yml` 记录按 `*_id`/`*Id` 命名推测的候选关联以及未入模的外键。脚本每次成功运行都会整体替换 `cube/generated/`；生成失败时保留上一次草稿。业务指标、中文语义、Segment、自定义 SQL 和 View 仍需人工治理。

Agent 运行时会自动读取场景根目录下的 `.env`，进程环境变量优先于该文件中的同名配置。该文件只管理 Agent 场景专属配置；`cube/start/.env` 继续独立管理 Cube 的 MySQL 连接配置。

运行时配置：

- `HYDROLOGY_SEMANTIC_QUERY_CUBE_URL`：Cube 服务地址、`/cubejs-api` 地址或已完整的 `/cubejs-api/v1` REST API 根地址。
- `HYDROLOGY_SEMANTIC_QUERY_CUBE_TOKEN`：Cube API Token；Cube 未启用认证时可留空。
- `HYDROLOGY_SEMANTIC_QUERY_TIMEOUT_SECONDS`：Cube HTTP 请求超时秒数。
- `HYDROLOGY_SEMANTIC_QUERY_CONTINUE_WAIT_RETRIES`：Cube 返回 continue wait 后的重试次数。
- `HYDROLOGY_SEMANTIC_QUERY_META_CACHE_TTL_SECONDS`：Cube 元数据缓存秒数。
- `HYDROLOGY_SEMANTIC_QUERY_MAX_AGENT_ITERATIONS`：ReAct Agent 最大决策轮数，默认为 `6`，允许 `3` 至 `12`。
- `HYDROLOGY_SEMANTIC_QUERY_MAX_RETRIES`：兼容旧配置保留，不再控制查询重试；检索、修正和重新执行统一受 Agent 决策轮数限制。
- `HYDROLOGY_SEMANTIC_QUERY_TIMEZONE`：IANA 时区，默认为 `Asia/Shanghai`。
- `HYDROLOGY_SEMANTIC_QUERY_MAX_ROWS` 和 `HYDROLOGY_SEMANTIC_QUERY_HARD_MAX_ROWS`：默认行数上限与硬上限。
- `HYDROLOGY_SEMANTIC_QUERY_CATALOG_STRATEGY`：上下文策略，可选 `full`、`vector` 或 `auto`，默认为 `auto`。
- `HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_MODEL`：本地 sentence-transformers 模型目录；留空或加载失败时通过同一接口使用词法检索。
- `HYDROLOGY_SEMANTIC_QUERY_MODEL_TOP_K`：第一阶段 View/Cube 候选模型数量，默认为 `5`。
- `HYDROLOGY_SEMANTIC_QUERY_CONTEXT_TOP_K`：第二阶段候选模型内成员和目录组命中数量，默认为 `20`。
- `HYDROLOGY_SEMANTIC_QUERY_VECTOR_INDEX_PATH`：语义目录向量索引的 SQLite 缓存路径；默认为场景内的 `cube/cache/semantic-catalog-vectors.sqlite3`，向量检索时不能为空。
- `HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_BATCH_SIZE`：目录文档嵌入批次大小，默认为 `32`。
- `HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_CONCURRENCY`：目录文档嵌入并发数，默认为 `3`。
- `HYDROLOGY_SEMANTIC_QUERY_AUTO_FULL_CONTEXT_MAX_CHARS`：`auto` 模式直接提供完整可访问目录的字符阈值，默认为 `30000`。

运行时主图为：初始化 → ReAct Agent ⇄ 工具 → 最终结果。初始化只加载 Cube `/meta`、解析请求约束并在需要时把多轮追问改写为独立问题，不会自动检索或生成查询。Agent 每轮只调用一个工具，并根据 Observation 决定继续检索、修正并重新执行，或结束查询。查询成功后，最终节点固定调用专用报告模型生成 Markdown 报告；报告失败时保留查询摘要并记录警告。

`search_semantic_catalog` 按问题检索相关 Cube、View 和 member。数据查询首次执行前必须至少成功调用一次；重复调用会合并语义上下文和检索轨迹。`run_semantic_query` 接受 `SemanticQuery`，统一执行完整可访问目录校验、Cube `/sql` 编译预检和 Cube `/load` 执行，不接受原始 SQL。成功 Observation 只向 Agent 提供规范化查询、列和最多 50 行结果预览，完整结果与编译 SQL 保存在最终结构化结果中。

目录检索只构造 Agent 上下文，不绑定成员、不决定最终模型路由，也不形成成员白名单。存在会话历史时，初始化阶段会把当前追问改写为独立问题；改写失败时回退原问题。`auto` 模式下，完整可访问目录不超过 30000 字符时直接提供完整目录；大目录先检索最相关的 View/Cube，再只在候选模型内检索 member 和 view folder。成员命中会补充父模型概览，Cube 命中会补充候选模型范围内的连通分量信息。向量组件不可用时使用相同的两阶段词法检索；目录向量必须在 Cube 部署阶段预先写入 SQLite，查询时只加载签名匹配的索引，不会现场重建。

修改正式 Cube 模型后，通过 `cube/start/start.sh` 重启 Cube 并自动检查语义目录索引。目录签名变化时脚本会在 Agent 启动前完成重建，失败则阻止部署继续。也可以独立执行或强制重建：

```bash
poetry run python \
  app/agents/scenarios/hydrology_semantic_query/cube/start/rebuild_vector_index.py
poetry run python \
  app/agents/scenarios/hydrology_semantic_query/cube/start/rebuild_vector_index.py \
  --force
```

索引缺失、损坏、版本过期或与最新 `/meta` 不一致时，向量目录检索会返回部署错误。若 Agent 已经运行，Cube 模型更新和索引重建完成后仍需重启 Agent，以清除旧的 `/meta` 与内存 Retriever。

metadata filters 先产生完整可访问目录，校验始终基于该目录，而不是检索命中项。`SemanticQuery` 使用 `query_mode` 与 `models` 表达路由，二者不会发送给 Cube `/sql` 或 `/load`。View 模式必须恰好一个 View；Cube 模式允许一个 Cube 或同一非空 `connectedComponent` 中的二至四个 Cube。请求 metadata 可使用 `catalog_mode` 和 `catalog_metadata_filters` 覆盖上下文策略及按 `model_name`、`model_type`、`title`、`business_domain` 过滤的可访问范围，不能绕过模型边界。

工具参数、本地 Validator 和 Cube 返回的错误会以结构化 Observation 反馈。未知模型、未知成员、成员范围或类型、连通性及可修正的 Cube 400 错误允许 Agent 自主检索和重试；单次检索上限为 40，并与已有上下文合并。认证、网络、超时和系统错误会终止工具循环；空结果直接返回 `no_data`，不扩大目录或改变过滤条件。到达最大决策轮数时，最后一轮禁用工具并要求 Agent 直接总结。

公开语义目录完全来自 Cube `/meta`；当前模型包含三个高频业务 View 与 108 个公开 `base_*` Cube，`role_label_parent` 保持私有。原有七个 Cube 继续承担人工治理的高频业务图，其余 101 个 Cube 按数据库中有数据的物理表粒度提供长尾查询；未声明主键的表不参与 Join。基础 Cube 只公开稳定业务字段，寄存器、脚本、文件路径、函数参数、凭据和审计字段继续隐藏。Python 不再扫描本地模型文件，非空 `connectedComponent` 用于多个 Cube 的查询前粗校验，独立 Cube 不要求该字段；精确 Join Path 由 Cube `/sql` 按部署模型解析。View 默认投影和固定过滤字段角色分别由模型与成员 `meta` 通过接口提供。业务范围、粒度、关系和隐藏成员见 [语义模型人工治理指南](cube/BUSINESS_VIEW_GOVERNANCE.md)。

可使用真实 Cube 和当前模型运行统一 Benchmark 并写入 Markdown 报告：

```bash
poetry run python -m app.agents.scenarios.nl2sql_benchmark validate \
  --target hydrology_semantic_query
poetry run python -m app.agents.scenarios.nl2sql_benchmark run \
  --target hydrology_semantic_query \
  --output /tmp/hydrology-semantic-result.md
```

报告输出 Execution Accuracy、Artifact Exact Match、P95 端到端耗时及 SemanticQuery 组件诊断，详见 [统一 Benchmark 文档](../nl2sql_benchmark/README.md)。
