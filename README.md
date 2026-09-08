# hydrology_semantic_query

该场景通过 Cube REST API 加载公开语义目录并执行 `SemanticQuery`。现有水文语义模型位于 `semantic/model/hydrology_model`，水文监测模型位于 `semantic/model/hydrology_monitor_model`。

## 代码结构

场景代码按职责组织：

```text
hydrology_semantic_query/
├── agent.py           # 场景注册入口
├── contracts.py       # 跨 Agent 输入、输出与共享模型契约
├── env/               # Agent 与 Cube 配置
├── graph.py           # 主图拓扑装配
├── knowledge.py       # 外部 Markdown 业务知识加载
├── node.py            # 主 Agent 初始化、计划、Replan、派发和收敛节点
├── prompts.py         # 主 Agent 与多轮问题改写提示词
├── state.py           # 主 Agent 工作流状态
├── query_child/       # 单个查询任务的 ReAct 子 Agent
│   ├── graph.py       # 查询子图拓扑装配
│   ├── node.py        # 查询准备、决策与收敛节点
│   ├── state.py       # 查询子 Agent 内部状态
│   ├── prompts.py     # 查询子 Agent 提示词
│   ├── models.py      # 查询目录与上下文内部模型
│   ├── runtime.py     # 查询服务与运行辅助
│   ├── client.py      # Cube HTTP 客户端及 meta 目录解析
│   ├── config.py      # 查询子 Agent 运行配置
│   ├── tool_call_parser.py
│   └── tools/         # 目录检索与语义查询工具链
├── report_child/      # 多任务结果分析与报告子 Agent
│   ├── graph.py       # 报告子图拓扑装配
│   ├── models.py      # 报告共享契约导出
│   ├── node.py        # 分析、叙事与渲染节点
│   ├── prompts.py     # 报告子 Agent 叙事提示词
│   ├── runtime.py     # 报告步骤运行辅助
│   ├── state.py       # 报告子 Agent 内部状态
│   ├── report.py      # 报告事实、叙事生成和输出
│   ├── report_analysis.py
│   ├── visualization.py # 章节级可视化候选、规划、校验与降级
│   └── report_rendering.py
├── semantic/          # Cube 语义模型、部署和脚本
└── tests/             # 场景行为、部署、集成和架构回归测试
```

主图只负责编排，查询子 Agent 封装 Cube 访问与工具链，报告子 Agent 只消费已聚合的查询输出契约。两个子目录只反向依赖根目录的 `contracts.py`，不导入主状态和主编排实现；根 `state.py` 也不导入任何子 Agent，从依赖方向上避免循环导入。

连接外部 MySQL 的本地 Cube 可通过 `semantic/start/start.sh` 一键启动，详见 [semantic/start/README.md](semantic/start/README.md)。

## Cube 基础模型生成

`semantic/scripts/generate_cube_models.py` 使用 SQLAlchemy 读取 MySQL 中的表、View、字段、主键和外键，生成仅供审阅的 Cube 草稿。草稿固定输出至 `semantic/generated/`，不会改写或加载当前 `semantic/model/` 中的治理模型。

脚本优先使用 `--database-url`：

```bash
poetry run python -m app.agents.scenarios.cqccri_smart_query.subgraph.hydrology_semantic_query.semantic.scripts.generate_cube_models \
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
poetry run python -m app.agents.scenarios.cqccri_smart_query.subgraph.hydrology_semantic_query.semantic.scripts.generate_cube_models \
  --table virtual_hydrological_monitoring \
  --table virtual_water_pumping
```

`semantic/generated/model/cubes/` 包含每个数据库对象对应的私有 Cube，只自动生成 Dimension、`count` 和安全的单字段外键 Join。`semantic/generated/join_candidates.yml` 记录按 `*_id`/`*Id` 命名推测的候选关联以及未入模的外键。脚本每次成功运行都会整体替换 `semantic/generated/`；生成失败时保留上一次草稿。业务指标、中文语义、Segment、自定义 SQL 和 View 仍需人工治理。

Agent 和 Cube 配置统一放在场景的 `env/` 目录。实际配置不提交版本库，首次配置 Agent 时执行：

```bash
cp env/agent.env.example env/agent.env
```

Agent 运行时自动读取 `env/agent.env`，进程环境变量优先于该文件中的同名配置。Cube 的启动参数、模型和数据库连接统一位于 `env/cube.env`。

运行时配置：

- `HYDROLOGY_SEMANTIC_QUERY_CUBE_URL`：Cube 服务地址、`/cubejs-api` 地址或已完整的 `/cubejs-api/v1` REST API 根地址。
- `HYDROLOGY_SEMANTIC_QUERY_CUBE_TOKEN`：Cube API Token；Cube 未启用认证时可留空。
- `HYDROLOGY_SEMANTIC_QUERY_TIMEOUT_SECONDS`：Cube HTTP 请求超时秒数。
- `HYDROLOGY_SEMANTIC_QUERY_CONTINUE_WAIT_RETRIES`：Cube 返回 continue wait 后的重试次数。
- `HYDROLOGY_SEMANTIC_QUERY_META_CACHE_TTL_SECONDS`：Cube 元数据缓存秒数。
- `HYDROLOGY_SEMANTIC_QUERY_MAX_AGENT_ITERATIONS`：ReAct Agent 最大决策轮数，默认为 `6`，允许 `3` 至 `12`。
- `HYDROLOGY_SEMANTIC_QUERY_MAX_QUERY_ROUNDS`：主控 Agent 最多可派发的查询任务数，默认为 `5`，允许 `1` 至 `5`；查询子 Agent 内部的检索和修正不额外占用任务预算。
- `HYDROLOGY_SEMANTIC_QUERY_TIMEZONE`：IANA 时区，默认为 `Asia/Shanghai`。
- `HYDROLOGY_SEMANTIC_QUERY_CATALOG_STRATEGY`：上下文策略，可选 `full`、`vector` 或 `auto`，默认为 `auto`。
- `HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_MODEL`：本地 sentence-transformers 模型目录；留空或加载失败时通过同一接口使用词法检索。
- `HYDROLOGY_SEMANTIC_QUERY_MODEL_TOP_K`：第一阶段 View/Cube 候选模型数量，默认为 `5`。
- `HYDROLOGY_SEMANTIC_QUERY_CONTEXT_TOP_K`：第二阶段候选模型内成员和目录组命中数量，默认为 `20`。
- `HYDROLOGY_SEMANTIC_QUERY_VECTOR_INDEX_PATH`：语义目录向量索引的 SQLite 缓存路径；默认为场景内的 `semantic/cache/semantic-catalog-vectors.sqlite3`，向量检索时不能为空。
- `HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_BATCH_SIZE`：目录文档嵌入批次大小，默认为 `32`。
- `HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_CONCURRENCY`：目录文档嵌入并发数，默认为 `3`。
- `HYDROLOGY_SEMANTIC_QUERY_AUTO_FULL_CONTEXT_MAX_CHARS`：`auto` 模式直接提供完整可访问目录的字符阈值，默认为 `30000`。

运行时主图为：初始化 → Main Agent 生成计划 → Query Agent 执行一个任务 → Main Agent Replan；查询任务结束后进入最终结果聚合 → Report Agent。Main Agent 可以根据已完成、无数据或失败的任务结果保留、取消或调整后续任务，但所有任务顺序执行且累计不超过任务预算。Query Agent 每次只接收一个自包含查询任务，加载 Cube `/meta` 并通过原有 ReAct、目录检索、SemanticQuery 校验、`/sql` 预检和 `/load` 执行能力返回一份查询结果，不规划下一项查询，也不生成报告。

外部业务知识放在场景目录的 `knowledge/*.md`。建图时会按文件名排序读取并缓存全部非空 Markdown，读取失败的文件会被跳过并写入结果警告；修改文件后必须重启或重新构建场景图才会生效。Main Agent 的通用提示词会附加这些文档，并且对一个问题最多选择一个最相关文件作为 playbook；命中时按文档约束查询步骤和报告章节，未命中或目录不存在时自主规划。请求 metadata 中旧的 `business_knowledge` 字段仍可传入以保持兼容，但不会注入任何 Agent 提示词。

Report Agent 接收原始问题、所有任务结果和 Main Agent 确定的章节结构，并以每个 `ReportSectionRequirement` 作为最终报告基本单元。每章按 `objective`、`analysis_methods` 和 `source_task_ids` 聚合事实、文字、图表与必要数据表；同一查询结果可以被多个章节从不同角度复用。报告子图依次执行章节分析、章节级可视化规划、章节叙事和交错渲染，使每章形成“数据依据—分析过程—图表/数据表—结论说明”的完整结构。

可视化规划采用“后端生成字段安全候选、模型按章节选择、无效章节确定性降级”的混合方式。规划器综合章节目标、分析方法、已验证事实、局限和结果画像，决定是否绘图及图表数量，而不是直接按字段类型输出。当前前端协议保留 `BAR`、`LINE`、`PIE`：相关性用精确时间对齐后的多序列折线图近似，异常分析用折线图或柱状图近似并保留异常/阈值数据表。部分任务失败时仍会基于成功任务生成报告，并在相关章节明确数据缺口；全部任务失败时返回结构化失败结果。跨任务相关性分析只对粒度一致且时间戳能够精确对齐的序列计算 Pearson 系数，至少需要 8 个共同样本，不进行插值或推测性对齐。

`search_semantic_catalog` 按查询任务检索相关 Cube、View 和 member。数据查询首次执行前必须至少成功调用一次；重复调用会合并语义上下文和检索轨迹。`run_semantic_query` 接受 `SemanticQuery` 和 `query_goal`，统一执行完整可访问目录校验、Cube `/sql` 编译预检和 Cube `/load` 执行，不接受原始 SQL。成功或无数据后当前 Query Agent 立即结束；规范化查询、列、完整结果和编译 SQL 保存在任务结果、查询历史和最终结构化结果中。

目录检索只构造 Agent 上下文，不绑定成员、不决定最终模型路由，也不形成成员白名单。存在会话历史时，初始化阶段会把当前追问改写为独立问题；改写失败时回退原问题。`auto` 模式下，完整可访问目录不超过 30000 字符时直接提供完整目录；大目录先检索最相关的 View/Cube，再只在候选模型内检索 member 和 view folder。成员命中会补充父模型概览，Cube 命中会补充候选模型范围内的连通分量信息。向量组件不可用时使用相同的两阶段词法检索；目录向量必须在 Cube 部署阶段预先写入 SQLite，查询时只加载签名匹配的索引，不会现场重建。

修改正式 Cube 模型后，通过 `semantic/start/start.sh` 重启 Cube 并自动检查语义目录索引。目录签名变化时脚本会在 Agent 启动前完成重建，失败则阻止部署继续。也可以独立执行或强制重建：

```bash
poetry run python \
  app/agents/scenarios/cqccri_smart_query/subgraph/hydrology_semantic_query/semantic/start/rebuild_vector_index.py
poetry run python \
  app/agents/scenarios/cqccri_smart_query/subgraph/hydrology_semantic_query/semantic/start/rebuild_vector_index.py \
  --force
```

索引缺失、损坏、版本过期或与最新 `/meta` 不一致时，向量目录检索会返回部署错误。若 Agent 已经运行，Cube 模型更新和索引重建完成后仍需重启 Agent，以清除旧的 `/meta` 与内存 Retriever。

metadata filters 先产生完整可访问目录，校验始终基于该目录，而不是检索命中项。`SemanticQuery` 使用 `query_mode` 与 `models` 表达路由，二者不会发送给 Cube `/sql` 或 `/load`。View 模式必须恰好一个 View；Cube 模式允许一个 Cube 或同一非空 `connectedComponent` 中的二至四个 Cube。CQCCRI 统一入口不接收请求级 `catalog_mode` 和 `catalog_metadata_filters`，上下文策略由场景环境配置控制。

工具参数、本地 Validator 和 Cube 返回的错误会以结构化 Observation 反馈。未知模型、未知成员、成员范围或类型、连通性及可修正的 Cube 400 错误允许 Query Agent 自主检索和重试；失败尝试由 `attempts` 统计，不增加主控任务预算。单次检索上限为 40，并与当前任务已有上下文合并。认证、网络、超时和系统错误会终止当前查询；空结果直接返回 `no_data`，不扩大目录或改变过滤条件。到达 Query Agent 最大决策轮数时终止当前任务，Main Agent 再根据结果 Replan。

公开语义目录完全来自 Cube `/meta`；当前模型包含三个高频业务 View 与 108 个公开 `base_*` Cube，`role_label_parent` 保持私有。原有七个 Cube 继续承担人工治理的高频业务图，其余 101 个 Cube 按数据库中有数据的物理表粒度提供长尾查询；未声明主键的表不参与 Join。基础 Cube 只公开稳定业务字段，寄存器、脚本、文件路径、函数参数、凭据和审计字段继续隐藏。Python 不再扫描本地模型文件，非空 `connectedComponent` 用于多个 Cube 的查询前粗校验，独立 Cube 不要求该字段；精确 Join Path 由 Cube `/sql` 按部署模型解析。View 默认投影和固定过滤字段角色分别由模型与成员 `meta` 通过接口提供。业务范围、粒度、关系和隐藏成员见 [语义模型人工治理指南](semantic/BUSINESS_VIEW_GOVERNANCE.md)。

原仓库的 `nl2sql_benchmark` 位于场景目录外，本次未迁移，因此当前目标仓库不提供对应 Benchmark 命令。
