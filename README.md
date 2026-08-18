# hydrology_semantic_query

该场景通过 Cube REST API 加载公开语义目录并执行 `SemanticQuery`。Cube 部署需将 `cube/model/cubes` 和 `cube/model/views` 挂载到 Cube 数据模型目录。

连接外部 MySQL 的本地 Cube 可通过 `cube/deployment/start.sh` 一键启动，详见 [cube/deployment/README.md](cube/deployment/README.md)。

## Cube 基础模型生成

`cube/scripts/generate_cube_models.py` 使用 SQLAlchemy 读取 MySQL 中的表、View、字段、主键和外键，生成仅供审阅的 Cube 草稿。草稿固定输出至 `cube/generated/`，不会改写或加载当前 `cube/model/` 中的治理模型。

脚本优先使用 `--database-url`：

```bash
poetry run python -m app.agents.scenarios.hydrology_semantic_query.cube.scripts.generate_cube_models \
  --database-url 'mysql+pymysql://root:123456@127.0.0.1:3306/hydrology_local'\
  --table device_info \
  --table device_x_value\
  --table label_sensor\
  --table label\
  --table warn_state_info\
  --table water_warn_sensor_set\
  --table multifactor_sensor
```

也可以使用 `CUBEJS_DB_HOST`、`CUBEJS_DB_PORT`、`CUBEJS_DB_NAME`、`CUBEJS_DB_USER`、`CUBEJS_DB_PASS` 环境变量；当 Cube 变量未设置时，兼容 `MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_DATABASE`、`MYSQL_USER`、`MYSQL_PASSWORD`。

不传 `--table` 时扫描当前数据库的全部表和 View，默认排除 `*_log`、`*_bak`、`tmp_*`、`sys_*`、`*_history_backup`。可以追加排除模式：

```bash
poetry run python -m app.agents.scenarios.hydrology_semantic_query.cube.scripts.generate_cube_models \
  --exclude-pattern '*_archive'
```

反复传入 `--table` 可仅生成指定对象，显式指定时不应用排除模式：

```bash
poetry run python -m app.agents.scenarios.hydrology_semantic_query.cube.scripts.generate_cube_models \
  --table virtual_hydrological_monitoring \
  --table virtual_water_pumping
```

`cube/generated/model/cubes/` 包含每个数据库对象对应的私有 Cube，只自动生成 Dimension、`count` 和安全的单字段外键 Join。`cube/generated/join_candidates.yml` 记录按 `*_id`/`*Id` 命名推测的候选关联以及未入模的外键。脚本每次成功运行都会整体替换 `cube/generated/`；生成失败时保留上一次草稿。业务指标、中文语义、Segment、自定义 SQL 和 View 仍需人工治理。

Agent 运行时会自动读取场景根目录下的 `.env`，进程环境变量优先于该文件中的同名配置。该文件只管理 Agent 场景专属配置；`cube/deployment/.env` 继续独立管理 Cube 的 MySQL 连接配置。

运行时配置：

- `HYDROLOGY_SEMANTIC_QUERY_CUBE_URL`：Cube 服务地址、`/cubejs-api` 地址或已完整的 `/cubejs-api/v1` REST API 根地址。
- `HYDROLOGY_SEMANTIC_QUERY_CUBE_TOKEN`：Cube API Token；Cube 未启用认证时可留空。
- `HYDROLOGY_SEMANTIC_QUERY_TIMEOUT_SECONDS`：Cube HTTP 请求超时秒数。
- `HYDROLOGY_SEMANTIC_QUERY_CONTINUE_WAIT_RETRIES`：Cube 返回 continue wait 后的重试次数。
- `HYDROLOGY_SEMANTIC_QUERY_META_CACHE_TTL_SECONDS`：Cube 元数据缓存秒数。
- `HYDROLOGY_SEMANTIC_QUERY_MAX_RETRIES`：首次 SemanticQuery 生成后的最大额外重试次数，不能超过 1。
- `HYDROLOGY_SEMANTIC_QUERY_TIMEZONE`：IANA 时区，默认为 `Asia/Shanghai`。
- `HYDROLOGY_SEMANTIC_QUERY_MAX_ROWS` 和 `HYDROLOGY_SEMANTIC_QUERY_HARD_MAX_ROWS`：默认行数上限与硬上限。
- `HYDROLOGY_SEMANTIC_QUERY_ENABLE_REPORT`：是否默认生成查询结果报告。
- `HYDROLOGY_SEMANTIC_QUERY_CATALOG_STRATEGY`：目录选择模式，可选 `full`、`vector` 或 `auto`，默认为 `vector`。
- `HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_MODEL`：本地 sentence-transformers 模型目录；留空或加载失败时改用有界词法检索。
- `HYDROLOGY_SEMANTIC_QUERY_VIEW_TOP_K`、`HYDROLOGY_SEMANTIC_QUERY_CUBE_TOP_K`：独立召回的 View 和 Cube 数量，默认分别为 `3`、`5`。
- `HYDROLOGY_SEMANTIC_QUERY_MEMBER_TOP_K`：全局成员索引召回数量，默认为 `15`；成员命中可以把未被 Cube Top-K 命中的父 Cube 拉回候选集。
- `HYDROLOGY_SEMANTIC_QUERY_VECTOR_INDEX_PATH`：语义目录向量索引的 SQLite 缓存路径；留空时仅使用内存索引。
- `HYDROLOGY_SEMANTIC_QUERY_RETRY_ON_EMPTY_RESULT`：是否在结果为空时进入同连通分量的 Cube 批次回退，默认为 `true`。
- `HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_BATCH_SIZE`：目录文档嵌入批次大小，默认为 `32`。
- `HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_CONCURRENCY`：目录文档嵌入并发数，默认为 `3`。
- `HYDROLOGY_SEMANTIC_QUERY_RETRIEVAL_CONCURRENCY`：检索并发数，默认为 `3`。
- `HYDROLOGY_SEMANTIC_QUERY_CONTEXT_MEMBER_LIMIT`：单次提示词允许的成员数，默认为 `12`，不能超过 `12`。
- `HYDROLOGY_SEMANTIC_QUERY_CATALOG_BATCH_SIZE`：连通分量回退的 Cube 批大小，默认为 `4`。
- `HYDROLOGY_SEMANTIC_QUERY_MAX_CUBE_MODELS`：一条 Cube Graph 查询允许的最大 Cube 数，默认为 `4`，不能超过 `4`。
- `HYDROLOGY_SEMANTIC_QUERY_MEMBER_MATCH_THRESHOLD`：概念覆盖判定的最低成员相关度，默认为 `0.55`。

运行时先把问题解析为指标、维度、时间、过滤、排序和条数意图，再分别检索 View Top-3、Cube Top-5 与全局 Member Top-15。候选按语义相关度 40%、概念覆盖率 30%、精确词或别名 10%、图连通性 10%、业务优先级 10% 重排。某个 View 覆盖全部必需概念时进入 `view` 模式并只允许一个 View；否则进入 `cube` 模式，从最多四个 Cube 构造精确、连通、无歧义的最小 Join 子图。

只把选中的 1 个 View 或 2 至 4 个 Cube、6 至 12 个成员、1 至 3 条 Join Path、固定业务语义和允许成员发送给查询生成模型，不把原始完整目录发送给模型。向量组件不可用时仍使用相同数量边界的词法召回。生成失败沿用原上下文重试；校验失败、可修正执行失败或空结果进入同连通分量或同业务域的分批分析；仍无法得到无歧义连通模型时返回结构化 `semantic_model_gap`，不自动退回原始 SQL。

`SemanticQuery` 使用 `query_mode` 与 `models` 表达路由，二者是 Agent 控制字段，不会发送给 Cube `/load`。View 模式必须恰好一个 View；Cube 模式允许 1 至 4 个基础 Cube，但所用模型必须通过显式 Join Edge 连通，且所有成员必须带选中模型前缀。请求 metadata 只允许使用 `catalog_mode` 和 `catalog_metadata_filters` 覆盖检索策略及按 `model_name`、`model_type`、`title` 过滤的候选范围，不能绕过模型边界。

公开语义目录固定接纳三个高频业务 View 与七个显式 `public: true` 的 `base_*` Cube；`role_label_parent` 保持私有。基础 Cube 只公开稳定业务字段，寄存器、脚本、文件路径、函数参数和审计字段继续隐藏。精确 Join Edge 来自受治理 `meta.join_edges`，`connectedComponent` 只用于粗召回，不能替代 Join 路径证明。业务范围、粒度、关系和隐藏成员见 [语义模型人工治理指南](cube/BUSINESS_VIEW_GOVERNANCE.md)。

可使用真实 Cube 和当前模型运行统一 Benchmark 并写入 Markdown 报告：

```bash
poetry run python -m app.agents.scenarios.nl2sql_benchmark validate \
  --target hydrology_semantic_query
poetry run python -m app.agents.scenarios.nl2sql_benchmark run \
  --target hydrology_semantic_query \
  --output /tmp/hydrology-semantic-result.md
```

报告输出 Execution Accuracy、Artifact Exact Match、P95 端到端耗时及 SemanticQuery 组件诊断，详见 [统一 Benchmark 文档](../nl2sql_benchmark/README.md)。
