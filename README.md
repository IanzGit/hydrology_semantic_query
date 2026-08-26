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

脚本不读取本地 Cube 模型范围，必须至少传入一个 `--table`。扩展数据库时先显式生成并审核草稿，再把需要自然语言查询的稳定业务粒度迁入正式 Cube；生成基础 Cube 不会同时生成 View。

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
- `HYDROLOGY_SEMANTIC_QUERY_CATALOG_STRATEGY`：上下文策略，可选 `full`、`vector` 或 `auto`，默认为 `auto`。
- `HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_MODEL`：本地 sentence-transformers 模型目录；留空或加载失败时通过同一接口使用词法检索。
- `HYDROLOGY_SEMANTIC_QUERY_CONTEXT_TOP_K`：统一目录检索的命中数量，默认为 `20`。
- `HYDROLOGY_SEMANTIC_QUERY_VECTOR_INDEX_PATH`：语义目录向量索引的 SQLite 缓存路径；留空时仅使用内存索引。
- `HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_BATCH_SIZE`：目录文档嵌入批次大小，默认为 `32`。
- `HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_CONCURRENCY`：目录文档嵌入并发数，默认为 `3`。
- `HYDROLOGY_SEMANTIC_QUERY_AUTO_FULL_CONTEXT_MAX_CHARS`：`auto` 模式直接提供完整可访问目录的字符阈值，默认为 `30000`。

运行时链路为：用户问题 → Cube 语义目录上下文检索 → Planner 全局生成 `SemanticQuery` → 完整可访问目录校验 → Cube `/sql` 编译预检 → Cube `/load` 执行。Planner 根据原问题、业务知识、会话上下文和目录上下文，自主决定 View/Cube、模型组合、成员、过滤、聚合、明细形态和默认投影；不会生成原始 SQL。

目录检索只构造 Planner 上下文，不解析问题、不绑定成员、不决定模型路由，也不形成成员白名单。`auto` 模式下，完整可访问目录不超过 30000 字符时直接提供完整目录；大目录使用原问题统一检索 model、member、view folder 和 join component，默认 Top-20。成员命中会补充父模型概览，Cube 命中会补充连通分量信息。向量组件不可用时使用相同接口的词法检索，向量结果使用 SQLite 缓存。

metadata filters 先产生完整可访问目录，校验始终基于该目录，而不是检索命中项。`SemanticQuery` 使用 `query_mode` 与 `models` 表达路由，二者不会发送给 Cube `/sql` 或 `/load`。View 模式必须恰好一个 View；Cube 模式允许一个 Cube 或同一非空 `connectedComponent` 中的二至四个 Cube。请求 metadata 可使用 `catalog_mode` 和 `catalog_metadata_filters` 覆盖上下文策略及按 `model_name`、`model_type`、`title`、`business_domain` 过滤的可访问范围，不能绕过模型边界。

JSON 或结构错误会携带原响应错误，在当前上下文中最多额外生成一次。未知模型、未知成员、成员范围或类型、连通性及可修正的 Cube 400 错误会用“原问题 + 结构化错误”刷新 Top-40 上下文，和已有上下文合并后重试。认证、网络、超时和系统错误不会交给模型修正；空结果直接返回 `no_data`，不扩大目录或改变过滤条件。

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
