# Cube 本地一键启动

该部署只启动一个 Cube 实例；通过 `env/cube.env` 选择模型与目标数据库，不包含数据库、Agent API、LLM 和框架数据库。

## 启动

```bash
cd app/agents/scenarios/cqccri_smart_query/subgraph/hydrology_semantic_query
cp env/cube.env.example env/cube.env
```

在 `env/cube.env` 中保留一套启用的模型和数据库配置，填写数据库连接，然后执行：

```bash
semantic/start/start.sh
```

脚本会先校验 Cube 配置、模型目录和 Compose 配置，再启动 Cube；配置发生变化时由 Compose 自动重建容器。通过 `/meta` 校验后准备语义目录向量索引，并输出各阶段耗时。任一步骤失败都会返回非零状态；数据库密码不会进入脚本输出。

## 切换模型与数据库

`env/cube.env.example` 默认启用水文 MySQL 配置，并以注释块提供水文监测 SQL Server 配置。例如：

```dotenv
CUBE_HOST=127.0.0.1
CUBE_PORT=4000
CUBEJS_DEV_MODE=true
MODEL_NAME=hydrology_model
CUBEJS_DB_TYPE=mysql
CUBEJS_DB_HOST=127.0.0.1
CUBEJS_DB_PORT=3306
CUBEJS_DB_NAME=hydrology_local
CUBEJS_DB_USER=readonly
CUBEJS_DB_PASS=replace_me
```

切换时必须完整注释当前的 `MODEL_NAME` 和 6 项 `CUBEJS_DB_*`，再完整取消目标配置块的注释。如果同一配置键同时启用多次，启动脚本会拒绝继续。实际 `env/cube.env` 不提交版本库，数据库账号需对选中模型引用的表具有 `SELECT` 权限。

## Agent 对接

```bash
export HYDROLOGY_SEMANTIC_QUERY_CUBE_URL=http://127.0.0.1:4000
```

如果修改 `env/cube.env` 中的 `CUBE_HOST` 或 `CUBE_PORT`，需同步修改上述 Agent 地址。切换模型成功后必须重启 Agent，以清除旧的 `/meta` 与内存 Retriever。

## 维护

```bash
curl http://127.0.0.1:4000/cubejs-api/v1/meta
python3 semantic/scripts/validate_cube_meta.py --url http://127.0.0.1:4000/cubejs-api/v1/meta
```

`env/cube.env` 会动态决定 Compose 的环境变量与只读模型挂载，维护时优先重新执行 `semantic/start/start.sh`，不要只执行 `docker compose restart cube`。需要忽略签名并强制重建时，在项目根目录执行：

```bash
poetry run python \
  app/agents/scenarios/cqccri_smart_query/subgraph/hydrology_semantic_query/semantic/start/rebuild_vector_index.py \
  --force
```

该配置使用 Linux host 网络。Dev Mode 下 Cube 可能通过宿主机其他网卡暴露 `CUBE_PORT`，仅限可信本地开发环境。`/meta` 校验只证明 Cube 就绪且模型符合契约，真实数据库查询仍需运行场景集成测试。
