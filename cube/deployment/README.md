# Cube 本地一键启动

该部署只启动 Cube，并连接已存在的 MySQL；不包含 MySQL、Agent API、LLM 和框架数据库。

## 启动

```bash
cd app/agents/scenarios/hydrology_semantic_query/cube/deployment
cp .env.example .env
```

填写 `.env` 中的 Cube 访问地址和端口，以及 MySQL 地址、端口、数据库、只读用户和密码，然后执行：

```bash
./start.sh
```

脚本会启动 Cube 并等待 `/readyz`。

## MySQL 前置条件

Cube 账号需对 [`../model/cubes/`](../model/cubes/) 中模型引用的表具有 `SELECT` 权限。Cube 使用 host 网络，本机 MySQL 默认连接 `127.0.0.1:3306`；连接其他 MySQL 时修改 `.env` 中的 `CUBEJS_DB_HOST` 和 `CUBEJS_DB_PORT`。

## Agent 对接

```bash
export HYDROLOGY_SEMANTIC_QUERY_CUBE_URL=http://127.0.0.1:4000
```

如果修改 `.env` 中的 `CUBE_HOST` 或 `CUBE_PORT`，需同步修改上述 Agent 地址。

## 维护

```bash
docker compose --env-file .env ps
docker compose --env-file .env logs -f cube
docker compose --env-file .env restart cube
docker compose --env-file .env down
curl http://127.0.0.1:4000/cubejs-api/v1/meta
python3 ../scripts/validate_cube_meta.py --url http://127.0.0.1:4000/cubejs-api/v1/meta
```

修改挂载的 `model/` 后执行 `restart cube`。

该配置使用 Linux host 网络和免认证 Dev Mode，Cube 共享宿主机网络，可能通过宿主机其他网卡暴露 `.env` 中 `CUBE_PORT` 指定的端口，仅限可信本地开发环境，不得用于远程或生产部署。`/meta` 校验只证明 Cube 就绪且模型符合契约，不保证 MySQL 数据完整或 `/load` 查询成功；真实查询需运行场景集成测试。
