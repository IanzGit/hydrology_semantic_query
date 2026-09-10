#!/usr/bin/env bash
set -euo pipefail

script_started=$SECONDS
start_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scenario_dir="$(cd "$start_dir/../.." && pwd)"
env_file="$scenario_dir/env/cube.env"
project_dir="$(cd "$start_dir/../../../../../../../.." && pwd)"

if [[ ! -f "$env_file" ]]; then
  echo "缺少 $env_file，请先复制 env/cube.env.example 并启用一套数据库配置。" >&2
  exit 1
fi

cd "$project_dir"
config_started=$SECONDS
if ! config_output="$(poetry run python "$start_dir/cube_config.py" --env-file "$env_file")"; then
  exit 1
fi
mapfile -t config_values <<< "$config_output"
if [[ "${#config_values[@]}" -ne 8 ]]; then
  echo "Cube启动配置解析结果无效。" >&2
  exit 1
fi

model_name="${config_values[0]}"
model_dir="${config_values[1]}"
cube_host="${config_values[2]}"
cube_port="${config_values[3]}"
cube_dev_mode="${config_values[4]}"
database_type="${config_values[5]}"
database_host="${config_values[6]}"
database_name="${config_values[7]}"
export CUBE_ENV_FILE="$env_file"
export CUBE_MODEL_DIR="$model_dir"
export CUBE_PORT="$cube_port"
export CUBEJS_DEV_MODE="$cube_dev_mode"
compose=(docker compose --env-file "$env_file")
cube_url="http://${cube_host}:${cube_port}"
printf 'Cube配置校验完成：elapsed=%ss\n' "$((SECONDS - config_started))"

redact_output() {
  (cd "$project_dir" && poetry run python "$start_dir/cube_config.py" --env-file "$env_file" --redact-stdin)
}

print_cube_logs() {
  local logs
  logs="$(cd "$start_dir" && "${compose[@]}" logs --tail=200 cube 2>&1 || true)"
  printf '%s\n' "$logs" | redact_output >&2
}

cd "$start_dir"
compose_started=$SECONDS
if ! compose_output="$("${compose[@]}" config --quiet 2>&1)"; then
  echo "Cube Compose配置校验失败。" >&2
  printf '%s\n' "$compose_output" | redact_output >&2
  exit 1
fi
# 模型目录可能被整体替换；仅按 Compose 配置比较不会刷新旧的 bind mount。
if ! compose_output="$("${compose[@]}" up -d --force-recreate 2>&1)"; then
  echo "Cube 启动或就绪检查失败。" >&2
  printf '%s\n' "$compose_output" | redact_output >&2
  print_cube_logs
  exit 1
fi
if ! (
  cd "$project_dir"
  CUBE_READY_URL="${cube_url}/readyz" poetry run python - <<'PY'
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ready_url = os.environ["CUBE_READY_URL"]
deadline = time.monotonic() + 120
last_error = "超时"
while time.monotonic() < deadline:
    try:
        with urlopen(ready_url, timeout=3) as response:
            if 200 <= response.status < 300:
                raise SystemExit(0)
            last_error = f"HTTP {response.status}"
    except HTTPError as exc:
        last_error = f"HTTP {exc.code}"
    except URLError as exc:
        last_error = str(exc.reason)
    time.sleep(1)

print(f"Cube /readyz 就绪检查失败：{last_error}", file=sys.stderr)
raise SystemExit(1)
PY
); then
  echo "Cube 启动或就绪检查失败。" >&2
  printf '%s\n' "$compose_output" | redact_output >&2
  print_cube_logs
  exit 1
fi
printf 'Cube容器启动完成：elapsed=%ss\n' "$((SECONDS - compose_started))"

cd "$project_dir"
vector_index_path="$start_dir/../cache/semantic-catalog-vectors.sqlite3"
meta_started=$SECONDS
if ! poetry run python -m app.agents.scenarios.cqccri_smart_query.subgraph.hydrology_semantic_query.semantic.scripts.validate_cube_meta --url "${cube_url}/cubejs-api/v1/meta"; then
  print_cube_logs
  exit 1
fi
printf 'Cube元数据校验完成：elapsed=%ss\n' "$((SECONDS - meta_started))"
vector_started=$SECONDS
if ! HYDROLOGY_SEMANTIC_QUERY_CUBE_URL="$cube_url" HYDROLOGY_SEMANTIC_QUERY_VECTOR_INDEX_PATH="$vector_index_path" poetry run python "$start_dir/rebuild_vector_index.py"; then
  print_cube_logs
  exit 1
fi
printf '语义目录向量索引处理完成：elapsed=%ss\n' "$((SECONDS - vector_started))"
printf 'Cube启动成功：model=%s db_type=%s db_host=%s db_name=%s\n' "$model_name" "$database_type" "$database_host" "$database_name"
printf 'Cube启动总耗时：elapsed=%ss\n' "$((SECONDS - script_started))"
printf '请重启Agent，以清除旧的Cube目录与语义检索缓存。\n'
