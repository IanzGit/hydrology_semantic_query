#!/usr/bin/env bash
set -euo pipefail

script_started=$SECONDS
start_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$start_dir/.env"
project_dir="$(cd "$start_dir/../../../../../.." && pwd)"

if [[ ! -f "$env_file" ]]; then
  echo "缺少 $env_file，请先复制 .env.example 并选择ACTIVE_PROFILE。" >&2
  exit 1
fi

cd "$project_dir"
profile_started=$SECONDS
if ! profile_output="$(poetry run python "$start_dir/profile_config.py" --selector "$env_file")"; then
  exit 1
fi
mapfile -t profile_values <<< "$profile_output"
if [[ "${#profile_values[@]}" -ne 10 ]]; then
  echo "Cube启动Profile解析结果无效。" >&2
  exit 1
fi

active_profile="${profile_values[0]}"
profile_file="${profile_values[1]}"
model_name="${profile_values[2]}"
model_dir="${profile_values[3]}"
cube_host="${profile_values[4]}"
cube_port="${profile_values[5]}"
cube_dev_mode="${profile_values[6]}"
database_type="${profile_values[7]}"
database_host="${profile_values[8]}"
database_name="${profile_values[9]}"
export CUBE_PROFILE_ENV="$profile_file"
export CUBE_MODEL_DIR="$model_dir"
export CUBE_PORT="$cube_port"
export CUBEJS_DEV_MODE="$cube_dev_mode"
compose=(docker compose --env-file "$profile_file" --env-file "$env_file")
printf 'Cube Profile校验完成：elapsed=%ss\n' "$((SECONDS - profile_started))"

redact_output() {
  (cd "$project_dir" && poetry run python "$start_dir/profile_config.py" --selector "$env_file" --redact-stdin)
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
if ! compose_output="$("${compose[@]}" up -d --wait --wait-timeout 120 2>&1)"; then
  echo "Cube 启动或就绪检查失败。" >&2
  printf '%s\n' "$compose_output" | redact_output >&2
  print_cube_logs
  exit 1
fi
printf 'Cube容器启动完成：elapsed=%ss\n' "$((SECONDS - compose_started))"

cd "$project_dir"
cube_url="http://${cube_host}:${cube_port}"
vector_index_path="$start_dir/../cache/semantic-catalog-vectors.sqlite3"
meta_started=$SECONDS
if ! poetry run python -m app.agents.scenarios.hydrology_semantic_query.semantic.scripts.validate_cube_meta --url "${cube_url}/cubejs-api/v1/meta"; then
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
printf 'Cube Profile启动成功：profile=%s model=%s db_type=%s db_host=%s db_name=%s\n' "$active_profile" "$model_name" "$database_type" "$database_host" "$database_name"
printf 'Cube启动总耗时：elapsed=%ss\n' "$((SECONDS - script_started))"
printf '请重启Agent，以清除旧的Cube目录与语义检索缓存。\n'
