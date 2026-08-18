#!/usr/bin/env bash
set -euo pipefail

deployment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$deployment_dir/.env"
compose=(docker compose --env-file "$env_file")
cd "$deployment_dir"

if [[ ! -f "$env_file" ]]; then
  echo "缺少 $env_file，请先复制 .env.example 并填写 MySQL 配置。" >&2
  exit 1
fi

if ! "${compose[@]}" up -d --wait --wait-timeout 120; then
  echo "Cube 启动或就绪检查失败。" >&2
  "${compose[@]}" logs --tail=200 cube || true
  exit 1
fi
