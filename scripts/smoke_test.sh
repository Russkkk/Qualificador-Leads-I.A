#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-${BASE_URL:-}}"
if [[ -z "${BASE_URL}" ]]; then
  echo "Uso: ./scripts/smoke_test.sh https://seu-backend.onrender.com" >&2
  exit 2
fi
BASE_URL="${BASE_URL%/}"

echo "== Smoke test: ${BASE_URL} =="

req() {
  local path="$1"
  echo "-- GET ${path}"
  # imprime headers + corpo (compacto) e valida status
  local tmp_h tmp_b
  tmp_h=$(mktemp)
  tmp_b=$(mktemp)
  local code
  code=$(curl -sS -D "$tmp_h" -o "$tmp_b" -w "%{http_code}" "${BASE_URL}${path}")
  echo "Status: ${code}"
  local rid
  rid=$(grep -i '^x-request-id:' "$tmp_h" | head -n1 | awk '{print $2}' | tr -d '\r')
  if [[ -n "${rid}" ]]; then
    echo "X-Request-ID: ${rid}"
  fi
  # mostra só primeira linha do body para não poluir
  head -c 400 "$tmp_b"; echo
  rm -f "$tmp_h" "$tmp_b"
  [[ "$code" =~ ^2 ]] || return 1
}

req "/health"
req "/public_config"
req "/pricing"

# Demo endpoint só se estiver habilitado
if curl -sS "${BASE_URL}/public_config" | grep -q '"demo"[: ]*true'; then
  req "/demo/acao_do_dia"
else
  echo "-- DEMO_MODE não está ativo (ok)"
fi

echo "✅ Smoke test passou"
