param(
  [Parameter(Mandatory=$true)][string]$BaseUrl
)

$BaseUrl = $BaseUrl.TrimEnd('/')
Write-Host "== Smoke test: $BaseUrl =="

function Invoke-Get([string]$Path){
  Write-Host "-- GET $Path"
  try {
    $resp = Invoke-WebRequest -Uri "$BaseUrl$Path" -UseBasicParsing
  } catch {
    Write-Host "Falha no request: $_" -ForegroundColor Red
    exit 1
  }
  Write-Host "Status: $($resp.StatusCode)"
  if ($resp.Headers['X-Request-ID']) {
    Write-Host "X-Request-ID: $($resp.Headers['X-Request-ID'])"
  }
  $body = $resp.Content
  if ($body.Length -gt 400) { $body = $body.Substring(0,400) }
  Write-Host $body
}

Invoke-Get "/health"
Invoke-Get "/public_config"
Invoke-Get "/pricing"

try {
  $cfg = Invoke-RestMethod -Uri "$BaseUrl/public_config" -Method Get
  if ($cfg.features.demo -eq $true) {
    Invoke-Get "/demo/acao_do_dia"
  } else {
    Write-Host "-- DEMO_MODE não está ativo (ok)"
  }
} catch {
  Write-Host "Não consegui ler /public_config para checar demo (ok)"
}

Write-Host "✅ Smoke test passou" -ForegroundColor Green
