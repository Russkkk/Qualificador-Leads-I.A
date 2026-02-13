# Operação e Deploy (Render)

Este guia foi escrito para **ativar observabilidade, anti‑spam e modo demo** sem quebrar o que já funciona.

## 0) Checklist rápido (produção)

**Obrigatório**
- `DATABASE_URL`
- `FLASK_SECRET_KEY` (forte)

**Recomendado**
- `TRUST_PROXY=true` (quando estiver atrás de proxy na Render)
- `DEBUG=false`
- `INCLUDE_TRACEBACK=false`
- `REQUIRE_API_KEY=true` (quando sua API key já estiver sendo exigida em rotas privadas)

## 1) Como editar variáveis de ambiente na Render

1. Abra o serviço no dashboard da Render.
2. Clique em **Environment**.
3. Em **Environment Variables**, clique em **+ Add Environment Variable**.
4. Salve.

> Dica: crie um **Environment Group** para compartilhar variáveis entre staging/prod.

## 2) Observabilidade

### 2.1 Request ID
O backend adiciona `X-Request-ID` em todas as respostas.
- Use isso para correlacionar um erro do usuário no front com logs da Render.

### 2.2 Sentry (backend)
1. Crie um projeto no Sentry (Plataforma: **Python / Flask**).
2. Copie o DSN.
3. Configure no Render:
   - `SENTRY_DSN=...`
   - `SENTRY_ENVIRONMENT=prod` (ou `staging`)
   - `SENTRY_TRACES_SAMPLE_RATE=0.0` (suba para `0.05` quando quiser performance tracing)

## 3) Report de erro do front (opcional)

Ativa um coletor leve para erros JS:
- `CLIENT_ERROR_REPORTING=true`
- `CLIENT_ERROR_SAMPLE_RATE=0.05`

O front envia para `POST /client_error` com amostragem (não impacta usuário).

## 4) Anti-spam (Turnstile) — rollout seguro

### 4.1 Criar widget e chaves
- Crie um widget Turnstile no Cloudflare.
- Adicione hostnames:
  - `qualificador-leads-ia.onrender.com`
  - seu domínio custom (ex.: `leadrank.com.br`)

### 4.2 Ativar em **modo soft** (não quebra signup)
Configure no Render:
- `TURNSTILE_SITE_KEY=...`
- `TURNSTILE_SECRET_KEY=...`
- `CAPTCHA_ENFORCE=false` (soft)

### 4.3 Subir para **enforce** (quando estiver estável)
- `CAPTCHA_ENFORCE=true`

## 5) Demo mode (opcional)

Para expor a demo pública:
- `DEMO_MODE=true`

Endpoints:
- `GET /demo/acao_do_dia` (read-only)

Página:
- `static_site/demo.html`

## 6) Smoke tests

Veja `scripts/smoke_test.sh`.
