import csv
from pathlib import Path

from flask import Blueprint, jsonify, request

import sentry_sdk

from extensions import limiter
from services import settings
from services.db import db
from services.utils import iso, json_ok, now_utc
from services.lead_service import lead_temperature

core_bp = Blueprint("core", __name__)


@core_bp.get("/")
@limiter.limit("100 per minute")
def root():
    return json_ok({"service": "LeadRank backend", "ts": iso(now_utc())})


@core_bp.get("/health")
@limiter.limit("100 per minute")
def health():
    return json_ok({"ts": iso(now_utc())})


@core_bp.get("/health_db")
@limiter.limit("100 per minute")
def health_db():
    if not settings.DATABASE_URL:
        return json_ok({"db": False, "error": "DATABASE_URL missing", "ts": iso(now_utc())})
    conn = None
    try:
        conn = db()
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return json_ok({"db": True, "error": "", "ts": iso(now_utc())})
    except Exception as exc:
        return json_ok({"db": False, "error": repr(exc), "ts": iso(now_utc())})
    finally:
        if conn is not None:
            conn.close()


@core_bp.get("/pricing")
@limiter.limit("100 per minute")
def pricing():
    return json_ok(
        {
            "plans": settings.PLAN_CATALOG,
            "currency": "BRL",
            "checkout_enabled": bool(settings.STRIPE_SECRET_KEY and settings.STRIPE_PRICE_IDS_JSON),
            "ts": iso(now_utc()),
        }
    )


@core_bp.get("/public_config")
@limiter.limit("300 per minute")
def public_config():
    """Config público para o front-end.

    Importante: não inclui segredos. Somente flags e chaves públicas.
    """

    captcha_mode = "off"
    if settings.TURNSTILE_SECRET_KEY and settings.TURNSTILE_SITE_KEY:
        captcha_mode = "enforce" if settings.CAPTCHA_ENFORCE else "soft"

    return jsonify(
        {
            "ok": True,
            "ts": iso(now_utc()),
            "client_error_sample_rate": settings.CLIENT_ERROR_SAMPLE_RATE,
            "features": {
                "demo": bool(settings.DEMO_MODE),
                "client_error_reporting": bool(settings.CLIENT_ERROR_REPORTING),
            },
            "captcha": {
                "provider": "turnstile",
                "mode": captcha_mode,
                "site_key": settings.TURNSTILE_SITE_KEY if captcha_mode != "off" else "",
            },
        }
    )


@core_bp.post("/client_error")
@limiter.limit("60 per minute")
def client_error():
    """Recebe erros do front (opcional).

    Não deve bloquear o usuário; serve só para observabilidade.
    """

    if not settings.CLIENT_ERROR_REPORTING:
        return json_ok({"enabled": False})

    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()[:300]
    stack = (data.get("stack") or "").strip()[:2000]
    page = (data.get("page") or "").strip()[:300]

    # Envia para Sentry como breadcrumb/mensagem (se DSN estiver configurado).
    try:
        if settings.SENTRY_DSN and msg:
            sentry_sdk.capture_message(f"client_error: {msg}")
    except Exception:
        pass

    # Log estruturado
    try:
        import structlog

        log = structlog.get_logger()
        log.warning(
            "client_error",
            message=msg,
            page=page,
            user_agent=(request.headers.get("User-Agent") or "")[:200],
            stack=stack,
        )
    except Exception:
        pass

    return json_ok({"enabled": True})


@core_bp.get("/demo/acao_do_dia")
@limiter.limit("120 per minute")
def demo_acao_do_dia():
    """Endpoint read-only com dados de exemplo.

    Safe: só fica ativo quando DEMO_MODE=true.
    """

    if not settings.DEMO_MODE:
        return json_ok({"enabled": False})

    # Lê o CSV de exemplo do static_site (sem depender do banco).
    base = Path(__file__).resolve().parent.parent
    sample_path = base / "static_site" / "assets" / "leads-sample.csv"
    rows = []
    try:
        with sample_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, r in enumerate(reader, start=1):
                score = int(float(r.get("Score") or 0))
                prob = max(0.0, min(1.0, score / 100.0))
                rows.append(
                    {
                        "id": idx,
                        "nome": (r.get("Nome") or "").strip(),
                        "email": (r.get("Email") or "").strip(),
                        "empresa": (r.get("Empresa") or "").strip(),
                        "score": score,
                        "probabilidade": prob,
                        "interesse": (r.get("Interesse") or "").strip(),
                        "origem": "Demo",
                        "virou_cliente": None,
                        "temperatura": lead_temperature(prob, score),
                    }
                )
                if len(rows) >= 30:
                    break
    except Exception as exc:
        return json_ok({"enabled": True, "rows": [], "error": repr(exc)})

    return json_ok({"enabled": True, "rows": rows})
