import re
import secrets
import time
import uuid

from flask import Flask, g, request
import structlog
import structlog.contextvars
from flask_cors import CORS
from flask_limiter.errors import RateLimitExceeded
from werkzeug.exceptions import HTTPException

from blueprints.admin import admin_bp
from blueprints.auth import auth_bp
from blueprints.billing import billing_bp
from blueprints.core import core_bp
from blueprints.leads import leads_bp
from blueprints.ml import ml_bp
from extensions import limiter, login_manager
from services.logging_config import configure_logging, init_sentry
from services import settings
from services.auth_service import load_user
from services.utils import json_err, log_exception, client_ip


app = Flask(__name__)
if settings.FLASK_SECRET_KEY:
    app.secret_key = settings.FLASK_SECRET_KEY
else:
    # Nunca use uma secret fraca em produção. Se o env não estiver configurado,
    # gera uma secret aleatória (sessions serão invalidadas a cada restart).
    app.secret_key = secrets.token_urlsafe(32)
    if not settings.DEBUG_MODE:
        print("[WARN] FLASK_SECRET_KEY não configurada. Gerando chave efêmera. Configure no ambiente para manter sessões estáveis.")


# Hardening básico de cookies/sessão (não afeta API Key, mas protege fluxos que usem login por sessão).
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=not settings.DEBUG_MODE,
)

# Em produção atrás de proxy (Render/Cloudflare/Nginx), habilite TRUST_PROXY=true para detectar HTTPS corretamente.
if settings.TRUST_PROXY:
    try:
        from werkzeug.middleware.proxy_fix import ProxyFix

        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
    except Exception:
        # Fallback silencioso (não deve quebrar startup).
        pass

CORS(app, resources={r"/*": {"origins": settings.ALLOWED_ORIGINS}})
configure_logging()
init_sentry()


@app.before_request
def bind_request_context():
    """Adiciona request_id + contexto básico para logs estruturados."""

    g._start_time = time.perf_counter()
    rid = (request.headers.get("X-Request-ID") or "").strip()
    if not rid or len(rid) > 64:
        rid = uuid.uuid4().hex

    g.request_id = rid
    structlog.contextvars.bind_contextvars(
        request_id=rid,
        method=request.method,
        path=request.path,
    )


@app.after_request
def access_log_and_request_id(response):
    # Header de correlação (ajuda a debugar no front e no suporte)
    try:
        response.headers.setdefault("X-Request-ID", getattr(g, "request_id", ""))
    except Exception:
        pass

    # Loga request (exceto health) em formato JSON (structlog)
    try:
        path = request.path or ""
        if not path.startswith("/health"):
            dur = None
            if hasattr(g, "_start_time"):
                dur = int((time.perf_counter() - g._start_time) * 1000)
            structlog.get_logger().info(
                "http_request",
                status=int(getattr(response, "status_code", 0) or 0),
                duration_ms=dur,
                # client_ip() usa TRUST_PROXY e pega o primeiro IP do X-Forwarded-For (origem)
                # sem depender de ProxyFix. Mantemos também a cadeia bruta para debug.
                client_ip=client_ip()[:100],
                x_forwarded_for=(request.headers.get("X-Forwarded-For") or "")[:200],
                remote_addr=(request.remote_addr or "")[:100],
                user_agent=(request.headers.get("User-Agent") or "")[:200],
            )
    except Exception:
        pass
    finally:
        try:
            structlog.contextvars.clear_contextvars()
        except Exception:
            pass
    return response

limiter.init_app(app)
login_manager.init_app(app)


@login_manager.user_loader
def load_user_callback(user_id: str):
    return load_user(user_id)


@app.errorhandler(Exception)
def handle_exception(err: Exception):
    if isinstance(err, RateLimitExceeded):
        return json_err(
            "Muitas requisições. Tente novamente em instantes.",
            429,
            error_code="rate_limit",
            error_type=err.__class__.__name__,
        )
    if isinstance(err, HTTPException):
        return json_err(
            err.description,
            err.code or 500,
            error_code="http_error",
            error_type=err.__class__.__name__,
        )
    trace = log_exception("Unhandled exception")
    payload = {
        "error_code": "internal_error",
        "code": "internal_error",
        "error_type": err.__class__.__name__,
    }
    if settings.DEBUG_MODE or settings.INCLUDE_TRACEBACK:
        payload["trace"] = trace
    return json_err("Erro interno do servidor", 500, **payload)


def _is_allowed_origin(origin: str) -> bool:
    for entry in settings.ALLOWED_ORIGINS:
        if entry == "null":
            continue
        if entry.startswith("^"):
            if re.match(entry, origin):
                return True
        elif entry == origin:
            return True
    return False


@app.after_request
def apply_security_headers(response):
    origin = request.headers.get("Origin")
    if origin and _is_allowed_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers.setdefault("Vary", "Origin")
    else:
        response.headers.pop("Access-Control-Allow-Origin", None)

    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()")
    # Permite que front-ends em outro domínio leiam headers úteis (quando CORS estiver ativo).
    response.headers.setdefault("Access-Control-Expose-Headers", "X-API-KEY, X-Request-ID, Content-Disposition")
    if settings.CSP_POLICY:
        response.headers.setdefault("Content-Security-Policy", settings.CSP_POLICY)
    if settings.ENABLE_HSTS and request.is_secure:
        response.headers.setdefault(
            "Strict-Transport-Security",
            f"max-age={settings.HSTS_MAX_AGE}; includeSubDomains; preload",
        )
    return response


app.register_blueprint(core_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(leads_bp)
app.register_blueprint(ml_bp)
app.register_blueprint(billing_bp)
app.register_blueprint(admin_bp)

