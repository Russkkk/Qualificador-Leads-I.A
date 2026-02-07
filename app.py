import re

from flask import Flask, request
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
from services.utils import json_err, log_exception


app = Flask(__name__)
app.secret_key = settings.FLASK_SECRET_KEY or "dev-secret"

CORS(app, resources={r"/*": {"origins": settings.ALLOWED_ORIGINS}})
configure_logging()
init_sentry()

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

