import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from flask import jsonify, request

from services import settings
from structlog import get_logger

logger = get_logger()


def log_exception(message: str) -> str:
    trace = traceback.format_exc()
    logger.exception(message)
    return trace


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def month_key(dt: Optional[datetime] = None) -> str:
    dt = dt or now_utc()
    return dt.strftime("%Y-%m")


def resp(payload: Dict[str, Any], code: int = 200):
    return jsonify(payload), code


def json_ok(payload: Dict[str, Any], code: int = 200):
    payload.setdefault("ok", True)
    return resp(payload, code)


def json_err(msg: str, code: int = 400, **extra):
    payload = {"ok": False, "error": msg}
    payload.update(extra)
    return resp(payload, code)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def get_header(name: str) -> str:
    return (request.headers.get(name) or "").strip()


def get_api_key_from_headers() -> str:
    key = get_header("X-API-KEY") or get_header("Authorization")
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


def get_client_id_from_request() -> str:
    client_id = get_header("X-CLIENT-ID")
    if client_id:
        return client_id
    # Query string (útil para GETs e downloads).
    client_id = (request.args.get("client_id") or "").strip()
    if client_id:
        return client_id
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    if client_id:
        return client_id
    return (request.form.get("client_id") or "").strip()


def client_ip() -> str:
    if settings.TRUST_PROXY:
        forwarded = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return (request.remote_addr or "unknown").strip()


def rate_limit_client_id() -> str:
    return get_client_id_from_request() or client_ip()
