from typing import Dict, Tuple

from flask import request

from services import settings
from services.utils import get_header

_DEMO_RL: Dict[str, int] = {}
_DEMO_RL_MAX = 5


def require_demo_key() -> Tuple[bool, str | None]:
    expected = (settings.DEMO_KEY or "").strip()
    if not expected:
        return False, "DEMO_KEY não configurada no ambiente"

    got = get_header("X-DEMO-KEY")
    if not got:
        auth = get_header("Authorization")
        if auth.lower().startswith("bearer "):
            got = auth.split(" ", 1)[1].strip()

    if not got:
        got = (request.args.get("demo_key") or "").strip()

    if not got:
        data = request.get_json(silent=True) or {}
        got = (data.get("demo_key") or "").strip()

    if not got:
        return False, "DEMO_KEY ausente"

    if got != expected:
        return False, "DEMO_KEY inválida"

    return True, None


def check_demo_key() -> bool:
    ok, _ = require_demo_key()
    return ok


def demo_rate_limited(key: str) -> bool:
    return _DEMO_RL.get(key, 0) >= _DEMO_RL_MAX


def bump_demo_counter(key: str) -> int:
    _DEMO_RL[key] = _DEMO_RL.get(key, 0) + 1
    return _DEMO_RL[key]
