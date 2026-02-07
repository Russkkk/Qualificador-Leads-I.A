import hashlib
import secrets
from typing import Any, Dict, Optional, Tuple

from werkzeug.security import check_password_hash, generate_password_hash

from models.user import AuthUser
from services import settings
from services.db import db, ensure_client_row
from services.utils import get_api_key_from_headers


def validate_password_strength(password: str) -> Tuple[bool, str]:
    if len(password) < 10:
        return False, "Senha deve ter no mínimo 10 caracteres."
    if not any(char.isupper() for char in password):
        return False, "Senha deve conter pelo menos 1 letra maiúscula."
    if not any(char.islower() for char in password):
        return False, "Senha deve conter pelo menos 1 letra minúscula."
    if not any(char.isdigit() for char in password):
        return False, "Senha deve conter pelo menos 1 número."
    if password.isalnum():
        return False, "Senha deve conter pelo menos 1 símbolo."
    return True, ""


def hash_password(password: str) -> str:
    return generate_password_hash(password, method=f"pbkdf2:sha256:{settings.PBKDF2_ITERATIONS}")


def needs_rehash(stored: str) -> bool:
    if stored.startswith("pbkdf2:sha256:"):
        try:
            iterations = int(stored.split(":", 2)[2].split("$", 1)[0])
            return iterations < settings.PBKDF2_ITERATIONS
        except Exception:
            return True
    return True


def verify_password(stored: str, password: str) -> bool:
    return check_password_hash(stored, password)


def gen_api_key(client_id: str) -> str:
    raw = f"{client_id}:{secrets.token_urlsafe(24)}:{secrets.token_hex(4)}"
    return "sk_live_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def require_client_auth(client_id: str) -> Tuple[bool, Dict[str, Any], str]:
    row = ensure_client_row(client_id, plan="trial")
    expected = (row.get("api_key") or "").strip()
    if not expected:
        if settings.REQUIRE_API_KEY:
            api_key = gen_api_key(client_id)
            conn = db()
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE clients SET api_key=%s, updated_at=NOW() WHERE client_id=%s",
                            (api_key, client_id),
                        )
                row["api_key"] = api_key
            finally:
                conn.close()
            return False, row, "api_key necessária. Gere/recupere uma chave e envie no header."
        return True, row, ""

    got = get_api_key_from_headers()
    if got != expected:
        return False, row, "api_key inválida ou ausente."
    return True, row, ""


def load_user(client_id: str) -> Optional[AuthUser]:
    conn = db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT client_id, email, plan, status FROM clients WHERE client_id=%s",
                    (client_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return AuthUser(
                    client_id=row[0],
                    email=row[1] or "",
                    plan=row[2] or "trial",
                    status=row[3] or "active",
                )
    finally:
        conn.close()
