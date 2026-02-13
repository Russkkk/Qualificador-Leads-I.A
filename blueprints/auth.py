from datetime import timedelta
import secrets

from flask import Blueprint, jsonify, request
from flask_login import login_user
from psycopg.rows import dict_row
import structlog

from extensions import limiter
from models.user import AuthUser
from services import settings
from services.auth_service import (
    gen_api_key,
    hash_password,
    needs_rehash,
    validate_password_strength,
    verify_password,
)
from services.captcha import verify_turnstile
from services.db import db
from services.utils import iso, json_err, json_ok, month_key, now_utc

auth_bp = Blueprint("auth", __name__)


@auth_bp.post("/signup")
@limiter.limit("5 per hour")
def signup():
    data = request.get_json(silent=True) or request.form or {}
    nome = (data.get("nome") or "").strip()
    email = (data.get("email") or "").strip().lower()
    empresa = (data.get("empresa") or "").strip()
    telefone = (data.get("telefone") or "").strip()
    password = (data.get("password") or data.get("senha") or "").strip()

    # Honeypot (anti-bot): bots costumam preencher campos invisíveis.
    # Safe: não afeta usuários reais; não cria conta quando acionado.
    honeypot = (data.get("company_site") or data.get("website") or "").strip()
    if honeypot:
        return jsonify({"ok": True, "success": True, "message": "Conta trial criada com sucesso!"})

    # Captcha (Cloudflare Turnstile) - opcional.
    # - Em modo enforce, exige token válido.
    # - Em modo soft, tenta validar quando presente, mas não bloqueia por instabilidade.
    captcha_token = (
        data.get("captcha_token")
        or data.get("cf_turnstile_response")
        or data.get("cf-turnstile-response")
        or data.get("cf_turnstile")
        or ""
    )
    if (settings.TURNSTILE_SECRET_KEY and settings.CAPTCHA_ENFORCE) and not str(captcha_token or "").strip():
        return json_err("Confirme que você não é um robô.", 400, code="captcha_required")

    if settings.TURNSTILE_SECRET_KEY and str(captcha_token or "").strip():
        remote = (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip()
        res = verify_turnstile(str(captcha_token), remoteip=remote)
        if not res.ok and settings.CAPTCHA_ENFORCE:
            return json_err("Falha na verificação anti-spam. Tente novamente.", 400, code="captcha_invalid")
        if not res.ok:
            structlog.get_logger().warning("captcha_soft_fail", reason=res.error)

    if not email or "@" not in email:
        return json_err("Email válido é obrigatório", 400)
    ok_pw, pw_msg = validate_password_strength(password or "")
    if not ok_pw:
        return json_err(pw_msg, 400)

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT client_id FROM clients WHERE email=%s", (email,))
                row = cur.fetchone()
                if row:
                    return (
                        jsonify(
                            {
                                "ok": False,
                                "success": False,
                                "error": "Este email já está cadastrado. Faça login.",
                                "code": "email_exists",
                            }
                        ),
                        409,
                    )

                client_id = f"trial-{secrets.token_hex(8)}"
                api_key = gen_api_key(client_id)
                valid_until = now_utc() + timedelta(days=14)
                pw_hash = hash_password(password)

                cur.execute(
                    """
                    INSERT INTO clients (
                        client_id, nome, email, empresa, telefone, valid_until,
                        api_key, plan, status, usage_month, leads_used_month,
                        password_hash, created_at, updated_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,'trial','active',%s,0,%s,NOW(),NOW())
                    """,
                    (
                        client_id,
                        nome or None,
                        email,
                        empresa or None,
                        telefone or None,
                        valid_until,
                        api_key,
                        month_key(),
                        pw_hash,
                    ),
                )

        response = jsonify(
            {
                "ok": True,
                "success": True,
                "client_id": client_id,
                "plan": "trial",
                "valid_until": iso(valid_until),
                "message": "Conta trial criada com sucesso!",
            }
        )
        response.headers["X-API-KEY"] = api_key
        response.headers["Authorization"] = f"Bearer {api_key}"
        return response
    finally:
        conn.close()


@auth_bp.post("/login")
@limiter.limit("10 per 5 minute")
def login():
    data = request.get_json(silent=True) or request.form or {}
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or data.get("senha") or "").strip()

    if not email or "@" not in email:
        return json_err("Email válido é obrigatório", 400)
    if not password:
        return json_err("Senha é obrigatória", 400)

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT client_id, api_key, password_hash, plan, status, valid_until FROM clients WHERE email=%s",
                    (email,),
                )
                row = cur.fetchone()
                if not row:
                    return json_err("Conta não encontrada", 404)

                pw_hash = (row.get("password_hash") or "").strip()
                if not pw_hash:
                    return json_err("Conta sem senha. Use o suporte.", 400)

                if not verify_password(pw_hash, password):
                    return json_err("Email ou senha inválidos", 401)

                if needs_rehash(pw_hash):
                    new_hash = hash_password(password)
                    cur.execute(
                        "UPDATE clients SET password_hash=%s, updated_at=NOW() WHERE client_id=%s",
                        (new_hash, row["client_id"]),
                    )

                api_key = (row.get("api_key") or "").strip()
                if not api_key:
                    api_key = gen_api_key(row["client_id"])
                    cur.execute(
                        "UPDATE clients SET api_key=%s, updated_at=NOW() WHERE client_id=%s",
                        (api_key, row["client_id"]),
                    )

                cur.execute("UPDATE clients SET last_login_at=NOW(), updated_at=NOW() WHERE client_id=%s", (row["client_id"],))

        login_user(AuthUser(client_id=row["client_id"], email=email, plan=row.get("plan") or "trial", status=row.get("status") or "active"))

        response = jsonify(
            {
                "ok": True,
                "success": True,
                "client_id": row.get("client_id"),
                "plan": (row.get("plan") or "trial"),
                "status": (row.get("status") or "active"),
                "valid_until": iso(row.get("valid_until")),
                "message": "Login realizado com sucesso.",
            }
        )
        response.headers["X-API-KEY"] = api_key
        response.headers["Authorization"] = f"Bearer {api_key}"
        return response
    finally:
        conn.close()
