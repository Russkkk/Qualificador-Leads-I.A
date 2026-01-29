# init_users.py
# -------------
# Script utilitário (local/CI) para inicializar um usuário (cliente) no Postgres do Render
# com EMAIL + SENHA (hash seguro PBKDF2).
#
# Uso (opcional):
#   export DATABASE_URL="postgres://..."
#   export INIT_USER_EMAIL="admin@exemplo.com"
#   export INIT_USER_PASSWORD="sua_senha"
#   export INIT_USER_CLIENT_ID="admin"
#   python init_users.py

import os
import time
import secrets
import hashlib
from datetime import datetime, timezone

try:
    import psycopg
except Exception as e:
    raise SystemExit(
        "ERRO: instale psycopg[binary]. Ex.: pip install 'psycopg[binary]'\n" + repr(e)
    )

DATABASE_URL = (os.environ.get("DATABASE_URL") or "").strip()
INIT_USER_EMAIL = (os.environ.get("INIT_USER_EMAIL") or "admin@leadrank.local").strip().lower()
INIT_USER_PASSWORD = (os.environ.get("INIT_USER_PASSWORD") or "Admin@12345").strip()
INIT_USER_CLIENT_ID = (os.environ.get("INIT_USER_CLIENT_ID") or "admin").strip()
INIT_USER_PLAN = (os.environ.get("INIT_USER_PLAN") or "trial").strip().lower()

# se já existir, NÃO sobrescreve senha por padrão
FORCE_RESET_PASSWORD = (os.environ.get("FORCE_RESET_PASSWORD") or "").strip().lower() in ("1", "true", "yes")


def _now_utc():
    return datetime.now(timezone.utc)


def _month_key(dt=None):
    dt = dt or _now_utc()
    return dt.strftime("%Y-%m")


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _gen_api_key(client_id: str) -> str:
    raw = f"{client_id}:{secrets.token_urlsafe(24)}:{time.time()}"
    return "sk_live_" + _sha256(raw)[:32]


def _pbkdf2_hash(password: str, salt: bytes | None = None, iterations: int = 260_000) -> str:
    """
    Retorna string no formato:
      pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
    """
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS clients (
                client_id TEXT PRIMARY KEY,
                api_key TEXT,
                plan TEXT NOT NULL DEFAULT 'trial',
                status TEXT NOT NULL DEFAULT 'active',
                usage_month TEXT NOT NULL DEFAULT '',
                leads_used_month INTEGER NOT NULL DEFAULT 0,
                nome TEXT,
                email TEXT,
                empresa TEXT,
                valid_until TIMESTAMPTZ,
                password_hash TEXT,
                last_login_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        # compat
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS api_key TEXT;")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'trial';")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS usage_month TEXT;")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS leads_used_month INTEGER NOT NULL DEFAULT 0;")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS nome TEXT;")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS email TEXT;")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS empresa TEXT;")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS valid_until TIMESTAMPTZ;")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS password_hash TEXT;")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMPTZ;")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")

        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_api_key ON clients(api_key) WHERE api_key <> '';")
        try:
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_email_unique "
                "ON clients(email) WHERE email IS NOT NULL AND email<>'';"
            )
        except Exception:
            pass

        mk = _month_key()
        cur.execute("UPDATE clients SET usage_month=%s WHERE usage_month IS NULL OR usage_month='';", (mk,))
        cur.execute("UPDATE clients SET api_key='' WHERE api_key IS NULL;")


def main():
    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL ausente. Configure DATABASE_URL para inicializar usuário no Postgres.")

    conn = psycopg.connect(DATABASE_URL)
    try:
        with conn:
            ensure_schema(conn)

            with conn.cursor() as cur:
                cur.execute("SELECT client_id, api_key, password_hash FROM clients WHERE email=%s", (INIT_USER_EMAIL,))
                row = cur.fetchone()

                if row:
                    client_id, api_key, password_hash = row
                    if FORCE_RESET_PASSWORD or not (password_hash or "").strip():
                        new_hash = _pbkdf2_hash(INIT_USER_PASSWORD)
                        if not (api_key or "").strip():
                            api_key = _gen_api_key(client_id)
                        cur.execute(
                            "UPDATE clients SET password_hash=%s, api_key=%s, updated_at=NOW() WHERE client_id=%s",
                            (new_hash, api_key, client_id),
                        )
                        print("✅ Usuário existente: senha definida/redefinida.")
                    else:
                        print("ℹ️ Usuário já existe (senha preservada).")

                    print(f"client_id={client_id}")
                    print(f"api_key={api_key}")
                    return

                password_hash = _pbkdf2_hash(INIT_USER_PASSWORD)
                api_key = _gen_api_key(INIT_USER_CLIENT_ID)
                mk = _month_key()

                cur.execute(
                    """
                    INSERT INTO clients
                      (client_id, api_key, plan, status, usage_month, leads_used_month,
                       nome, email, empresa, password_hash, created_at, updated_at)
                    VALUES
                      (%s,%s,%s,'active',%s,0,NULL,%s,NULL,%s,NOW(),NOW())
                    """,
                    (INIT_USER_CLIENT_ID, api_key, INIT_USER_PLAN, mk, INIT_USER_EMAIL, password_hash),
                )

                print("✅ Usuário criado com sucesso!")
                print(f"client_id={INIT_USER_CLIENT_ID}")
                print(f"api_key={api_key}")
                print(f"email={INIT_USER_EMAIL}")
                print("⚠️ Senha padrão só para DEV. Troque INIT_USER_PASSWORD em produção.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
