# popular_db.py
# -------------
# Seed de leads no Postgres (Render) para um client_id.
#
# Uso:
#   export DATABASE_URL="postgres://..."
#   export SEED_CLIENT_ID="demo_seed"
#   export SEED_N="30"
#   python popular_db.py

import os
import json
import random
import secrets
import time
import hashlib
from datetime import datetime, timezone

try:
    import psycopg
except Exception as e:
    raise SystemExit(
        "ERRO: instale psycopg[binary]. Ex.: pip install 'psycopg[binary]'\n" + repr(e)
    )

DATABASE_URL = (os.environ.get("DATABASE_URL") or "").strip()
SEED_CLIENT_ID = (os.environ.get("SEED_CLIENT_ID") or "demo_seed").strip()
SEED_N = int((os.environ.get("SEED_N") or "30").strip())


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


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                id BIGSERIAL PRIMARY KEY,
                client_id TEXT NOT NULL,
                nome TEXT,
                email_lead TEXT,
                telefone TEXT,
                origem TEXT,
                tempo_site INTEGER,
                paginas_visitadas INTEGER,
                clicou_preco INTEGER,
                probabilidade DOUBLE PRECISION,
                virou_cliente DOUBLE PRECISION,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                score INTEGER,
                label INTEGER,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_leads_client_created ON leads(client_id, created_at DESC);")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS clients (
                client_id TEXT PRIMARY KEY,
                api_key TEXT,
                plan TEXT NOT NULL DEFAULT 'trial',
                status TEXT NOT NULL DEFAULT 'active',
                usage_month TEXT NOT NULL DEFAULT '',
                leads_used_month INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS api_key TEXT;")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'trial';")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS usage_month TEXT;")
        cur.execute("ALTER TABLE clients ADD COLUMN IF NOT EXISTS leads_used_month INTEGER NOT NULL DEFAULT 0;")

        mk = _month_key()
        cur.execute("UPDATE clients SET usage_month=%s WHERE usage_month IS NULL OR usage_month='';", (mk,))
        cur.execute("UPDATE clients SET api_key='' WHERE api_key IS NULL;")


def heuristic_prob(tempo_site: int, paginas: int, clicou_preco: int, nome: str, telefone: str) -> float:
    base = 0.10
    base += min(tempo_site / 400, 0.25)
    base += min(paginas / 10, 0.25)
    base += 0.20 if clicou_preco else 0.0
    if telefone and len(telefone) >= 10:
        base += 0.06
    if nome and len(nome) >= 4:
        base += 0.04
    return max(0.02, min(0.98, base))


def main():
    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL ausente. Configure DATABASE_URL para popular o Postgres.")

    conn = psycopg.connect(DATABASE_URL)
    try:
        with conn:
            ensure_schema(conn)

            with conn.cursor() as cur:
                cur.execute("SELECT client_id, api_key FROM clients WHERE client_id=%s", (SEED_CLIENT_ID,))
                row = cur.fetchone()
                if not row:
                    api_key = _gen_api_key(SEED_CLIENT_ID)
                    mk = _month_key()
                    cur.execute(
                        """
                        INSERT INTO clients (client_id, api_key, plan, status, usage_month, leads_used_month, created_at, updated_at)
                        VALUES (%s,%s,'demo','active',%s,0,NOW(),NOW())
                        """,
                        (SEED_CLIENT_ID, api_key, mk),
                    )
                    print(f"✅ Workspace criado: client_id={SEED_CLIENT_ID} api_key={api_key}")
                else:
                    print(f"ℹ️ Workspace já existe: client_id={SEED_CLIENT_ID}")

                inserted = 0
                for i in range(SEED_N):
                    tempo_site = random.randint(10, 520)
                    paginas = random.randint(1, 12)
                    clicou = random.choice([0, 1])
                    nome = "Demo " + secrets.token_hex(2).upper()
                    email = f"demo{i}@leadrank.local"
                    telefone = "11" + str(random.randint(900000000, 999999999))
                    origem = random.choice(["google", "instagram", "whatsapp", "indicacao", "desconhecida"])

                    prob = heuristic_prob(tempo_site, paginas, clicou, nome, telefone)
                    score = int(round(prob * 100))
                    label = 1 if prob >= 0.70 else (0 if prob < 0.35 else None)
                    virou_cliente = random.choices([None, 1.0, 0.0], weights=[0.55, 0.25, 0.20])[0]

                    payload = {
                        "nome": nome,
                        "email": email,
                        "telefone": telefone,
                        "origem": origem,
                        "tempo_site": tempo_site,
                        "paginas_visitadas": paginas,
                        "clicou_preco": clicou,
                    }

                    cur.execute(
                        """
                        INSERT INTO leads
                          (client_id, nome, email_lead, telefone, origem, tempo_site, paginas_visitadas, clicou_preco,
                           payload, probabilidade, score, label, virou_cliente, created_at, updated_at)
                        VALUES
                          (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW())
                        """,
                        (
                            SEED_CLIENT_ID,
                            nome,
                            email,
                            telefone,
                            origem,
                            tempo_site,
                            paginas,
                            clicou,
                            json.dumps(payload),
                            float(prob),
                            int(score),
                            label,
                            virou_cliente,
                        ),
                    )
                    inserted += 1

                print(f"✅ Leads inseridos: {inserted}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
