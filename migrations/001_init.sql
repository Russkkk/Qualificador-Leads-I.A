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

CREATE TABLE IF NOT EXISTS clients (
    client_id TEXT PRIMARY KEY,
    nome TEXT,
    email TEXT,
    empresa TEXT,
    telefone TEXT,
    valid_until TIMESTAMPTZ,
    password_hash TEXT,
    last_login_at TIMESTAMPTZ,
    api_key TEXT,
    plan TEXT NOT NULL DEFAULT 'trial',
    status TEXT NOT NULL DEFAULT 'active',
    usage_month TEXT NOT NULL DEFAULT '',
    leads_used_month INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS thresholds (
    client_id TEXT PRIMARY KEY,
    threshold DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_meta (
    client_id TEXT PRIMARY KEY,
    can_train BOOLEAN NOT NULL DEFAULT FALSE,
    labeled_count INTEGER NOT NULL DEFAULT 0,
    classes_rotuladas TEXT NOT NULL DEFAULT '[]',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS subscriptions (
    client_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT 'manual',
    status TEXT NOT NULL DEFAULT 'inactive',
    plan TEXT NOT NULL DEFAULT 'trial',
    current_period_start TIMESTAMPTZ,
    current_period_end TIMESTAMPTZ,
    cancel_at_period_end BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS billing_events (
    id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT 'manual',
    event_type TEXT NOT NULL,
    client_id TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
