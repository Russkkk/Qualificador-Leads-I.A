CREATE INDEX IF NOT EXISTS idx_leads_client_created ON leads(client_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_leads_client_label ON leads(client_id, virou_cliente);
CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_api_key ON clients(api_key) WHERE api_key <> '';
CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_email ON clients(email) WHERE email IS NOT NULL AND email <> '';
CREATE INDEX IF NOT EXISTS idx_billing_events_client_created ON billing_events(client_id, created_at DESC);
