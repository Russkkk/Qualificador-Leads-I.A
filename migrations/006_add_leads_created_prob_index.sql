CREATE INDEX IF NOT EXISTS idx_leads_client_created_prob
ON leads (client_id, created_at DESC, probabilidade DESC);
