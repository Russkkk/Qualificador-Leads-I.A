ALTER TABLE leads
ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_leads_client_deleted_at
ON leads (client_id, deleted_at);
