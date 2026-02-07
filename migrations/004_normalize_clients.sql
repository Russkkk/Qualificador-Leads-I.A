ALTER TABLE clients ALTER COLUMN api_key DROP NOT NULL;
UPDATE clients SET usage_month = TO_CHAR(NOW(), 'YYYY-MM') WHERE usage_month IS NULL OR usage_month = '';
UPDATE clients SET api_key = '' WHERE api_key IS NULL;
UPDATE clients SET updated_at = NOW() WHERE updated_at IS NULL;
