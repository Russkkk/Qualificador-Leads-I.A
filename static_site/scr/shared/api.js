const backendMeta = document.querySelector('meta[name="backend-url"]');
const backend = (backendMeta?.content || "https://qualificador-leads-ia.onrender.com").replace(/\/$/, "");

const getFirstStored = (keys) => {
  for (const key of keys) {
    try {
      const v = (sessionStorage.getItem(key) || localStorage.getItem(key) || "").trim();
      if (v) return v;
    } catch (_) {}
  }
  return "";
};

const getApiKey = () => getFirstStored(["leadrank_api_key", "api_key", "LR_API_KEY"]);
const getClientId = () => getFirstStored(["leadrank_client_id", "client_id", "LR_CLIENT_ID"]);

const authHeaders = () => {
  const apiKey = getApiKey();
  const clientId = getClientId();
  return {
    ...(apiKey ? { "X-API-KEY": apiKey } : {}),
    ...(clientId ? { "X-CLIENT-ID": clientId } : {})
  };
};

const requestJson = async (path, options = {}) => {
  const response = await fetch(`${backend}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
      ...authHeaders()
    }
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = data.error || data.message || `Erro ${response.status}`;
    throw new Error(message);
  }
  return data;
};

const getActionList = async () => {
  const data = await requestJson("/acao_do_dia", { method: "GET" });
  return data.action_list || data.items || data.rows || [];
};

const seedTestLeads = async (count = 10) => {
  const clientId = getClientId();
  return requestJson("/seed_test_leads", {
    method: "POST",
    body: JSON.stringify({ client_id: clientId, n: count })
  });
};

const exportCsv = async () => {
  const response = await fetch(`${backend}/leads_export.csv`, { headers: authHeaders() });
  if (!response.ok) {
    throw new Error(`Erro ${response.status}`);
  }
  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") || "";
  const match = /filename="([^"]+)"/i.exec(disposition);
  const filename = match?.[1] || `leadrank_${getClientId() || "export"}.csv`;
  return { blob, filename };
};

export {
  backend,
  getActionList,
  seedTestLeads,
  exportCsv,
  requestJson,
  getClientId,
  getApiKey
};
