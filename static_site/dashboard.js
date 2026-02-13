(function () {
  const backendMeta = document.querySelector('meta[name="backend-url"]');
  const BACKEND = (window.BACKEND_URL || backendMeta?.content || "https://qualificador-leads-ia.onrender.com").replace(/\/$/, "");

  function $(sel, root = document) {
    return root.querySelector(sel);
  }
  function $all(sel, root = document) {
    return Array.from(root.querySelectorAll(sel));
  }

  function getStoredFirst(keys) {
    for (const key of keys) {
      try {
        const v = (sessionStorage.getItem(key) || localStorage.getItem(key) || "").trim();
        if (v) return v;
      } catch (_) {}
    }
    return "";
  }
  function getApiKey() {
    return getStoredFirst(["leadrank_api_key", "api_key", "LR_API_KEY"]);
  }
  function getClientId() {
    return getStoredFirst(["leadrank_client_id", "client_id", "LR_CLIENT_ID"]);
  }

  function setState(state) {
    const wrap = $("#actionList") || $(".action-list");
    if (wrap) wrap.dataset.state = state; // loading | empty | error | ready
  }

  function normalizeTemp(item) {
    const t = (item.temperature || item.temperatura || item.temp || "").toLowerCase();
    if (t) return t;
    const prob = Number(item.probabilidade ?? item.prob ?? 0);
    const score = Number(item.score ?? 0);
    if (prob >= 0.7 || score >= 70) return "hot";
    if (prob >= 0.35 || score >= 35) return "warm";
    return "cold";
  }

  function pillForTemp(t) {
    if (t === "hot") return `<span class="pill pill--muted">🔥 Quente</span>`;
    if (t === "warm") return `<span class="pill pill--muted">🟡 Morno</span>`;
    return `<span class="pill pill--muted">❄️ Frio</span>`;
  }

  function renderList(items) {
    const container = $("#actionReady");
    if (!container) return;
    container.innerHTML = "";

    items.forEach((lead) => {
      const t = normalizeTemp(lead);

      const row = document.createElement("div");
      row.className = "lead-row";
      row.tabIndex = 0;
      row.setAttribute("role", "button");

      row.dataset.openDetails = "";
      row.dataset.leadId = lead.id ?? "";
      row.dataset.nome = lead.nome ?? "";
      row.dataset.telefone = lead.telefone ?? "";
      row.dataset.email = lead.email_lead ?? lead.email ?? "";
      row.dataset.origem = lead.origem ?? "";
      row.dataset.score = lead.score ?? "";
      row.dataset.prob = (lead.probabilidade != null ? String(lead.probabilidade) : (lead.prob ?? "")) ?? "";

      row.innerHTML = `
        <div>
          <p class="lead-row__name">${row.dataset.nome || "Lead"}</p>
          <p class="text-muted">${row.dataset.origem || "—"} • ${row.dataset.telefone || "—"}</p>
        </div>
        <div class="lead-row__meta">
          ${pillForTemp(t)}
          <button class="btn btn--small" type="button" data-open-details
            data-lead-id="${row.dataset.leadId}"
            data-nome="${escapeHtml(row.dataset.nome)}"
            data-telefone="${escapeHtml(row.dataset.telefone)}"
            data-email="${escapeHtml(row.dataset.email)}"
            data-origem="${escapeHtml(row.dataset.origem)}"
            data-score="${escapeHtml(row.dataset.score)}"
            data-prob="${escapeHtml(row.dataset.prob)}"
          >Ver detalhes</button>
        </div>
      `;

      container.appendChild(row);
    });
  }

  function escapeHtml(s) {
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  async function fetchActionList() {
    if (!BACKEND) {
      console.error("BACKEND_URL não definido");
      setState("error");
      return;
    }

    const apiKey = getApiKey();
    const clientId = getClientId();

    if (!clientId) {
      console.warn("client_id ausente no localStorage");
      setState("empty");
      return;
    }
    if (!apiKey) {
      console.warn("api_key ausente no localStorage");
      setState("error");
      return;
    }

    setState("loading");
    try {
      const resp = await fetch(`${BACKEND}/acao_do_dia`, {
        headers: { "X-API-KEY": apiKey, "X-CLIENT-ID": clientId }
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const data = await resp.json();
      const list = data.action_list || data.rows || data.items || [];
      if (!Array.isArray(list) || list.length === 0) {
        setState("empty");
        return;
      }

      renderList(list);
      setState("ready");
    } catch (err) {
      console.error("Erro ao carregar /acao_do_dia", err);
      setState("error");
    }
  }

  function wireUI() {
    const retry = document.querySelector("[data-retry]");
    if (retry) retry.addEventListener("click", fetchActionList);

    const seed10 = document.querySelector("[data-seed10]");
    if (seed10) {
      seed10.addEventListener("click", async () => {
        const apiKey = getApiKey();
        const clientId = getClientId();
        if (!apiKey || !clientId) return;

        try {
          await fetch(`${BACKEND}/seed_test_leads`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-API-KEY": apiKey,
              "X-CLIENT-ID": clientId
            },
            body: JSON.stringify({ client_id: clientId, n: 10 })
          });
        } catch (e) {
          console.warn("seed_test_leads falhou", e);
        }
        fetchActionList();
      });
    }

    const exportBtn = document.querySelector("[data-export]");
    if (exportBtn) {
      exportBtn.addEventListener("click", () => {
        const clientId = getClientId();
        const apiKey = getApiKey();
        if (!clientId) return;
        if (!apiKey) return;

        // Download com headers (window.open não permite headers).
        fetch(`${BACKEND}/leads_export.csv`, {
          headers: { "X-API-KEY": apiKey, "X-CLIENT-ID": clientId },
        })
          .then((resp) => {
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            return resp.blob();
          })
          .then((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `leads_${clientId}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
          })
          .catch((e) => console.warn("Falha ao exportar CSV", e));
      });
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    wireUI();
    fetchActionList();
  });
})();
