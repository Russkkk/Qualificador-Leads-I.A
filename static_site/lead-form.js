const leadForm = document.getElementById("leadForm");
const formStatus = document.getElementById("formStatus");
const backendMeta = document.querySelector('meta[name="backend-url"]');
const BACKEND = (backendMeta?.content || "https://qualificador-leads-ia.onrender.com").replace(/\/$/, "");
const THANK_YOU_URL = "obrigado.html";

if (window.lucide) {
  window.lucide.createIcons();
}

const showStatus = (message, tone = "success") => {
  if (!formStatus) return;
  formStatus.textContent = message;
  formStatus.dataset.tone = tone;
};

const generatePassword = () => {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789";
  const symbols = "!@#$%";
  const pick = (set) => set[Math.floor(Math.random() * set.length)];
  let result = "";
  for (let i = 0; i < 8; i += 1) {
    result += pick(alphabet);
  }
  result += pick(symbols);
  result += pick("0123456789");
  return result;
};

const saveLeadLocally = (payload) => {
  const stored = localStorage.getItem("leadrank_leads");
  const leads = stored ? JSON.parse(stored) : [];
  leads.push({
    ...payload,
    createdAt: new Date().toISOString(),
  });
  localStorage.setItem("leadrank_leads", JSON.stringify(leads));
};

const serializeForm = (form) => {
  const data = {};
  new FormData(form).forEach((value, key) => {
    data[key] = value;
  });
  return data;
};

// A conversão (Lead) é disparada apenas na página de Obrigado (um único lugar),
// via site-shell.js. Aqui só gravamos um payload "pending" e redirecionamos.
const setPendingLeadConversion = ({ email = "", clientId = "" } = {}) => {
  const payload = {
    email: String(email).trim().toLowerCase(),
    client_id: String(clientId).trim(),
    source: "landing_form",
    submitted_at: new Date().toISOString(),
  };

  if (window.LeadrankTracking?.setPendingLeadConversion) {
    window.LeadrankTracking.setPendingLeadConversion(payload);
    return;
  }

  try {
    localStorage.setItem("leadrank_lead_pending", JSON.stringify(payload));
  } catch (_) {}
};

const showThankYouState = (tempPassword) => {
  if (!leadForm) return;
  leadForm.innerHTML = `
    <div class="space-y-3" role="status" aria-live="polite">
      <p class="text-sm font-semibold text-emerald-300">Obrigado! Seu cadastro foi concluído.</p>
      <p class="text-sm text-slate-200">Conta trial criada com sucesso. Senha temporária: <strong>${tempPassword}</strong>.</p>
      <p class="text-xs text-slate-400">Você também pode seguir para o onboarding para finalizar a configuração.</p>
      <a class="btn btn--block" href="onboarding.html" aria-label="Ir para onboarding após criar conta">Continuar para onboarding</a>
    </div>
  `;
};

if (leadForm) {
  leadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(leadForm);
    const payload = Object.fromEntries(formData.entries());
    const tempPassword = generatePassword();
    const submitBtn = leadForm.querySelector("button[type='submit']");

    try {
      if (submitBtn) submitBtn.disabled = true;
      showStatus("Criando sua conta...", "warning");
      const response = await fetch(`${BACKEND}/signup`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          nome: payload.name || "",
          email: payload.email || "",
          telefone: payload.telefone || "",
          password: tempPassword,
        }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        const msg = data?.error || data?.message || "Não conseguimos criar sua conta agora.";
        showStatus(msg, "error");
        return;
      }

      const clientId = data?.client_id || "";
      const apiKey = response.headers.get("X-API-KEY") || "";
      if (clientId) {
        localStorage.setItem("leadrank_client_id", clientId);
        localStorage.setItem("client_id", clientId);
      }
      if (apiKey) {
        localStorage.setItem("leadrank_api_key", apiKey);
        localStorage.setItem("api_key", apiKey);
      }
      localStorage.setItem("leadrank_temp_password", tempPassword);

      // Para mostrar dados na página de Obrigado.
      localStorage.setItem("leadrank_last_lead_email", String(payload.email || "").trim());

      setPendingLeadConversion({ email: payload.email || "", clientId });

      leadForm.reset();
      showStatus("Cadastro concluído. Redirecionando...", "success");

      try {
        window.location.assign(THANK_YOU_URL);
        return;
      } catch (_) {
        // Fallback (caso raro): mantém o estado na própria página.
        showThankYouState(tempPassword);
      }
    } catch (error) {
      saveLeadLocally(serializeForm(leadForm));
      showStatus(
        "Falha de conexão. Salvamos o lead localmente para não perder o contato.",
        "warning"
      );
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  });
}
