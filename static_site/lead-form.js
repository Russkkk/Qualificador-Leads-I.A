const leadForm = document.getElementById("leadForm");
const formStatus = document.getElementById("formStatus");
const backendMeta = document.querySelector('meta[name="backend-url"]');
const BACKEND = (backendMeta?.content || "https://qualificador-leads-ia.onrender.com").replace(/\/$/, "");

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

      leadForm.reset();
      showStatus(
        `Conta criada! Senha temporária: ${tempPassword}. Acesse o onboarding para continuar.`,
        "success"
      );
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
