const leadForm = document.getElementById("leadForm");
const formStatus = document.getElementById("formStatus");
const backendMeta = document.querySelector('meta[name="backend-url"]');
// Evita hardcode de ambiente: usa meta quando definida, senão o host atual.
const BACKEND = (window.BACKEND_URL || backendMeta?.content || window.location.origin || "").replace(/\/$/, "");
const THANK_YOU_URL = "obrigado.html";
const OFFLINE_META_KEY = "leadrank_lead_offline_meta";
const STORAGE_TTL_MS = 24 * 60 * 60 * 1000;

const normalizePhone = (raw) => String(raw || "").replace(/\D+/g, "");

const formatPhoneBR = (raw) => {
  const digits = normalizePhone(raw).slice(0, 11);
  if (digits.length < 3) return digits;
  const ddd = digits.slice(0, 2);
  const rest = digits.slice(2);
  if (rest.length <= 4) return `(${ddd}) ${rest}`;
  if (rest.length <= 8) return `(${ddd}) ${rest.slice(0, 4)}-${rest.slice(4)}`;
  return `(${ddd}) ${rest.slice(0, 5)}-${rest.slice(5)}`;
};

if (window.lucide) {
  window.lucide.createIcons();
}

const bindPhoneMask = () => {
  if (!leadForm) return;
  const phoneInput = leadForm.querySelector('input[name="telefone"]');
  if (!phoneInput) return;
  phoneInput.addEventListener("input", () => {
    const next = formatPhoneBR(phoneInput.value);
    if (next !== phoneInput.value) phoneInput.value = next;
  });
};

bindPhoneMask();

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

const safeStorageGet = (key) => {
  try {
    return window.localStorage.getItem(key);
  } catch (_) {
    return null;
  }
};

const safeStorageSet = (key, value) => {
  try {
    window.localStorage.setItem(key, value);
    return true;
  } catch (_) {
    return false;
  }
};

const safeStorageRemove = (key) => {
  try {
    window.localStorage.removeItem(key);
  } catch (_) {}
};

const readJson = (raw) => {
  try {
    return raw ? JSON.parse(raw) : null;
  } catch (_) {
    return null;
  }
};

const cleanupSensitiveLocalData = () => {
  // Remove legado com PII armazenada em texto puro.
  safeStorageRemove("leadrank_leads");
  safeStorageRemove("leadrank_last_lead_email");

  // Remove pending antigo que possua email.
  const pending = readJson(safeStorageGet("leadrank_lead_pending"));
  if (pending && typeof pending === "object" && String(pending.email || "").trim()) {
    safeStorageRemove("leadrank_lead_pending");
  }

  // TTL para metadado offline sem PII.
  const meta = readJson(safeStorageGet(OFFLINE_META_KEY));
  const expiresAt = Number(meta?.expires_at || 0);
  if (!meta || !expiresAt || Date.now() > expiresAt) {
    safeStorageRemove(OFFLINE_META_KEY);
  }
};

const saveOfflineAttemptMeta = ({ clientId = "" } = {}) => {
  const now = Date.now();
  const payload = {
    source: "landing_form",
    client_id: String(clientId || "").trim(),
    created_at: new Date(now).toISOString(),
    expires_at: now + STORAGE_TTL_MS,
  };
  safeStorageSet(OFFLINE_META_KEY, JSON.stringify(payload));
};

cleanupSensitiveLocalData();

// A conversão (Lead) é disparada apenas na página de Obrigado (um único lugar),
// via site-shell.js. Aqui só gravamos um payload "pending" e redirecionamos.
const setPendingLeadConversion = ({ email = "", clientId = "" } = {}) => {
  const payload = {
    // Sem PII em storage persistente: mantemos só metadados não sensíveis.
    client_id: String(clientId).trim(),
    source: "landing_form",
    submitted_at: new Date().toISOString(),
  };

  if (window.LeadrankTracking?.setPendingLeadConversion) {
    window.LeadrankTracking.setPendingLeadConversion(payload);
    return;
  }

  safeStorageSet("leadrank_lead_pending", JSON.stringify(payload));
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

const storageSetSession = (key, value) => {
  try {
    window.sessionStorage.setItem(key, String(value ?? ""));
  } catch (_) {}
};

const storageSetPersist = (key, value) => {
  try {
    window.localStorage.setItem(key, String(value ?? ""));
  } catch (_) {}
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
          telefone: normalizePhone(payload.telefone || ""),
          password: tempPassword,
          // Honeypot (anti-bot)
          company_site: payload.company_site || "",
          // Captcha (Turnstile) - quando habilitado no servidor, o token é preenchido via site-shell.js
          captcha_token:
            window.__LEADRANK_CAPTCHA_TOKEN ||
            payload["cf-turnstile-response"] ||
            payload["cf_turnstile_response"] ||
            "",
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
        // client_id não é segredo: pode persistir.
        storageSetPersist("leadrank_client_id", clientId);
        storageSetPersist("client_id", clientId);
      }
      if (apiKey) {
        // API key é sensível: guarde por padrão apenas na sessão.
        storageSetSession("leadrank_api_key", apiKey);
        storageSetSession("api_key", apiKey);
      }
      // Senha temporária é altamente sensível: evita persistência em localStorage.
      // Mantemos apenas na sessão para exibir na página de Obrigado.
      storageSetSession("leadrank_temp_password", tempPassword);

      // Segurança/LGPD: evita persistir email em storage do navegador.
      safeStorageRemove("leadrank_last_lead_email");

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
      saveOfflineAttemptMeta();
      showStatus(
        "Falha de conexão. Registramos apenas uma tentativa local sem dados pessoais.",
        "warning"
      );
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  });
}
