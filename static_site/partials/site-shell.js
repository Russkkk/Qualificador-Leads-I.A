const loadPartial = async (selector, partialPath) => {
  const target = document.querySelector(selector);
  if (!target) {
    return null;
  }

  const response = await fetch(partialPath);
  if (!response.ok) {
    console.warn(`Não foi possível carregar ${partialPath}.`);
    return null;
  }

  const html = await response.text();
  target.innerHTML = html;
  return target;
};

// --- Back-end base URL (para flags públicas, captcha e report de erro do front) ---
const backendMeta = document.querySelector('meta[name="backend-url"]');
// Deploy-safe: prioriza override explícito (window/meta) e, no fallback, usa o host atual.
const BACKEND = (window.BACKEND_URL || backendMeta?.content || window.location.origin || "").replace(/\/$/, "");

const fetchWithTimeout = async (url, options = {}, timeoutMs = 2500) => {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { ...options, signal: controller.signal });
    return resp;
  } finally {
    clearTimeout(t);
  }
};

const loadPublicConfig = async () => {
  try {
    const resp = await fetchWithTimeout(`${BACKEND}/public_config`, { method: "GET" }, 2500);
    if (!resp.ok) return null;
    const data = await resp.json().catch(() => null);
    if (!data || data.ok !== true) return null;
    window.LEADRANK_PUBLIC_CONFIG = data;
    return data;
  } catch (_) {
    return null;
  }
};

const applyFeatureFlags = (config) => {
  const demoEnabled = Boolean(config?.features?.demo);
  document.querySelectorAll('[data-feature="demo"]').forEach((el) => {
    if (demoEnabled) {
      el.hidden = false;
      el.classList.remove("hidden");
      el.removeAttribute("aria-hidden");
    } else {
      el.hidden = true;
      el.classList.add("hidden");
      el.setAttribute("aria-hidden", "true");
    }
  });
};

// --- Captcha (Cloudflare Turnstile) ---
let _turnstileScriptPromise = null;

const loadTurnstileScript = () => {
  if (_turnstileScriptPromise) return _turnstileScriptPromise;
  _turnstileScriptPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector('script[data-turnstile]');
    if (existing) {
      resolve(true);
      return;
    }
    const script = document.createElement("script");
    script.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
    script.async = true;
    script.defer = true;
    script.setAttribute("data-turnstile", "1");
    script.onload = () => resolve(true);
    script.onerror = () => reject(new Error("turnstile_load_failed"));
    document.head.appendChild(script);
  });
  return _turnstileScriptPromise;
};

const initTurnstile = async (config) => {
  const captcha = config?.captcha || {};
  const siteKey = String(captcha.site_key || "").trim();
  const mode = String(captcha.mode || "off").trim();
  if (!siteKey || mode === "off") return;

  const slots = Array.from(document.querySelectorAll("[data-captcha-slot]"));
  if (!slots.length) return;

  try {
    await loadTurnstileScript();
  } catch (_) {
    return; // falha silenciosa (não quebra form)
  }

  // Renderiza em todos os slots encontrados (ex.: landing form, onboarding)
  slots.forEach((slot) => {
    if (!(slot instanceof HTMLElement)) return;
    if (slot.dataset.rendered === "1") return;
    slot.dataset.rendered = "1";
    slot.hidden = false;
    slot.classList.remove("hidden");

    // Turnstile expõe o token via callback; mantemos em window para o lead-form.js anexar.
    const onSuccess = (token) => {
      window.__LEADRANK_CAPTCHA_TOKEN = String(token || "");
    };
    const onExpire = () => {
      window.__LEADRANK_CAPTCHA_TOKEN = "";
    };

    try {
      if (window.turnstile && typeof window.turnstile.render === "function") {
        window.turnstile.render(slot, {
          sitekey: siteKey,
          callback: onSuccess,
          "expired-callback": onExpire,
          theme: "dark",
        });
      } else {
        // Caso raro: script carregou mas objeto ainda não está pronto.
        slot.textContent = "";
      }
    } catch (_) {
      // não quebra a página
    }
  });
};

// --- Client error reporting (opcional) ---
const bindClientErrorReporting = (config) => {
  const enabled = Boolean(config?.features?.client_error_reporting);
  if (!enabled) return;
  const sampleRate = Number(config?.client_error_sample_rate ?? 0.05);

  const shouldSend = () => {
    const r = Math.random();
    return r < (isNaN(sampleRate) ? 0.05 : sampleRate);
  };

  const send = (payload) => {
    if (!shouldSend()) return;
    fetch(`${BACKEND}/client_error`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...payload,
        page: window.location.href,
      }),
    }).catch(() => {});
  };

  window.addEventListener("error", (event) => {
    send({
      message: String(event?.message || "error"),
      stack: String(event?.error?.stack || ""),
    });
  });

  window.addEventListener("unhandledrejection", (event) => {
    const reason = event?.reason;
    send({
      message: String(reason?.message || reason || "unhandledrejection"),
      stack: String(reason?.stack || ""),
    });
  });
};

const applyActiveNav = (root, activeNav) => {
  if (!root || !activeNav) {
    return;
  }

  const activeLink = root.querySelector(`[data-nav="${activeNav}"]`);
  if (activeLink) {
    activeLink.setAttribute("aria-current", "page");
  }
};

const applyHeaderActions = (root) => {
  const actionsSlot = root?.querySelector("[data-header-actions]");
  const actionsTemplate = document.getElementById("header-actions-template");
  if (!actionsSlot || !actionsTemplate) {
    return;
  }

  actionsSlot.innerHTML = "";
  actionsSlot.append(actionsTemplate.content.cloneNode(true));
};

const ICONS = {
  moon: `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a7.5 7.5 0 1 0 9 9 9 9 0 1 1-9-9z"/></svg>`,
  sun: `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="M4.93 4.93l1.41 1.41"/><path d="M17.66 17.66l1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="M6.34 17.66l-1.41 1.41"/><path d="M19.07 4.93l-1.41 1.41"/></svg>`
};

const THEME_KEY = "leadrank_theme";

// --- Attribution (UTM / click ids) ---
// Mantém rastreamento de origem sem depender de libs externas. Usado para:
// 1) anexar UTMs em links de checkout (Kiwify)
// 2) enriquecer eventos de tracking (gtag/fbq/datalayer) quando existirem
const ATTR_LAST_KEY = "leadrank_attr_last";
const ATTR_FIRST_KEY = "leadrank_attr_first";
const ATTR_FIELDS = [
  "utm_source",
  "utm_medium",
  "utm_campaign",
  "utm_term",
  "utm_content",
  "gclid",
  "fbclid",
  "ttclid",
  "msclkid",
];

const safeLocalStorageGet = (key) => {
  try {
    return window.localStorage.getItem(key);
  } catch (_) {
    return null;
  }
};

const safeLocalStorageSet = (key, value) => {
  try {
    window.localStorage.setItem(key, value);
    return true;
  } catch (_) {
    return false;
  }
};

const readJson = (raw) => {
  try {
    return raw ? JSON.parse(raw) : null;
  } catch (_) {
    return null;
  }
};

const captureAttributionFromUrl = () => {
  let params;
  try {
    params = new URLSearchParams(window.location.search || "");
  } catch (_) {
    return;
  }

  const incoming = {};
  let hasAny = false;
  ATTR_FIELDS.forEach((field) => {
    const value = (params.get(field) || "").trim();
    if (!value) return;
    incoming[field] = value;
    hasAny = true;
  });

  // Também guarda contexto (página/referrer) para facilitar auditoria.
  if (hasAny) {
    incoming.landing_path = `${window.location.pathname}${window.location.search}${window.location.hash || ""}`;
    incoming.referrer = document.referrer || "";
    incoming.captured_at = new Date().toISOString();
  }

  if (!hasAny) return;

  const lastStored = readJson(safeLocalStorageGet(ATTR_LAST_KEY)) || {};
  const nextLast = { ...lastStored, ...incoming };
  safeLocalStorageSet(ATTR_LAST_KEY, JSON.stringify(nextLast));

  // First-touch: só grava se ainda não existir.
  const firstStored = safeLocalStorageGet(ATTR_FIRST_KEY);
  if (!firstStored) {
    safeLocalStorageSet(ATTR_FIRST_KEY, JSON.stringify(nextLast));
  }
};

const getAttribution = () => {
  return readJson(safeLocalStorageGet(ATTR_LAST_KEY)) || {};
};

const decorateCheckoutUrl = (href) => {
  if (typeof href !== "string" || !href) return href;
  let url;
  try {
    url = new URL(href, window.location.href);
  } catch (_) {
    return href;
  }

  // Só anexamos UTMs em checkout Kiwify.
  if (!/pay\.kiwify\.com\.br/i.test(url.hostname)) return href;

  const attr = getAttribution();
  ATTR_FIELDS.forEach((field) => {
    const value = (attr[field] || "").toString().trim();
    if (!value) return;
    if (!url.searchParams.get(field)) {
      url.searchParams.set(field, value);
    }
  });
  return url.toString();
};

let lastTrackedCheckoutKey = "";

const trackCheckoutClick = ({ planId = "", planName = "", href = "" } = {}) => {
  const key = `${planId}::${planName}::${href}`;
  if (!key || key === lastTrackedCheckoutKey) return;

  const attr = getAttribution();
  const payload = {
    event: "begin_checkout",
    plan_id: planId,
    plan_name: planName,
    destination: href,
    ...ATTR_FIELDS.reduce((acc, field) => {
      if (attr[field]) acc[field] = attr[field];
      return acc;
    }, {}),
  };

  if (typeof window.gtag === "function") {
    window.gtag("event", "begin_checkout", payload);
  }
  if (typeof window.fbq === "function") {
    window.fbq("track", "InitiateCheckout", payload);
  }
  if (Array.isArray(window.dataLayer)) {
    window.dataLayer.push(payload);
  }

  lastTrackedCheckoutKey = key;
};

const bindCheckoutLinks = () => {
  // Delegação para cobrir cards gerados dinamicamente (ex.: pricing.html).
  document.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;

    const link = target.closest("a[href]");
    if (!link) return;
    const href = link.getAttribute("href") || "";
    if (!href) return;

    let absoluteHref = href;
    try {
      absoluteHref = new URL(href, window.location.href).toString();
    } catch (_) {}

    if (!/pay\.kiwify\.com\.br/i.test(absoluteHref)) return;

    const container = link.closest("[data-plan]");
    const planId = (container?.getAttribute("data-plan") || "").trim();
    const planName = (container?.querySelector(".pricing__microcopy")?.textContent || link.textContent || "")
      .trim();

    const decorated = decorateCheckoutUrl(absoluteHref);
    if (decorated && decorated !== absoluteHref) {
      // Atualiza o href antes de navegar.
      link.setAttribute("href", decorated);
    }

    trackCheckoutClick({ planId, planName, href: decorated || absoluteHref });
  });
};

// --- Lead Conversion (Obrigado page) ---
// Para garantir disparo de conversão em um único lugar, gravamos um payload "pending"
// no submit do formulário e processamos o evento apenas na página de Obrigado.
const LEAD_PENDING_KEY = "leadrank_lead_pending";
const LEAD_TRACKED_KEY = "leadrank_lead_tracked_key";

const setPendingLeadConversion = (data) => {
  if (!data) return;
  safeLocalStorageSet(LEAD_PENDING_KEY, JSON.stringify(data));
};

const readPendingLeadConversion = () => {
  return readJson(safeLocalStorageGet(LEAD_PENDING_KEY)) || null;
};

const clearPendingLeadConversion = () => {
  try {
    window.localStorage.removeItem(LEAD_PENDING_KEY);
  } catch (_) {}
};

const trackLeadConversion = ({ email = "", clientId = "", source = "landing_form" } = {}) => {
  const key = `${String(email).trim().toLowerCase()}::${String(clientId).trim()}`;
  if (!key || key === "::") return;

  const lastKey = (safeLocalStorageGet(LEAD_TRACKED_KEY) || "").trim();
  if (lastKey && lastKey === key) return;

  const attr = getAttribution();
  const payload = {
    event: "Lead",
    // LGPD/políticas de mídia: não enviar PII (email) em texto puro para tags.
    client_id: String(clientId).trim(),
    source,
    ...ATTR_FIELDS.reduce((acc, field) => {
      if (attr[field]) acc[field] = attr[field];
      return acc;
    }, {}),
  };

  if (typeof window.gtag === "function") {
    window.gtag("event", "Lead", payload);
  }
  if (typeof window.fbq === "function") {
    window.fbq("track", "Lead", payload);
  }
  if (Array.isArray(window.dataLayer)) {
    window.dataLayer.push(payload);
  }

  safeLocalStorageSet(LEAD_TRACKED_KEY, key);
};

const processPendingLeadConversion = () => {
  const isThankYou =
    document.body?.dataset?.conversionPage === "lead" ||
    /\/obrigado\.html$/i.test(window.location.pathname || "");
  if (!isThankYou) return;

  const pending = readPendingLeadConversion();
  if (!pending) return;

  trackLeadConversion({
    email: pending.email || "",
    clientId: pending.client_id || pending.clientId || "",
    source: pending.source || "landing_form",
  });
  clearPendingLeadConversion();
};

// Exposição mínima para uso por lead-form.js / obrigado.js (sem dependências externas)
window.LeadrankTracking = window.LeadrankTracking || {};
window.LeadrankTracking.setPendingLeadConversion = setPendingLeadConversion;
window.LeadrankTracking.trackLeadConversion = trackLeadConversion;
window.LeadrankTracking.getAttribution = getAttribution;
window.LeadrankTracking.decorateCheckoutUrl = decorateCheckoutUrl;

const resolveInitialTheme = () => {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored === "light" || stored === "dark") return stored;
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  return prefersDark ? "dark" : "light";
};

const applyTheme = (theme) => {
  document.body.dataset.theme = theme;
  localStorage.setItem(THEME_KEY, theme);
};


const scrollToContatoIfNeeded = () => {
  const params = new URLSearchParams(window.location.search || "");
  const intent = params.get("intent");

  // Mantém compatibilidade com links antigos (#contato), mas quando houver intenção
  // de migração/enterprise direciona para o formulário (próximo passo claro).
  const targetId = intent === "enterprise"
    ? "lead-capture"
    : window.location.hash === "#contato"
      ? "contato"
      : "";

  if (!targetId) return;

  const target = document.getElementById(targetId);
  if (!target) return;

  // Aguarda layout estabilizar e o header sticky ser renderizado
  setTimeout(() => {
    target.scrollIntoView({ behavior: "smooth", block: "start" });

    if (targetId === "lead-capture") {
      const leadForm = document.getElementById("leadForm");
      const firstField = leadForm ? leadForm.querySelector("input, textarea, select") : null;
      firstField?.focus({ preventScroll: true });
      return;
    }

    // Para outros anchors, tenta focar o próprio alvo para ajudar navegação via teclado.
    if (!target.hasAttribute("tabindex")) {
      target.setAttribute("tabindex", "-1");
    }
    target.focus?.({ preventScroll: true });
  }, 60);
};

const bindIntentLinks = () => {
  document.querySelectorAll("[data-intent]").forEach((link) => {
    link.addEventListener("click", () => {
      const intent = link.getAttribute("data-intent");
      if (!intent) return;
      try {
        const url = new URL(window.location.href);
        url.searchParams.set("intent", intent);
        window.history.replaceState({}, "", url.toString());
      } catch (_) {}
    });
  });
};

const applyThemeToggle = (root) => {
  const actionsSlot = root?.querySelector("[data-header-actions]");
  if (!actionsSlot) return;

  const button = document.createElement("button");
  button.type = "button";
  button.className = "btn btn--ghost btn--small theme-toggle icon-button";

  const renderLabel = (theme) => {
    if (theme === "dark") {
      button.innerHTML = `${ICONS.sun} Claro`;
    } else {
      button.innerHTML = `${ICONS.moon} Escuro`;
    }
  };

  renderLabel(document.body.dataset.theme || "dark");

  button.addEventListener("click", () => {
    const nextTheme = document.body.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
    renderLabel(nextTheme);
  });

  actionsSlot.append(button);
};

applyTheme(resolveInitialTheme());

document.addEventListener("DOMContentLoaded", async () => {
  const activeNav = document.body.dataset.activeNav;

  // Captura UTMs/click-ids e prepara tracking de checkout sem interferir no restante.
  captureAttributionFromUrl();
  processPendingLeadConversion();
  bindCheckoutLinks();
  processPendingLeadConversion();

  // Flags públicas (demo, captcha...) - best effort.
  const publicConfig = await loadPublicConfig();

  try {
    const header = await loadPartial("[data-include='site-header']", "partials/site-header.html");
    applyActiveNav(header, activeNav);
    applyHeaderActions(header);
    applyThemeToggle(header);

    if (publicConfig) {
      applyFeatureFlags(publicConfig);
      bindClientErrorReporting(publicConfig);
      initTurnstile(publicConfig);
    }

    document.dispatchEvent(new CustomEvent("site-shell:header-ready"));
    bindIntentLinks();
    scrollToContatoIfNeeded();

    await loadPartial("[data-include='site-footer']", "partials/site-footer.html");
  } catch (error) {
    console.warn("Falha ao carregar o template base.", error);
  }
});
