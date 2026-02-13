(() => {
  const get = (key) => {
    try {
      // Preferência: sessionStorage (menos exposição) e fallback para localStorage (compat.).
      return window.sessionStorage.getItem(key) || window.localStorage.getItem(key) || "";
    } catch (_) {
      return "";
    }
  };

  const clearSession = (key) => {
    try {
      window.sessionStorage.removeItem(key);
    } catch (_) {}
  };

  const getFirstAvailable = (keys) => {
    for (const key of keys) {
      const value = get(key).trim();
      if (value) return value;
    }
    return "";
  };

  const setText = (id, value) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = value;
  };

const setListItems = (listEl, items) => {
  if (!listEl) return;
  listEl.innerHTML = "";
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    listEl.appendChild(li);
  }
};



  const maskSecret = (value, visibleTail = 4) => {
    if (!value) return "";
    const tail = value.slice(-visibleTail);
    const maskLen = Math.max(8, value.length - visibleTail)
    return `${"•".repeat(maskLen)}${tail}`;
  };

  const copyToClipboard = async (value) => {
    if (!value) return false;
    // Modern Clipboard API
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        await navigator.clipboard.writeText(value);
        return true;
      }
    } catch (_) {
      // fall through
    }

    // Fallback for older browsers
    try {
      const textarea = document.createElement("textarea");
      textarea.value = value;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "absolute";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(textarea);
      return Boolean(ok);
    } catch (_) {
      return false;
    }
  };

  document.addEventListener("DOMContentLoaded", () => {
    const email = get("leadrank_last_lead_email").trim();
    const tempPassword = get("leadrank_temp_password").trim();

    const clientId = getFirstAvailable(["leadrank_client_id", "client_id", "LR_CLIENT_ID"]).trim();
    const apiKey = getFirstAvailable(["leadrank_api_key", "api_key", "LR_API_KEY"]).trim();
    const hasAuth = Boolean(clientId && apiKey);


    const integrationSteps = document.getElementById("thankYouIntegrationSteps");
    const integrationsCta = document.getElementById("thankYouIntegrationsCta");


    // Micro-bloco de API Key: mostra apenas quando existir e permite copiar com segurança.
    const apiKeyBlock = document.getElementById("thankYouApiKeyBlock");
    const apiKeyValue = document.getElementById("thankYouApiKeyValue");
    const apiKeyCopyBtn = document.getElementById("thankYouCopyApiKey");
    const apiKeyToggleBtn = document.getElementById("thankYouToggleApiKey");
    const apiKeyStatus = document.getElementById("thankYouApiKeyStatus");

    let apiKeyRevealed = false;
    const renderApiKey = () => {
      if (!apiKeyValue) return;
      apiKeyValue.textContent = apiKeyRevealed ? apiKey : maskSecret(apiKey);
      if (apiKeyToggleBtn) {
        apiKeyToggleBtn.textContent = apiKeyRevealed ? "Ocultar" : "Mostrar";
        apiKeyToggleBtn.setAttribute(
          "aria-label",
          apiKeyRevealed ? "Ocultar API Key" : "Mostrar API Key"
        );
      }
    };

    if (apiKey && apiKeyBlock && apiKeyValue) {
      apiKeyBlock.classList.remove("hidden");
      renderApiKey();

      if (apiKeyToggleBtn) {
        apiKeyToggleBtn.addEventListener("click", () => {
          apiKeyRevealed = !apiKeyRevealed;
          if (apiKeyStatus) apiKeyStatus.textContent = "";
          renderApiKey();
        });
      }

      if (apiKeyCopyBtn) {
        apiKeyCopyBtn.addEventListener("click", async () => {
          const previous = apiKeyCopyBtn.textContent;
          apiKeyCopyBtn.textContent = "Copiando...";
          const ok = await copyToClipboard(apiKey);
          apiKeyCopyBtn.textContent = ok ? "Copiada!" : "Falhou";
          if (apiKeyStatus) {
            apiKeyStatus.textContent = ok
              ? "API Key copiada. Cole com cuidado em um local seguro."
              : "Não foi possível copiar automaticamente. Selecione a chave e copie manualmente.";
          }
          window.setTimeout(() => {
            apiKeyCopyBtn.textContent = previous;
          }, 1400);
        });
      }
    }

    const primaryCta = document.getElementById("thankYouPrimaryCta");
    const secondaryCta = document.getElementById("thankYouSecondaryCta");
    const hint = document.getElementById("thankYouNextStepHint");

    const hasDetails = Boolean(email || tempPassword);
    if (hasDetails) {
      const details = document.getElementById("thankYouDetails");
      if (details) {
        details.classList.remove("hidden");
      }
      setText("thankYouEmail", email || "(não informado)");
      setText("thankYouTempPassword", tempPassword || "(gerada após cadastro)");
      // Limpa a senha temporária da sessão após renderizar.
      if (tempPassword) clearSession("leadrank_temp_password");
    }

    // CTA inteligente: evita levar o usuário para um caminho que o deixe travado.
    // - Se já temos credenciais (client_id + api_key), vai direto para a Ação do Dia e/ou onboarding com step avançado.
    // - Se não temos credenciais, leva para login (pré-preenchido quando possível) para obter o X-API-KEY.
    if (hasAuth) {
      if (primaryCta) {
        primaryCta.textContent = "Ir para Ação do Dia";
        primaryCta.setAttribute("href", "acao.html");
        primaryCta.setAttribute("aria-label", "Ir para a Ação do Dia do LeadRank");
      }
      if (secondaryCta) {
        secondaryCta.textContent = "Ver API Key e integrações";
        secondaryCta.setAttribute("href", "onboarding.html?step=2");
        secondaryCta.setAttribute("aria-label", "Abrir onboarding na etapa de API Key e integrações");
      }
if (hint) {
  hint.textContent = "Conta criada e credenciais salvas. Próximo passo: acesse a Ação do Dia ou copie sua API Key para integrar.";
}

setListItems(integrationSteps, [
  "Copie sua API Key acima.",
  "Abra a etapa de integrações e conecte Zapier/Make/Webhooks em minutos.",
]);
if (integrationsCta) {
  integrationsCta.textContent = "Abrir integrações";
  integrationsCta.setAttribute("href", "onboarding.html?step=2");
  integrationsCta.setAttribute("aria-label", "Abrir integrações no onboarding do LeadRank");
}

return;
    }

    // Sem auth completa: direciona para login para garantir client_id + api_key.
    const next = encodeURIComponent("onboarding.html?step=2");
    const emailParam = email ? `&email=${encodeURIComponent(email)}` : "";
    // Segurança: nunca trafegar senha em URL.
    const loginHref = `login.html?next=${next}${emailParam}`;

    if (primaryCta) {
      primaryCta.textContent = "Entrar e continuar";
      primaryCta.setAttribute("href", loginHref);
      primaryCta.setAttribute("aria-label", "Fazer login e continuar o onboarding do LeadRank");
    }
    if (secondaryCta) {
      secondaryCta.textContent = "Continuar no onboarding";
      secondaryCta.setAttribute("href", "onboarding.html?mode=login");
      secondaryCta.setAttribute("aria-label", "Abrir onboarding no modo login");
    }
    if (hint) {
      hint.textContent = "Próximo passo: faça login para confirmar suas credenciais e copiar sua API Key.";
    }

    // (Ainda pode estar visível no DOM; a remoção evita que continue disponível por storage.)

setListItems(integrationSteps, [
  "Faça login para obter sua API Key.",
  "Abra a etapa de integrações e conecte Zapier/Make/Webhooks em minutos.",
]);
if (integrationsCta) {
  integrationsCta.textContent = "Fazer login e abrir integrações";
  integrationsCta.setAttribute("href", loginHref);
  integrationsCta.setAttribute("aria-label", "Fazer login e abrir integrações do LeadRank");
}

  });
})();
