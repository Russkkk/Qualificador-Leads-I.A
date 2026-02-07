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
  const shouldScroll = window.location.hash === "#contato" || params.get("intent") === "enterprise";
  if (!shouldScroll) return;

  const anchor = document.getElementById("contato");
  if (!anchor) return;

  // Aguarda layout estabilizar e o header sticky ser renderizado
  setTimeout(() => {
    anchor.scrollIntoView({ behavior: "smooth", block: "start" });
    const leadForm = document.getElementById("leadForm");
    const firstField = leadForm ? leadForm.querySelector("input, textarea, select") : null;
    if (firstField) {
      firstField.focus({ preventScroll: true });
    } else {
      anchor.focus?.({ preventScroll: true });
    }
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

  try {
    const header = await loadPartial("[data-include='site-header']", "partials/site-header.html");
    applyActiveNav(header, activeNav);
    applyHeaderActions(header);
    applyThemeToggle(header);
    document.dispatchEvent(new CustomEvent("site-shell:header-ready"));
    bindIntentLinks();
    scrollToContatoIfNeeded();

    await loadPartial("[data-include='site-footer']", "partials/site-footer.html");
  } catch (error) {
    console.warn("Falha ao carregar o template base.", error);
  }
});
