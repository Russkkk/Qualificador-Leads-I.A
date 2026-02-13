<template>
  <div class="auth onboarding">
    <section>
      <p class="eyebrow">LeadRank onboarding</p>
      <h1>Configure sua conta em 4 passos</h1>
      <p class="subtitle">Crie sua conta, acesse sua API Key e integre o LeadRank.</p>
      <div class="stepper">
        <div
          v-for="step in steps"
          :key="step.id"
          class="stepper__item"
          :class="{ 'is-active': step.id === currentStep }"
        >
          <span class="stepper__dot">{{ step.id }}</span>
          <div>
            <strong>{{ step.title }}</strong>
            <p class="text-muted">{{ step.subtitle }}</p>
          </div>
        </div>
      </div>
    </section>

    <section class="card">
      <div v-if="currentStep === 1" class="step-panel is-active">
        <h2>Passo 1: Cadastro</h2>
        <p class="subtitle">Teste grátis por 14 dias. Sem cartão.</p>
        <form class="form-grid" @submit.prevent="submitSignup">
          <label>
            Nome completo
            <input v-model="signup.nome" type="text" required placeholder="João Silva" />
          </label>
          <label>
            Email profissional
            <input v-model="signup.email" type="email" required placeholder="joao@empresa.com" />
          </label>
          <label>
            Empresa / Marca (opcional)
            <input v-model="signup.empresa" type="text" placeholder="Minha Empresa" />
          </label>
          <label>
            WhatsApp (opcional)
            <input v-model="signup.telefone" type="tel" placeholder="(11) 98765-4321" />
          </label>
          <label>
            Senha
            <input v-model="signup.password" type="password" required minlength="10" placeholder="Mínimo 10 caracteres" />
          </label>
          <button class="btn btn--block" type="submit" :disabled="loading">Criar conta</button>
          <p v-if="error" class="alert alert--error">{{ error }}</p>
        </form>
      </div>

      <div v-if="currentStep === 2" class="step-panel is-active">
        <h2>Passo 2: Sua API Key</h2>
        <p class="subtitle">Guarde sua API Key para integrar o LeadRank.</p>
        <div class="api-key">
          <span>{{ apiKey || "—" }}</span>
          <button class="btn btn--ghost btn--small" type="button" @click="copyApiKey">Copiar</button>
        </div>
        <div class="step-actions">
          <button class="btn" type="button" @click="currentStep = 3">Continuar</button>
        </div>
      </div>

      <div v-if="currentStep === 3" class="step-panel is-active">
        <h2>Passo 3: Como integrar</h2>
        <p class="subtitle">Use sua API Key e conecte seus canais.</p>
        <div class="integration-list">
          <div class="integration-item">
            <strong>Endpoint /prever</strong>
            <code>POST {{ backend }}/prever</code>
          </div>
          <div class="integration-item">
            <strong>Headers</strong>
            <code>X-API-KEY: {{ apiKey || "sua-chave" }}</code>
            <code>X-CLIENT-ID: {{ clientId || "seu-workspace" }}</code>
          </div>
        </div>
        <div class="step-actions">
          <button class="btn" type="button" @click="currentStep = 4">Ir para o passo final</button>
        </div>
      </div>

      <div v-if="currentStep === 4" class="step-panel is-active">
        <h2>Pronto! Vá para o dashboard</h2>
        <p class="subtitle">Sua conta está configurada e a Ação do Dia está pronta para você.</p>
        <a class="btn btn--block" href="acao.html">Ir para Ação do Dia</a>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { backend } from "../shared/api.js";

const steps = [
  { id: 1, title: "Cadastro", subtitle: "Crie sua conta trial" },
  { id: 2, title: "Sua API Key", subtitle: "Guarde com segurança" },
  { id: 3, title: "Como integrar", subtitle: "Escolha o melhor caminho" },
  { id: 4, title: "Ir para o dashboard", subtitle: "Ação do dia pronta" }
];

const currentStep = ref(1);
const loading = ref(false);
const error = ref("");
const apiKey = ref("");
const clientId = ref("");

const signup = ref({
  nome: "",
  email: "",
  empresa: "",
  telefone: "",
  password: ""
});

const submitSignup = async () => {
  error.value = "";
  loading.value = true;
  try {
    const response = await fetch(`${backend}/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(signup.value)
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || data.message || "Erro ao criar conta.");
    }
    apiKey.value = response.headers.get("X-API-KEY") || sessionStorage.getItem("api_key") || localStorage.getItem("api_key") || "";
    clientId.value = data.client_id || "";
    if (clientId.value) {
      localStorage.setItem("client_id", clientId.value);
      localStorage.setItem("leadrank_client_id", clientId.value);
      localStorage.setItem("LR_CLIENT_ID", clientId.value);
    }
    if (apiKey.value) {
      // API key é sensível: por padrão, salva apenas na sessão.
      sessionStorage.setItem("api_key", apiKey.value);
      sessionStorage.setItem("leadrank_api_key", apiKey.value);
      sessionStorage.setItem("LR_API_KEY", apiKey.value);
    }
    currentStep.value = 2;
  } catch (err) {
    error.value = err.message || "Erro ao criar conta.";
  } finally {
    loading.value = false;
  }
};

const copyApiKey = async () => {
  if (!apiKey.value) return;
  try {
    await navigator.clipboard.writeText(apiKey.value);
  } catch (err) {
    // ignore
  }
};
</script>
