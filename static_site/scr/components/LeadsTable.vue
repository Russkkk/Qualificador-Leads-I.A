<template>
  <div class="action-list" :data-state="state">
    <div class="action-list__header">
      <h2 class="icon-button">
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.8 19.8 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.08 4.18 2 2 0 0 1 4.06 2h3a2 2 0 0 1 2 1.72c.12.82.31 1.62.57 2.4a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.68-1.09a2 2 0 0 1 2.11-.45c.78.26 1.58.45 2.4.57A2 2 0 0 1 22 16.92z"/>
        </svg>
        Leads para agir agora
      </h2>
      <div class="action-list__actions">
        <button class="btn btn--ghost btn--small" type="button" @click="$emit('refresh')">Recarregar</button>
        <button class="btn btn--ghost btn--small" type="button" @click="$emit('seed')">Enviar leads de teste</button>
      </div>
    </div>

    <div v-if="state === 'loading'" class="card card--soft">
      <div class="skeleton-table">
        <div v-for="n in 4" :key="`lead-skeleton-${n}`" class="skeleton-row">
          <div class="skeleton-avatar"></div>
          <div class="skeleton-meta">
            <div class="skeleton-line skeleton-line--lg"></div>
            <div class="skeleton-line skeleton-line--md"></div>
            <div class="skeleton-line skeleton-line--xs"></div>
          </div>
          <div class="skeleton-badge"></div>
        </div>
      </div>
    </div>
    <div v-else-if="state === 'empty'" class="card card--soft">Nenhum lead disponível no momento.</div>
    <div v-else-if="state === 'error'" class="card card--soft">Falha ao carregar. Tente novamente.</div>

    <div v-else class="action-list__rows">
      <div v-for="lead in leads" :key="lead.id" class="lead-row">
        <div>
          <p class="lead-row__name">{{ lead.nome || "Lead" }}</p>
          <p class="text-muted">{{ lead.origem || "—" }} • {{ lead.telefone || "—" }}</p>
        </div>
        <div class="lead-row__meta">
          <span class="pill pill--muted">
            <svg v-if="temperatureType(lead) === 'hot'" class="icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M8.5 14.5A2.5 2.5 0 0 0 11 17a2 2 0 0 0 2-2c0-1.53-1.5-2.6-1.5-4A2.5 2.5 0 0 1 14 8.5c0-3-2.5-5.5-6-6 1 1.5 1 4.5-1 6-1.5 1.1-3 2.6-3 5a6 6 0 0 0 6 6 6 6 0 0 0 6-6c0-1.8-.4-2.8-1.6-4.4 0 1.5-1 2.5-2.4 3.4-1.2.8-2.5 1.8-2.5 3.5"/>
            </svg>
            <svg v-else-if="temperatureType(lead) === 'warm'" class="icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 2a2 2 0 0 0-2 2v10a4 4 0 1 0 4 0V4a2 2 0 0 0-2-2z"/>
            </svg>
            <svg v-else class="icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 2v20"/>
              <path d="M5.5 5.5l13 13"/>
              <path d="M5.5 18.5l13-13"/>
            </svg>
            {{ temperatureLabel(temperatureType(lead)) }}
          </span>
          <button class="btn btn--small" type="button" @click="$emit('open', lead)">Ver detalhes</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  leads: { type: Array, default: () => [] },
  state: { type: String, default: "loading" }
});

const temperatureType = (lead) => {
  const prob = Number(lead.probabilidade ?? lead.prob ?? 0);
  const score = Number(lead.score ?? 0);
  if (prob >= 0.7 || score >= 70) return "hot";
  if (prob >= 0.35 || score >= 35) return "warm";
  return "cold";
};

const temperatureLabel = (type) => {
  if (type === "hot") return "Quente";
  if (type === "warm") return "Morno";
  return "Frio";
};
</script>
