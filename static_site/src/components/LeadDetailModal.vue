<template>
  <div class="modal" :class="{ 'modal--open': open }">
    <div class="modal__backdrop" @click="$emit('close')"></div>
    <div class="modal__content">
      <div class="modal__header">
        <div>
          <h3>Detalhes do lead</h3>
          <p class="text-muted">Informações principais para contato.</p>
        </div>
        <button class="btn btn--ghost btn--small" type="button" @click="$emit('close')">Fechar</button>
      </div>
      <div class="modal__grid">
        <div>
          <p class="text-muted">Nome</p>
          <p class="modal__value">{{ lead?.nome || "—" }}</p>
        </div>
        <div>
          <p class="text-muted">Email</p>
          <p class="modal__value">{{ lead?.email_lead || lead?.email || "—" }}</p>
        </div>
        <div>
          <p class="text-muted">Telefone</p>
          <p class="modal__value">{{ lead?.telefone || "—" }}</p>
        </div>
        <div>
          <p class="text-muted">Origem</p>
          <p class="modal__value">{{ lead?.origem || "—" }}</p>
        </div>
        <div class="modal__score">
          <p class="text-muted">Score / Probabilidade</p>
          <p class="modal__value">{{ lead?.score ?? "—" }} • {{ formatProb(lead?.probabilidade) }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  open: { type: Boolean, default: false },
  lead: { type: Object, default: null }
});

const formatProb = (value) => {
  if (value == null) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
};
</script>
