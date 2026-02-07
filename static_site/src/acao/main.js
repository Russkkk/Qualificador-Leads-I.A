import { createApp, ref } from "vue";
import LeadsTable from "../components/LeadsTable.vue";
import LeadDetailModal from "../components/LeadDetailModal.vue";
import { exportCsv, getActionList, seedTestLeads } from "../shared/api.js";

const app = {
  components: { LeadsTable, LeadDetailModal },
  setup() {
    const leads = ref([]);
    const state = ref("loading");
    const selectedLead = ref(null);
    const modalOpen = ref(false);
    const message = ref("");

    const loadLeads = async () => {
      state.value = "loading";
      try {
        leads.value = await getActionList();
        state.value = leads.value.length ? "ready" : "empty";
      } catch (err) {
        state.value = "error";
        message.value = err.message || "Falha ao carregar leads.";
      }
    };

    const openLead = (lead) => {
      selectedLead.value = lead;
      modalOpen.value = true;
    };

    const closeLead = () => {
      modalOpen.value = false;
      selectedLead.value = null;
    };

    const seedLeads = async () => {
      try {
        await seedTestLeads(10);
        await loadLeads();
      } catch (err) {
        message.value = err.message || "Erro ao criar leads de teste.";
      }
    };

    const exportLeads = async () => {
      try {
        const { blob, filename } = await exportCsv();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(url);
      } catch (err) {
        message.value = err.message || "Erro ao exportar.";
      }
    };

    loadLeads();

    return {
      leads,
      state,
      selectedLead,
      modalOpen,
      message,
      loadLeads,
      openLead,
      closeLead,
      seedLeads,
      exportLeads
    };
  },
  template: `
    <section class="dashboard">
      <div v-if="message" class="alert alert--error">{{ message }}</div>
      <LeadsTable :leads="leads" :state="state" @open="openLead" @refresh="loadLeads" @seed="seedLeads" />
      <div class="action-list__footer">
        <button class="btn btn--ghost" type="button" @click="exportLeads">Exportar CSV</button>
      </div>
      <LeadDetailModal :open="modalOpen" :lead="selectedLead" @close="closeLead" />
    </section>
  `
};

createApp(app).mount("#acao-app");
