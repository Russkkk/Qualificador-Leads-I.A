# Plano enxuto de GO-LIVE — LeadRank

## 1) CHECKLIST DE GO-LIVE (TÉCNICO + PRODUTO)

**Performance mínima aceitável**
- [ ] Ação do Dia abre em até 3s com conexão padrão.
- [ ] Lista carrega em até 5s sem travar a página.
- [ ] Botões ✅/❌/⏳ respondem em até 2s.

**Tratamento de erros visíveis ao usuário**
- [ ] Erros de API mostram mensagem clara (“tente novamente”).
- [ ] Botão “Tentar novamente” funciona.
- [ ] Erro de autenticação orienta “Conectar workspace”.

**Estados vazios e mensagens**
- [ ] Estado sem leads exibe CTA “Gerar leads de teste”.
- [ ] Texto orienta “Ver como integrar”.
- [ ] Mensagens de sucesso/erro ficam visíveis.

**Limites de plano e mensagens de upgrade**
- [ ] Limite de leads/mês visível no painel.
- [ ] Avisos em 70% e 100% do limite.
- [ ] CTA “Upgrade para Pro” em momento de limite.

**Segurança básica**
- [ ] API key não aparece na interface.
- [ ] Inputs sensíveis não são exibidos em tela.
- [ ] Logs não incluem dados pessoais completos.

**Conteúdos obrigatórios**
- [ ] Landing com copy final.
- [ ] Vídeo demo publicado.
- [ ] Demo guiada ativa na primeira visita.
- [ ] Pricing com Essencial/Pro e trial 14 dias.

---

## 2) MÉTRICAS ESSENCIAIS (SEM COMPLEXIDADE)

### Eventos FRONTEND (mínimos)
| Evento | Quando disparar | Como registrar | Como analisar no início |
|---|---|---|---|
| signup_started | Clique em “Criar conta” | `console.log` + POST `/events` | Contar no log diário |
| signup_completed | Conta criada com sucesso | POST `/events` | Comparar com início |
| demo_started | Demo guiada aberta | `console.log` | Ver % que inicia |
| demo_completed | Final da demo | POST `/events` | % conclusão |
| action_day_viewed | Ação do Dia carregou | POST `/events` | Uso diário |
| lead_labeled_yes | Clique ✅ | POST `/events` | Engajamento |
| lead_labeled_no | Clique ❌ | POST `/events` | Engajamento |
| integration_clicked | Clique “Ver como integrar” | `console.log` | Interesse real |
| pricing_viewed | Abriu página planos | POST `/events` | Intenção |
| upgrade_clicked | Clique no CTA upgrade | POST `/events` | Pronto para pagar |

### Eventos BACKEND (logs simples)
| Evento | Quando registrar | Como registrar | Como analisar |
|---|---|---|---|
| trial_started | Primeiro login com workspace | Log servidor | Contar no dia |
| first_lead_received | 1º lead recebido | Log servidor | Ativação |
| first_label_done | 1º ✅/❌ registrado | Log servidor | Uso real |
| plan_limit_reached | Limite atingido | Log servidor | Upsell |
| upgrade_completed | Pagamento aprovado | Log servidor | Receita |

**Como registrar (simples)**
- Frontend: `fetch('/events', { method: 'POST', body: JSON.stringify({ event, ts, client_id }) })`.
- Backend: log em arquivo/console com `event_name | client_id | timestamp`.

**Como analisar manualmente no início**
- Exportar logs diários e contar eventos em planilha.
- Comparar funnels simples: `signup_started → signup_completed → action_day_viewed → demo_completed → first_label_done`.

---

## 3) ONBOARDING DE CONVERSÃO (AJUSTES FINAIS)

**Ajustes obrigatórios**
- Seed automático de leads no primeiro acesso.
- Demo guiada obrigatória antes de explorar.
- Pós-demo com CTA claro: “Continuar usando” ou “Integrar meus leads”.
- Mensagens in-app nos dias 1, 3 e 7 do trial.

**Textos prontos (in-app)**
- **Dia 1:** “Bem-vindo! Sua lista do dia já está pronta. Comece pelos primeiros e marque ✅/❌.”
- **Dia 3:** “Quanto mais você marca resultado, melhor a prioridade. Quer integrar seus leads reais?”
- **Dia 7:** “Seu trial está avançando. Quer manter a lista diária ativa com o Essencial ou Pro?”

---

## 4) SOFT LAUNCH — PLANO DE 7 DIAS

| Dia | Ação principal | Métrica para observar | Sinal de alerta |
|---|---|---|---|
| 1 | Publicar landing + vídeo demo | Visitas → cadastros | Poucos cadastros |
| 2 | Outreach manual (WhatsApp/LinkedIn) | Respostas e demos marcadas | Baixa resposta |
| 3 | Primeiras demos | Demo → trial | Trial baixo |
| 4 | Acompanhar ativação | % que vê Ação do Dia | Usuários perdidos |
| 5 | Ajustes rápidos de copy | CTA → cadastro | Clique sem cadastro |
| 6 | Primeiros upgrades | Trial → pago | Nenhum upgrade |
| 7 | Revisão + priorização | Retenção dia 7 | Queda forte de uso |

---

## 5) FEEDBACK LOOP (O QUE O FUNDADOR DEVE FAZER)

**Perguntas para usuários ativos**
- “O que fez você abrir o LeadRank hoje?”
- “Qual parte da lista do dia mais te ajuda?”
- “O que deixaria essa rotina mais rápida?”

**Perguntas para quem não ativou**
- “Onde você travou no primeiro acesso?”
- “O que faltou para começar a usar?”
- “A tela deixou claro quem ligar primeiro?”

**Perguntas para quem cancelou/ignorou**
- “Por que decidiu não seguir com o trial?”
- “O preço fez sentido?”
- “O que precisava ter acontecido para continuar?”

**Como transformar feedback em backlog**
- Juntar respostas em 3 colunas: **bloqueio**, **confusão**, **pedido**.
- Priorizar o que aparece 3+ vezes.
- Ajustar copy/UX antes de criar novas funções.

---

## 6) DEFINIÇÃO DE SUCESSO (30 DIAS)

**Metas realistas**
- 40–60% dos cadastros vendo Ação do Dia.
- 20–30% marcando ✅/❌ pelo menos 1 vez.
- 10–15% de upgrade entre quem ativou.

**Sinais de product-market fit inicial**
- Usuários voltam diariamente sem lembrete.
- Pedem aumento de limite de leads.
- Dão feedback de “lista pronta” como principal valor.

**Sinais de problema**
- **Mensagem ruim:** visitas altas e poucos cadastros.
- **Produto confuso:** usuários não entendem “Ação do Dia” em 1 frase.
- **Preço errado:** alta ativação e quase nenhum upgrade.
