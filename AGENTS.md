# AGENTS.md

## Diretrizes permanentes para o Codex neste repositório

1. **Princípio de mudanças mínimas (sem refactors):**
   - Fazer apenas alterações estritamente necessárias para cumprir o objetivo.
   - Não refatorar, não reorganizar estrutura e não renomear arquivos/rotas sem necessidade explícita.
   - Não tocar em partes já funcionando.

2. **Branch + commits pequenos:**
   - Trabalhar sempre em branch dedicada para a tarefa.
   - Manter commits pequenos, incrementais e descritivos.
   - Garantir `git status` limpo antes de iniciar e ao finalizar.

3. **Sempre rodar build/test/lint existentes:**
   - Detectar o stack e usar apenas comandos já existentes no projeto.
   - Rodar verificações relevantes (build, testes e lint) antes de concluir.
   - Não inventar tooling novo.

4. **Preferência por diffs pequenos:**
   - Aplicar alterações localizadas e de baixo risco.
   - Priorizar patch/diff minimalista para facilitar revisão.

5. **Não adicionar dependências sem necessidade:**
   - Evitar novas dependências por padrão.
   - Se for inevitável, justificar claramente e manter escopo mínimo.

6. **Estilo/linters: seguir padrão do projeto:**
   - Respeitar convenções de código, nomenclatura e organização já existentes.
   - Seguir regras de lint/format do próprio repositório.

7. **Checklist de entrega (obrigatório):**
   - Listar arquivos alterados.
   - Resumir diff/mudanças realizadas.
   - Informar comandos executados e resultado.
   - Descrever como testar manualmente.
