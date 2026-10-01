# Registro de Uso de Inteligência Artificial (AI_LOG.md)

**Disciplina:** Aprendizado Profundo  
**Projeto:** Programming Assignment 2 — Identidade ao longo do tempo: detecção, recorrência e rastreamento  
**Professor:** Dario Oliveira  
**Monitor:** Erick Brito  

---

## 1. Ferramentas Utilizadas
* **Antigravity (Google DeepMind / Gemini 3.8 Flash & Pro):** Pair programming, estruturação da arquitetura de pastas, geração de código de suporte, depuração e elaboração dos testes unitários.

---

## 2. Episódios de Uso e Tomada de Decisão

### Episódio 1: Estruturação Modular e Escolha dos Eixos
* **Contexto:** Análise inicial do enunciado para selecionar os eixos e trilhas mais sólidos tecnicamente, viáveis computacionalmente e com máxima sinergia pedagógica.
* **Uso da IA:** A IA analisou o `PA2.pdf` e identificou que a **Trilha A (Modelo de Movimento com caixas delimitadoras)** combinada com o **Eixo 1 (Ablação de células RNN vs LSTM vs GRU e janela de BPTT)** oferecia integração perfeita com a **Parte 4 (Medição analítica de desaparecimento de gradiente)** e reaproveitamento do simulador da **Parte 0** no teste de estresse da **Parte 5**.
* **Resultado:** Escolhas alinhadas com foco em reprodutibilidade rápida e clareza conceitual na apresentação.

### Episódio 2: Implementação das Métricas Proprietárias e Testes Unitários
* **Contexto:** O enunciado proíbe o uso de bibliotecas de tracking prontas (`motmetrics`, `TrackEval`).
* **Uso da IA:** Auxílio na formulação do cálculo do IDF1 via algoritmo Húngaro global (associação biunívoca ótima entre trajetórias ground truth e preditas) e na criação dos 3 casos de teste sintéticos construídos à mão (caso identidade perfeita, troca de ID e fragmentação de track).
* **Resultado:** Validação formal das métricas antes de qualquer experimentação em larga escala.

### Episódio 3: Refinamento dos Requisitos da Parte 0 e Parte 1
* **Contexto:** Auditoria detalhada do enunciado revelou a necessidade da figura obrigatória de oclusão controlada (onde o objeto desaparece por $N$ quadros e retorna), do ensaio de quebra girando os botões do gerador e da formatação estrita do painel de descolamento em dois andares sobre as mesmas sequências.
* **Uso da IA:** A IA implementou o gerador de oclusão controlada no $z$-buffer com renderização da tira cronológica (`docs/parte0_trajetoria_oclusao.png`), o ensaio multi-botão (`docs/parte0_curva_quebra_gerador.png`), a execução da Fonte 2 (Faster R-CNN COCO com `custom_nms`) em dados reais do MOT17 (`docs/parte1_comparacao_fontes_real.png`) e o painel obrigatório do descolamento com dois subplots verticais (`docs/painel_descolamento_parte1.png`).
* **Resultado:** Partes 0 e 1 100% aderentes a todos os critérios e figuras do enunciado oficial.

