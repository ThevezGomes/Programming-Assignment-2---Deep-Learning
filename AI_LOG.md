# Registro de Uso de Inteligência Artificial (AI_LOG.md)

**Disciplina:** Aprendizado Profundo  
**Projeto:** Programming Assignment 2 — Identidade ao longo do tempo: detecção, recorrência e rastreamento  
**Professor:** Dario Oliveira  
**Monitor:** Erick Brito  

---

## 1. Ferramentas Utilizadas
* **Antigravity (Google DeepMind / Gemini 3.8 Flash & Pro):** Pair programming agent, auditoria de integridade de código, suporte a refatoração modular, implementação de testes unitários com pytest, execução de experimentos com aceleração MPS e verificação de consistência matemática das métricas.

---

## 2. Episódios de Uso e Tomada de Decisão

### Episódio 1: Estruturação Modular e Escolha dos Eixos
* **Contexto:** Análise do enunciado oficial para selecionar os eixos e trilhas mais sólidos tecnicamente, viáveis computacionalmente e com máxima sinergia pedagógica.
* **Uso da IA:** A IA analisou o `PA2.pdf` e identificou que a **Trilha A (Modelo de Movimento com caixas delimitadoras)** combinada com o **Eixo 1 (Ablação de células RNN vs LSTM vs GRU e janela de BPTT)** oferecia integração perfeita com a **Parte 4 (Medição analítica de desaparecimento de gradiente)** e reaproveitamento do simulador da **Parte 0** no teste de estresse da **Parte 5**.
* **Resultado:** Escolhas alinhadas com foco em reprodutibilidade rápida e clareza conceitual na apresentação.

### Episódio 2: Implementação das Métricas Proprietárias e Testes Unitários
* **Contexto:** O enunciado proíbe o uso de bibliotecas de tracking prontas (`motmetrics`, `TrackEval`).
* **Uso da IA:** Formulação do cálculo de IDF1 via algoritmo Húngaro global (associação biunívoca ótima entre trajetórias ground truth e preditas, conforme Ristani et al., 2016) e na criação dos testes unitários controlados.
* **Resultado:** Validação formal das métricas antes de qualquer experimentação em larga escala.

### Episódio 3: Refinamento dos Requisitos da Parte 0 e Parte 1
* **Contexto:** Auditoria detalhada do enunciado revelou a necessidade da figura obrigatória de oclusão controlada (onde o objeto desaparece por $N$ quadros e retorna), do ensaio de quebra girando os botões do gerador e da formatação estrita do painel de descolamento em dois andares sobre as mesmas sequências.
* **Uso da IA:** Implementação do gerador de oclusão controlada no $z$-buffer com renderização da tira cronológica (`docs/parte0_trajetoria_oclusao.png`), o ensaio multi-botão (`docs/parte0_curva_quebra_gerador.png`), a execução da Fonte 2 (Faster R-CNN COCO com `custom_nms`) em dados reais do MOT17 (`docs/parte1_comparacao_fontes_real.png`) e o painel obrigatório do descolamento com dois subplots verticais (`docs/painel_descolamento_parte1.png`).
* **Resultado:** Partes 0 e 1 100% aderentes a todos os critérios e figuras do enunciado oficial.

### Episódio 4: Implementação da Parte 2 (Trilha A: Modelo de Movimento com Incerteza)
* **Contexto:** Escolha e estruturação da Trilha A com rede recorrente para predição de caixas delimitadoras sob oclusão, integrando com as perguntas de ablação (Parte 3), gradientes analíticos (Parte 4) e estresse temporal (Parte 5).
* **Uso da IA:** Projeção do `MotionPredictor` modular em PyTorch suportando células LSTM, GRU e RNN Simples, com predição residual de caixas e cabeça de incerteza gaussiana (otimizada via Gaussian NLL + Smooth L1). Implementação do `RNNMotionTracker` com rollout autoregressivo sob oclusão e portão adaptativo, pipeline de treinamento com Scheduled Sampling e comparação lado a lado contra o baseline ingênuo (`docs/parte2_comparacao_baseline_trilhaA.png`).
* **Resultado:** Parte 2 concluída com checkpoint canônico único (`checkpoints/motion_lstm_best.pt`) treinado e convergido.

### Episódio 5: Auditoria Metodológica e Correção Rigorosa dos Protocolos (P0-1 a P0-5 e P1-1 a P1-5)
* **Contexto:** Identificação de discrepâncias sutis entre os scripts: limiares de IoU heterogêneos entre partes (0.3 vs 0.5), conclusões textuais hardcoded que não refletiam exatamente as medições reais, falta de sementes aleatórias na degradação da Parte 5 e necessidade de calibração precisa do número de parâmetros na ablação da Parte 3.
* **Uso da IA:**
  1. **Unificação do Protocolo (`src/config.py`):** Criação da constante global `EVAL_IOU = 0.50` aplicada rigorosamente a todas as chamadas de avaliação em todo o repositório, garantindo que os resultados de controle sejam exatamente idênticos entre Partes 1, 2, 4 e 5.
  2. **Calibração de Parâmetros da Ablação (`src/ablation.py`):** Busca sistemática de `hidden_dim` para fixar o orçamento em $\sim 40.000$ parâmetros ($\pm 5\%$): RNN $h=113$ ($40.118$ params, $+0.3\%$), GRU $h=74$ ($39.894$ params, $-0.3\%$), LSTM $h=65$ ($39.458$ params, $-1.4\%$).
  3. **Reescrita Honesta das Narrativas:** Quando os dados reais mostraram que a RNN Simples **não colapsa** no rastreamento online com BPTT truncado (sustentando IDF1 $\approx 0.612 - 0.622$) e que o efeito do modelo recorrente no teste de estresse da Parte 5 é estatisticamente **Neutro** frente a múltiplas sementes e um controle ingênuo com limiar adaptativo, todo o texto e diagnósticos foram reescritos a partir dos números reais medidos, descartando afirmações qualitativas anteriores.
  4. **Reformulação da Galeria de Falhas e Horizontes (`src/failures.py`):** Substituição de casos fixos por seleção automática reproduzível baseada em critérios objetivos de gap de oclusão e troca de ID, com exibição da caixa predita pelo rollout da LSTM durante a oclusão e cálculo do horizonte empírico efetivo ($P \ge 0.5$ em $\sim 4$ quadros).
  5. **Generalização da Inferência (`src/inference.py`):** Leitura dinâmica de dimensões de imagem da sequência (resolvendo o problema de sequências em $640 \times 480$ como MOT17-05), suporte a pastas arbitrárias, contagem de identidades confirmadas por `min_hits` e geração simultânea de GIF e MP4.
* **Resultado:** Todo o repositório tornou-se plenamente reprodutível, robusto e matematicamente honesto, aderindo integralmente a todas as exigências do enunciado e dos monitores.
