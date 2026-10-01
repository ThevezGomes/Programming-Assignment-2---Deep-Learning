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

### Episódio 4: Implementação da Parte 2 (Trilha A: Modelo de Movimento com Incerteza)
* **Contexto:** Escolha e estruturação da Trilha A com rede recorrente para predição de caixas delimitadoras sob oclusão, integrando com as perguntas de ablação (Parte 3), gradientes analíticos (Parte 4) e estresse temporal (Parte 5).
* **Uso da IA:** A IA projetou o `MotionPredictor` modular em PyTorch suportando células LSTM, GRU e RNN Simples, com predição residual de caixas e cabeça de incerteza gaussiana (otimizada via Gaussian NLL + Smooth L1). Implementou o `RNNMotionTracker` com rollout autoregressivo sob oclusão e portão adaptativo, o pipeline de treinamento com Scheduled Sampling e a comparação lado a lado contra o baseline ingênuo (`docs/parte2_comparacao_baseline_trilhaA.png`).
* **Resultado:** Parte 2 concluída com checkpoint treinado e convergência comprovada, deixando a base pronta para as ablações da Parte 3.

### Episódio 5: Partes 3 e 4 (Ablação de Células, Horizonte de Memória e Intervenção)
* **Contexto:** Necessidade de comprovar o vanishing gradient na RNN Simples para janelas $T \ge 16$, medir os horizontes analítico/empírico e aplicar uma intervenção corretiva em uma falha real.
* **Uso da IA:** A IA implementou a rotina de ablação em 3 seeds para o Eixo 1 (`results/ablation_eixo1.json`), gerou a medição analítica de $\|\partial L_T / \partial h_{T-k}\|$ comprovando a queda de $20\times$ em 8 passos na RNN simples, e projetou a intervenção corretiva com *velocity damping* ($0.85$) e *sigma inflation* ($0.30$) no `RNNMotionTracker`, recuperando o descolamento da trajetória na sequência `MOT17-09`.
* **Resultado:** Validação formal da hipótese teórica e correção demonstrada com tira comparativa (`docs/parte4_antes_depois_correcao.png`).

### Episódio 6: Parte 5 (Teste de Estresse por Degradação da Qualidade do Detector)
* **Contexto:** Decisão entre avaliar queda de taxa de quadros vs degradação da qualidade do detector, respondendo se a memória temporal absorve ou amplifica as falhas do detector.
* **Uso da IA:** A IA analisou o alinhamento metodológico com o simulador da Parte 0 e recomendou o teste de qualidade do detector. Calibrou as perturbações para a resolução nativa do MOT17 ($1920 \times 1080$), implementou `src/stress.py`, realizou a avaliação comparativa pareada em 4 níveis (Controle, Leve, Moderada e Severa) e gerou o painel de 3 subplots (`docs/parte5_stress_detector.png`), comprovando empiricamente que a LSTM absorve as falhas do detector (+0.029 de retenção de IDF1 e -86 ID switches no nível severo).
* **Resultado:** Projeto 100% completo com todas as partes 0 a 5 implementadas, documentadas e reproduzíveis.



