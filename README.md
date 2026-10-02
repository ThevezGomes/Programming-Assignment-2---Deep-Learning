# Deep Learning: Programming Assignment 2 — Identidade ao longo do tempo: detecção, recorrência e rastreamento

Repositório oficial para o **Programming Assignment 2 (PA2)** da disciplina **Aprendizado Profundo** (FGV EMAp — 2026).

## Integrantes do Grupo
* **Henrique Gabriel Gasparelo**
* **José Thevez Gomes Guedes**

---

## Conformidade Rigorosa com as Regras do Enunciado
- **Zero bibliotecas proibidas:** Nenhuma dependência externa de rastreamento (`SORT`, `DeepSORT`, `ByteTrack`, `OC-SORT`, `BoT-SORT`, `Norfair`, `motpy`, `supervision`, `model.track`), métricas prontas (`motmetrics`, `TrackEval`) ou NMS do torchvision (`torchvision.ops.nms`). Todas as métricas (IDF1, ID switches, fragmentações, mAP) e o NMS foram desenvolvidos do zero em Python/NumPy/SciPy.
- **Kalman apenas como baseline:** O filtro de Kalman não é utilizado como modelo temporal na Parte 2; o modelo temporal central é estritamente uma rede neural recorrente (LSTM).
- **Sem conclusões ou números manuais:** Todo texto, diagnóstico, título de gráfico e valor de tabela é derivado e formatado dinamicamente via medição direta dos dados.
- **Protocolo Único de Avaliação:** Fixado globalmente em `EVAL_IOU = 0.50` (`src/config.py`), garantindo consistência estrita entre todas as Partes (0 a 5).
- **Modelo Final Único:** O checkpoint `checkpoints/motion_lstm_best.pt` ($T=16, h=128$, 2 camadas, `predict_uncertainty=True`) é o modelo canônico avaliado nas Partes 2, 4 e 5 e no `inferencia.ipynb`.

---

## Estrutura do Repositório

```text
.
├── checkpoints/
│   ├── motion_lstm_best.pt      # Modelo canônico final (LSTM 2 camadas, h=128, T=16)
│   └── ablation/                # Checkpoints da ablação Eixo 1 (RNN, GRU, LSTM)
├── data/
│   └── MOT17/                   # Sequências oficiais do MOT17 (train/MOT17-*-SDP)
├── docs/                        # Figuras geradas pelo pipeline para a apresentação
│   ├── parte0_trajetoria_oclusao.png
│   ├── parte0_curva_quebra.png
│   ├── parte1_comparacao_fontes_real.png
│   ├── painel_descolamento_parte1.png
│   ├── parte2_comparacao_baseline_trilhaA.png
│   ├── parte3_ablacao_eixo1.png
│   ├── parte4_horizonte_analitico.png
│   ├── parte4_horizonte_empirico.png
│   ├── parte4_falha*.png
│   └── parte5_stress_detector.png
├── results/                     # Resultados brutos em JSON
│   ├── ablation_eixo1.json
│   └── stress_detector.json
├── src/
│   ├── config.py                # EVAL_IOU = 0.5, DEFAULT_MAX_LOST_FRAMES = 15, checkpoints
│   ├── data.py                  # Gerador sintético com z-order, oclusão controlada e parser MOT17
│   ├── metrics.py               # IDF1 (Ristani et al. 2016), ID switches, mAP e custom_nms
│   ├── models.py                # MotionPredictor recorrente (RNN, GRU, LSTM) com incerteza
│   ├── tracker.py               # NaiveTracker e RNNMotionTracker (Hungarian + portão adaptativo)
│   ├── training.py              # Dataloaders contíguos de trajetórias e treino com Scheduled Sampling
│   ├── evaluation.py            # Avaliação nas sequências MOT17 e tabelas 1.0, 1.1, 1.2
│   ├── ablation.py              # Ablação Eixo 1 (busca de hidden_dim ~40k, 3 seeds, ADE/FDE)
│   ├── failures.py              # Horizonte analítico (||dL/dh||), empírico e galeria de falhas
│   ├── stress.py                # Teste de estresse com 3 seeds e baseline Naive relaxado
│   ├── inference.py             # Pipeline universal de inferência (MOT17, pastas ou vídeos)
│   ├── visualization.py         # Gráficos e painéis sem textos hardcoded
│   └── pipeline.ipynb           # Notebook completo de ponta a ponta (Partes 0 a 5)
├── tests/
│   └── test_metrics.py          # Testes unitários formais (casos a, b, c, d, e)
├── inferencia.ipynb             # Notebook de inferência sem retreino (produz GIF e MP4)
├── metrics.py                   # Reexport das métricas proprietárias na raiz
├── pytest.ini                   # Configuração do pytest
├── AI_LOG.md                    # Registro transparente de uso de ferramentas de IA
├── README.md                    # Instruções completas de configuração e reprodução
└── requirements.txt             # Dependências do projeto
```

---

## Instalação e Ambiente

1. Crie e ative o ambiente virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Verifique os testes unitários das métricas:
```bash
pytest -v
```

---

## Comandos Diretos de Reprodução

### 1. Testes Unitários das Métricas (Parte 0)
```bash
python3 metrics.py
# ou
pytest -v
```

### 2. Treinamento do Modelo Canônico (Parte 2 — Trilha A)
Treina a rede LSTM ($h=128$, 2 camadas, $T=16$) com Scheduled Sampling e perda combinada Smooth-L1 + Gaussian NLL:
```bash
PYTHONPATH=. python3 src/training.py
```

### 3. Ablação do Eixo 1 (Parte 3 — Célula Recorrente e Janela BPTT)
Treina as 12 configurações (RNN $h=113$, GRU $h=74$, LSTM $h=65$ controladas em $\sim 40\text{k}$ parâmetros) em 3 sementes aleatórias ($42, 100, 2026$) e gera a tabela com IDF1, Smooth-L1 e ADE@1..30:
```bash
PYTHONPATH=. python3 src/ablation.py
```

### 4. Horizonte de Memória e Galeria de Falhas (Parte 4)
Calcula a taxa de decaimento do gradiente analítico, o horizonte empírico de oclusão e seleciona automaticamente 3 falhas reais com predição na oclusão:
```bash
PYTHONPATH=. python3 -c "
from src.failures import compute_analytical_gradient_norm, compute_empirical_horizon, plot_failure_gallery, demonstrate_fix
compute_analytical_gradient_norm()
compute_empirical_horizon()
plot_failure_gallery()
demonstrate_fix()
"
```

### 5. Teste de Estresse do Detector (Parte 5)
Avalia 3 sementes aleatórias por nível (Controle, Leve, Moderada, Severa) nas sequências MOT17-09 e MOT17-11 contra o Baseline Ingênuo e o Naive com limiar adaptativo:
```bash
PYTHONPATH=. python3 src/stress.py
```

### 6. Inferência Direta em Sequência Arbitrária (`inferencia.ipynb`)
Gera GIF e vídeo MP4 com cores consistentes por ID e contagem de pedestres únicos confirmados ($\ge 3$ detecções):
```bash
PYTHONPATH=. python3 -c "
from src.inference import run_tracking_inference
run_tracking_inference(
    seq_path='data/MOT17/train/MOT17-09-SDP',
    model_path='checkpoints/motion_lstm_best.pt',
    output_video_path='results/inferencia_output.gif',
    max_frames=40
)
"
```

---

## Notebooks Principais

* **`src/pipeline.ipynb`**: Notebook mestre executando de ponta a ponta as Partes 0 a 5, gerando todas as tabelas e gráficos que fundamentam o relatório e a apresentação oral.
* **`inferencia.ipynb`**: Notebook executável para inferência direta sem retreino, aceitando pastas MOT17, diretórios de imagens ou arquivos de vídeo.
