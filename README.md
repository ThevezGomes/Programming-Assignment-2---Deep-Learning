# Deep Learning: Programming Assignment 2 — Identidade ao longo do tempo: detecção, recorrência e rastreamento

O objetivo deste repositório é atender aos requisitos do Programming Assignment 2 da disciplina Aprendizado Profundo da FGV EMAp.

## Integrantes do Grupo
* Henrique Gabriel Gasparelo
* [Nome do Colega de Dupla]

---

## Estrutura do Repositório

```text
.
├── checkpoints/
│   └── lstm_best.pt             # Pesos do modelo temporal treinado
├── data/
│   ├── MOT17/                   # Sequências do MOT17 (anotações e detecções)
│   └── synthetic/               # Dados sintéticos gerados (Parte 0)
├── docs/
│   └── PA2.pdf                  # Enunciado oficial do projeto
├── src/
│   ├── __init__.py 
│   ├── data.py                  # Gerador sintético com z-order, simulador de falhas e parser MOT17
│   ├── failures.py              # Horizonte de memória analítico (||dL/dh||) e empírico + correção
│   ├── inference.py             # Funções de suporte para o inferencia.ipynb
│   ├── losses.py                # Smooth L1 Loss para trajetórias de caixas
│   ├── metrics.py               # IDF1, ID switches, fragmentações e NMS próprio
│   ├── models.py                # Modelos de movimento (RNN Simples, LSTM, GRU)
│   ├── stress.py                # Testes de estresse degradando o detector em 3 intensidades
│   ├── tracker.py               # Lógica de associação Húngara e gestão do ciclo de vida das tracks
│   ├── training.py              # Rotinas de treino, BPTT truncado e ablações multi-seed
│   ├── utils.py                 # Seeds, seleção de dispositivo (MPS/CUDA/CPU) e checkpoints
│   ├── visualization.py         # Gráficos de descolamento, curvas de gradiente e vídeos com tracks
│   └── pipeline.ipynb           # Notebook completo de ponta a ponta com as Partes 0 a 5
├── inferencia.ipynb             # Notebook para testar o modelo em uma nova sequência
├── metrics.py                   # Implementação própria de IDF1 e ID switches exposta na raiz
├── AI_LOG.md                    # Relatório de uso de ferramentas de IA
├── README.md                    # Instruções de reprodução e comandos
└── requirements.txt             # Dependências do projeto
```

---

## Ambiente e Instalação

Crie um ambiente virtual e instale as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Download dos Dados

Para baixar o conjunto de anotações e detecções públicas do MOT17 (~10 MB):

```bash
mkdir -p data/MOT17
curl -O https://motchallenge.net/data/MOT17Labels.zip
unzip MOT17Labels.zip -d data/MOT17/
```

Para baixar as imagens completas caso deseje renderizar vídeos (opcional para o treino do modelo de caixas):
[https://motchallenge.net/data/MOT17/](https://motchallenge.net/data/MOT17/)

---

## Parte 1 — Regras de Associação e Gestão do Ciclo de Vida (Item 4)

O rastreador de linha de base (`NaiveTracker` em `src/tracker.py`) opera sem memória temporal recorrente, associando caixas entre quadros consecutivos ($t$ e $t-1$):

1. **Matriz de Custo e Matching:**
   * Calcula a matriz de sobreposição espacial por **IoU** (`box_iou_matrix` de autoria própria).
   * Associação ótima 1-para-1 via **Algoritmo Húngaro** (`scipy.optimize.linear_sum_assignment(-ious)`), com opção de matching **Guloso** (ordenação decrescente por IoU).
   * **Limiar de corte:** $\text{IoU} \ge 0.3$. Pares com sobreposição inferior a 0.3 são descartados para evitar falso pareamento.

2. **Gestão do Ciclo de Vida:**
   * **Nascimento:** Toda detecção que não é associada a nenhuma trajetória existente inicia uma nova track com um identificador numérico único incremental (`next_id += 1`).
   * **Oclusão / Lost:** Trajetórias ativas que não recebem detecção no quadro atual acumulam um contador `time_since_update += 1` e passam para o estado `lost`.
   * **Morte:** Se uma track permanecer mais de $k = 15$ quadros consecutivos sem observação (`time_since_update > max_lost_frames`), ela é marcada como `dead` e extinta permanentemente. Se o objeto reaparecer após $k$ quadros, ele receberá um novo ID (gerando fragmentação e ID switch).

---

## Comandos de Treinamento e Avaliação

### 1. Treinamento
Para rodar o treinamento do modelo temporal e as ablações com 3 seeds:
```bash
python3 -c "import src.training as t; print('Comando de treino')"
```

### 2. Avaliação
Para executar a validação das métricas e os testes unitários da Parte 0 (casos a, b, c):
```bash
python3 metrics.py
```

Para reproduzir a avaliação completa do baseline ingênuo nas sequências do MOT17 (Parte 1):
```bash
python3 -c "from src.tracker import NaiveTracker; from src.evaluation import run_mot17_baseline; run_mot17_baseline('data/MOT17/train', NaiveTracker())"
```

---

## Notebooks

* **`src/pipeline.ipynb`**: Contém todos os experimentos de ponta a ponta (Partes 0 a 5), tabelas, gráficos de descolamento obrigatórios e análises que embasam a apresentação oral.
* **`inferencia.ipynb`**: Executa a inferência direta em uma sequência fornecida, carregando o checkpoint salvo sem necessidade de retreino e exibindo o rastreamento renderizado.

