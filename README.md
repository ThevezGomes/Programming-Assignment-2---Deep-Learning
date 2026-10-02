# Deep Learning: Programming Assignment 2 — Identidade ao longo do tempo: detecção, recorrência e rastreamento

## Integrantes do Grupo
* Henrique Gabriel Gasparelo
* José Thevez Gomes Guedes

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

## Comandos de Treinamento e Avaliação

Agora contamos com uma interface CLI centralizada (`main.py`) para facilitar o treinamento e a avaliação da rede (LSTM) contra o Baseline.

### Treinamento da LSTM (Trilha A)
Para treinar a rede recorrente (LSTM com cabeça de incerteza) a partir do zero nas trajetórias de pedestres do MOT17:
```bash
python3 main.py train
```
*(Você pode customizar o treino com flags: `--epochs 10 --seq_len 32`)*

### Avaliação (Tabela Detalhada)
Para avaliar a LSTM treinada (já com as intervenções de amortecimento e inflação da Parte 4 ligadas por padrão) exibindo uma tabela detalhada de métricas (IDF1, ID Switches, Fragmentações, etc.) por sequência:
```bash
python3 main.py eval
```

---

## Notebooks

* **`src/pipeline.ipynb`**: Contém todos os experimentos de ponta a ponta (Partes 0 a 5), tabelas, gráficos de descolamento obrigatórios e análises que embasam a apresentação oral.
* **`inferencia.ipynb`**: Executa a inferência direta em uma sequência fornecida, carregando o checkpoint salvo sem necessidade de retreino e exibindo o rastreamento renderizado.

