"""
src/config.py
Constantes centrais e protocolo único de avaliação para todo o PA2.
"""

import os

# Protocolo único de avaliação de tracking e detecção (P0-1)
EVAL_IOU = 0.5

# Ciclo de vida padrão de tracks
DEFAULT_MAX_LOST_FRAMES = 15
DEFAULT_TRACKER_IOU = 0.3

# Checkpoint único do Modelo Final (P0-2)
FINAL_CHECKPOINT_PATH = os.path.join("checkpoints", "motion_lstm_best.pt")
FINAL_MODEL_CONFIG = {
    "cell_type": "lstm",
    "hidden_dim": 128,
    "num_layers": 2,
    "bptt_window": 16,
    "predict_uncertainty": True,
}
