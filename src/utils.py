"""
src/utils.py
Utilitários de sistema, sementes aleatórias, dispositivo e checkpoints.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Garante reprodutibilidade através de todas as bibliotecas."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """Seleciona o melhor dispositivo disponível (CUDA, MPS ou CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model, path: str, extra_info: dict = None):
    """Salva checkpoint do modelo com metadados."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "extra_info": extra_info or {}
    }
    torch.save(payload, path)


def load_checkpoint(model, path: str, device: torch.device = None):
    """Carrega pesos salvos no modelo."""
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return checkpoint.get("extra_info", {})
