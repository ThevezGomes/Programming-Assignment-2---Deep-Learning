"""
src/training.py
Rotinas de extração de trajetórias e treinamento do modelo de movimento da Trilha A:
- TrajectoryDataset: dataset de sequências normalizadas de caixas delimitadoras
- Suporte a janelas configuráveis de BPTT truncado (T in {4, 8, 16, 32} para a Parte 3)
- Regimes de treino: Teacher Forcing e Scheduled Sampling
- Gradient Clipping configurável
- Checkpointing automático do melhor modelo na validação por sequência
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from src.data import load_mot17_sequence, get_mot17_splits
from src.models import MotionPredictor
from src.losses import CombinedMotionLoss


class TrajectoryDataset(Dataset):
    """
    Dataset que extrai trajetórias de pedestres do Ground Truth do MOT17.
    Cada amostra contém:
    - x_seq: (T, 7) características de entrada [cx, cy, w, h, vx, vy, dt]
    - init_box: (4,) caixa no instante inicial t=0
    - target_seq: (T, 4) caixas ground truth a serem previstas para t=1..T
    """
    def __init__(self, seq_paths: list, seq_len: int = 16, stride: int = 4):
        self.seq_len = seq_len
        self.samples = []
        self._extract_samples(seq_paths, seq_len, stride)

    def _extract_samples(self, seq_paths: list, T: int, stride: int):
        for p in seq_paths:
            if not os.path.exists(p):
                continue
            gt, _, info = load_mot17_sequence(p)
            W = float(info.get("imWidth", 1920))
            H = float(info.get("imHeight", 1080))

            # Agrupa caixas por identidade ao longo do tempo
            tracks = {}
            for f in sorted(gt.keys()):
                for tid, box in gt[f].items():
                    tracks.setdefault(tid, []).append((f, box))

            for tid, seq in tracks.items():
                if len(seq) < T + 1:
                    continue
                # Agrupa apenas passos estritamente consecutivos no tempo (dt == 1)
                subtracks = []
                curr_sub = [seq[0]]
                for i in range(1, len(seq)):
                    if seq[i][0] == seq[i - 1][0] + 1:
                        curr_sub.append(seq[i])
                    else:
                        if len(curr_sub) >= T + 1:
                            subtracks.append(curr_sub)
                        curr_sub = [seq[i]]
                if len(curr_sub) >= T + 1:
                    subtracks.append(curr_sub)

                for sub in subtracks:
                    for start in range(0, len(sub) - T, stride):
                        chunk = sub[start:start + T + 1]

                        # Converte para [cx, cy, w, h] normalizado em [0, 1]
                        raw_boxes = np.array([item[1] for item in chunk], dtype=np.float32)
                        cboxes = np.zeros_like(raw_boxes)
                        cboxes[:, 0] = (raw_boxes[:, 0] + raw_boxes[:, 2] / 2.0) / W
                        cboxes[:, 1] = (raw_boxes[:, 1] + raw_boxes[:, 3] / 2.0) / H
                        cboxes[:, 2] = raw_boxes[:, 2] / W
                        cboxes[:, 3] = raw_boxes[:, 3] / H

                        self.samples.append(cboxes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # chunk: (T + 1, 4)
        chunk = self.samples[idx]
        T = self.seq_len

        # init_box: caixa em t=0
        init_box = torch.tensor(chunk[0], dtype=torch.float32)

        # target_seq: caixas em t=1..T que a rede deve prever
        target_seq = torch.tensor(chunk[1:T + 1], dtype=torch.float32)

        # x_seq: entradas observadas em t=0..T-1
        # [cx, cy, w, h, vx, vy, dt]
        observed = chunk[:T]
        vx = np.zeros((T, 1), dtype=np.float32)
        vy = np.zeros((T, 1), dtype=np.float32)
        vx[1:] = observed[1:, 0:1] - observed[:-1, 0:1]
        vy[1:] = observed[1:, 1:2] - observed[:-1, 1:2]
        dt = np.ones((T, 1), dtype=np.float32)

        x_seq_np = np.concatenate([observed, vx, vy, dt], axis=-1)
        x_seq = torch.tensor(x_seq_np, dtype=torch.float32)

        return x_seq, init_box, target_seq


def build_trajectory_dataloaders(
    data_dir: str = "data/MOT17/train",
    det_suffix: str = "SDP",
    seq_len: int = 16,
    batch_size: int = 64,
    stride: int = 4
):
    """Cria os DataLoaders de treino e validação por split de sequências inteiras."""
    splits = get_mot17_splits(data_dir, det_suffix=det_suffix)
    train_ds = TrajectoryDataset(splits["train"], seq_len=seq_len, stride=stride)
    val_ds = TrajectoryDataset(splits["val"], seq_len=seq_len, stride=stride)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_motion_model(
    model: MotionPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gradient_clip: float = 1.0,
    teacher_forcing_ratio: float = 1.0,
    scheduled_sampling_decay: float = 0.05,
    save_path: str = "checkpoints/motion_lstm_best.pt",
    device: str = "cuda",
    verbose: bool = True
):
    """
    Treina o modelo de movimento com loss combinada Smooth L1 + Gaussian NLL.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)
    model.to(device)

    criterion = CombinedMotionLoss(alpha_nll=0.2, beta_smooth=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_smooth_l1 = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_smooth_l1": []}

    current_tf = teacher_forcing_ratio

    if verbose:
        print(f"Iniciando treinamento de {model.cell_type.upper()} ({epochs} épocas no {device})...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0

        for x_seq, init_box, target_seq in train_loader:
            x_seq = x_seq.to(device)
            init_box = init_box.to(device)
            target_seq = target_seq.to(device)

            optimizer.zero_grad()
            pred_boxes, pred_logvars, _ = model(
                x_seq, init_box,
                teacher_forcing_ratio=current_tf,
                target_seq=target_seq
            )

            loss, _, _ = criterion(pred_boxes, pred_logvars, target_seq)
            loss.backward()

            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            train_loss_acc += loss.item()

        # Decaimento do Teacher Forcing (Scheduled Sampling)
        current_tf = max(0.0, current_tf - scheduled_sampling_decay)
        scheduler.step()

        # Avaliação na Validação (rollout livre de observações)
        model.eval()
        val_loss_acc = 0.0
        val_reg_acc = 0.0

        with torch.no_grad():
            for x_seq, init_box, target_seq in val_loader:
                x_seq = x_seq.to(device)
                init_box = init_box.to(device)
                target_seq = target_seq.to(device)

                pred_boxes, pred_logvars, _ = model(
                    x_seq, init_box,
                    teacher_forcing_ratio=0.0,  # Autoregressivo puro na validação!
                    target_seq=target_seq
                )
                loss, l_reg, _ = criterion(pred_boxes, pred_logvars, target_seq)
                val_loss_acc += loss.item()
                val_reg_acc += l_reg.item()

        train_loss = train_loss_acc / max(1, len(train_loader))
        val_loss = val_loss_acc / max(1, len(val_loader))
        val_reg = val_reg_acc / max(1, len(val_loader))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_smooth_l1"].append(val_reg)

        if verbose:
            print(f"Época [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (Smooth-L1: {val_reg:.4f}) | TF: {current_tf:.2f}")

        # Salva pelo critério exigido: Smooth-L1 em rollout livre
        if val_reg < best_val_smooth_l1 and save_path is not None:
            best_val_smooth_l1 = val_reg
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "cell_type": model.cell_type,
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "predict_uncertainty": model.predict_uncertainty,
                "best_val_loss": val_loss,
                "best_val_smooth_l1": best_val_smooth_l1
            }, save_path)

    if verbose and save_path:
        print(f"Treinamento concluído. Checkpoint salvo em {save_path} (Melhor Val Smooth-L1: {best_val_smooth_l1:.4f})")

    return history


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executando treinamento da Trilha A (MotionPredictor LSTM) no dispositivo: {device}")
    train_loader, val_loader = build_trajectory_dataloaders(
        data_dir="data/MOT17/train", det_suffix="SDP", seq_len=16, batch_size=128, stride=4
    )
    model = MotionPredictor(cell_type="lstm", hidden_dim=128, num_layers=2, predict_uncertainty=True)
    train_motion_model(
        model, train_loader, val_loader,
        epochs=10, lr=1e-3, gradient_clip=1.0,
        save_path="checkpoints/motion_lstm_best.pt",
        device=device, verbose=True
    )

