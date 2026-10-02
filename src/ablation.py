"""
src/ablation.py
Parte 3 — Ablação do Eixo 1: A Célula Recorrente e Janela de BPTT.

Estudo empírico comparando:
- Células: RNN Simples vs GRU vs LSTM calibradas com contagem de parâmetros igual (~40k, tolerância ±5%).
- Janelas de BPTT truncado: T in {4, 8, 16, 32}.
- Reprodutibilidade com 3 seeds por configuração (42, 100, 2026).
- Métricas: Val Smooth-L1 em rollout livre, Erro de Rollout (ADE e FDE para 1, 5, 15, 30 passos sem observação)
  e IDF1 com protocolo unificado EVAL_IOU = 0.5 na sequência MOT17-11 (separada da escolha de hiperparâmetros).
"""

import sys
import os
import json
import torch
import numpy as np

from src.config import EVAL_IOU, DEFAULT_TRACKER_IOU, DEFAULT_MAX_LOST_FRAMES
from src.models import MotionPredictor
from src.training import build_trajectory_dataloaders, train_motion_model
from src.tracker import RNNMotionTracker, NaiveTracker
from src.metrics import evaluate_tracking
from src.data import load_mot17_sequence
from src.utils import set_seed


def count_parameters(model):
    """Conta os parâmetros treináveis do modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_hidden_dim_for_budget(cell_type: str, target_params: int = 40000, tolerance: float = 0.05) -> tuple:
    """
    Busca sistemática do hidden_dim que aproxima o orçamento de ~40k parâmetros (±5%).
    Retorna (hidden_dim, num_parameters).
    """
    best_h = None
    best_diff = float("inf")
    best_params = 0

    for h in range(16, 256):
        m = MotionPredictor(cell_type=cell_type, hidden_dim=h, num_layers=1, predict_uncertainty=True)
        p = count_parameters(m)
        diff = abs(p - target_params)
        if diff < best_diff:
            best_diff = diff
            best_h = h
            best_params = p

    pct_diff = (best_params - target_params) / target_params
    if abs(pct_diff) > tolerance:
        print(f"[AVISO] {cell_type.upper()} com h={best_h} tem {best_params} parâmetros ({pct_diff:+.1%}), fora da tolerância ±{tolerance:.0%}")
    return best_h, best_params


def evaluate_rollout_metrics(model, val_loader, steps_list=(1, 5, 15, 30), device="cpu"):
    """
    Calcula erro de rollout livre sem observação:
    - ADE (Average Displacement Error)
    - FDE (Final Displacement Error)
    Para horizontes de 1, 5, 15 e 30 passos.
    """
    model.eval()
    device = torch.device(device)
    max_steps = max(steps_list)

    ade_accum = {k: [] for k in steps_list}
    fde_accum = {k: [] for k in steps_list}

    with torch.no_grad():
        for x_seq, init_box, target_seq in val_loader:
            # Amostra com comprimento suficiente
            B, T, _ = target_seq.shape
            rollout_horizon = min(max_steps, T)
            if rollout_horizon < 1:
                continue

            x_seq = x_seq.to(device)
            init_box = init_box.to(device)
            target_seq = target_seq.to(device)

            h0 = model.init_hidden(B, device)
            # Primeiro passo com entrada inicial
            pred_boxes_rollout, _, _ = model.rollout(init_box, steps=rollout_horizon, hidden=h0)

            for k in steps_list:
                if k <= rollout_horizon:
                    pred_k = pred_boxes_rollout[:, :k, :2]  # centro [cx, cy]
                    tgt_k = target_seq[:, :k, :2]
                    # L2 distance
                    dist = torch.norm(pred_k - tgt_k, dim=-1)  # (B, k)
                    ade_val = dist.mean().item()
                    fde_val = dist[:, -1].mean().item()
                    ade_accum[k].append(ade_val)
                    fde_accum[k].append(fde_val)

    ade_res = {f"ade_{k}": float(np.mean(ade_accum[k])) if ade_accum[k] else 0.0 for k in steps_list}
    fde_res = {f"fde_{k}": float(np.mean(fde_accum[k])) if fde_accum[k] else 0.0 for k in steps_list}
    return {**ade_res, **fde_res}


def run_ablation_axis1(
    data_dir="data/MOT17/train",
    eval_seq_id="11",
    force_recompute=False,
    results_path="results/ablation_eixo1.json",
    epochs=10
):
    """
    Executa a ablação completa do Eixo 1 (Célula Recorrente e Janela de BPTT).
    Treina RNN, GRU e LSTM com parâmetros igualados (~40k) para T em [4, 8, 16, 32]
    com 3 sementes aleatórias.
    Reporta Smooth-L1 em rollout livre, ADE/FDE para 1, 5, 15, 30 passos e IDF1 (média ± desvio).
    """
    if not force_recompute and os.path.exists(results_path):
        print(f"Resultados encontrados em {results_path}. Carregando cache (force_recompute=False)...")
        with open(results_path, "r") as f:
            return json.load(f)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Iniciando Ablação Eixo 1 no dispositivo: {device}")

    cells = ["rnn", "gru", "lstm"]
    windows = [4, 8, 16, 32]
    seeds = [42, 100, 2026]

    # 1. Calibração e impressão dos parâmetros reais
    SEP = "=" * 90
    print("\n" + SEP)
    print("CALIBRAÇÃO DE PARÂMETROS POR BUSCA SISTEMÁTICA (Orçamento alvo: ~40.000 parâmetros)")
    print(SEP)
    dim_budget = {}
    for c in cells:
        h, p = find_hidden_dim_for_budget(c, target_params=40000, tolerance=0.05)
        dim_budget[c] = (h, p)
        print(f"• {c.upper():<4} => hidden_dim = {h:<3d} | Parâmetros Reais = {p:<6d} (desvio: {(p - 40000)/40000:+.1%})")
    print(SEP + "\n")

    eval_seq_path = os.path.join(data_dir, f"MOT17-{eval_seq_id}-SDP")
    if not os.path.exists(eval_seq_path):
        eval_seq_path = os.path.join(data_dir, "MOT17-09-SDP")
        print(f"Sequência MOT17-{eval_seq_id} não encontrada. Utilizando {os.path.basename(eval_seq_path)} para avaliação.")

    gt_eval, dets_eval, info_eval = load_mot17_sequence(eval_seq_path)
    im_w = float(info_eval.get("imWidth", 1920))
    im_h = float(info_eval.get("imHeight", 1080))

    results = []
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    os.makedirs("checkpoints/ablation", exist_ok=True)

    for cell in cells:
        h_dim, n_params = dim_budget[cell]
        for T in windows:
            idf1_list = []
            smooth_l1_list = []
            rollout_metrics_list = []

            print(f"Executando {cell.upper()} | Janela T={T:02d} | Params={n_params} (3 seeds)...", end="", flush=True)

            train_loader, val_loader = build_trajectory_dataloaders(
                data_dir=data_dir, det_suffix="SDP", seq_len=T, batch_size=128, stride=4
            )

            for seed in seeds:
                set_seed(seed)
                model = MotionPredictor(cell_type=cell, hidden_dim=h_dim, num_layers=1, predict_uncertainty=True)
                save_path = f"checkpoints/ablation/model_{cell}_T{T}_seed{seed}.pt"

                # Treinamento silencioso
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")
                try:
                    history = train_motion_model(
                        model, train_loader, val_loader,
                        epochs=epochs, lr=1e-3, gradient_clip=1.0,
                        teacher_forcing_ratio=0.8, scheduled_sampling_decay=0.08,
                        save_path=save_path, device=device, verbose=False
                    )
                    best_l1 = min(history.get("val_smooth_l1", [1.0]))
                    smooth_l1_list.append(best_l1)

                    # Rollout errors
                    rm = evaluate_rollout_metrics(model, val_loader, steps_list=(1, 5, 15, 30), device=device)
                    rollout_metrics_list.append(rm)

                    # Avaliação do Tracker com protocolo unificado EVAL_IOU = 0.5
                    tracker = RNNMotionTracker(save_path, iou_threshold=DEFAULT_TRACKER_IOU,
                                               max_lost_frames=DEFAULT_MAX_LOST_FRAMES, device=device)
                    preds = tracker.track_sequence(dets_eval, im_width=im_w, im_height=im_h)
                    m = evaluate_tracking(gt_eval, preds, iou_threshold=EVAL_IOU)
                    idf1_list.append(m["idf1"])
                finally:
                    sys.stdout = old_stdout

                print(".", end="", flush=True)

            mean_idf1 = float(np.mean(idf1_list))
            std_idf1 = float(np.std(idf1_list))
            mean_l1 = float(np.mean(smooth_l1_list))
            std_l1 = float(np.std(smooth_l1_list))

            avg_rollout = {}
            for k in ["ade_1", "ade_5", "ade_15", "ade_30", "fde_1", "fde_5", "fde_15", "fde_30"]:
                avg_rollout[k] = float(np.mean([r[k] for r in rollout_metrics_list]))

            print(f" Concluído => IDF1: {mean_idf1:.4f} ± {std_idf1:.4f} | Val Smooth-L1: {mean_l1:.4f}")

            entry = {
                "cell": cell,
                "T": T,
                "hidden_dim": h_dim,
                "num_params": n_params,
                "idf1_mean": mean_idf1,
                "idf1_std": std_idf1,
                "raw_idf1s": idf1_list,
                "val_smooth_l1_mean": mean_l1,
                "val_smooth_l1_std": std_l1,
                **avg_rollout
            }
            results.append(entry)

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    print("\nAblação Eixo 1 concluída com sucesso! Resultados salvos em:", results_path)
    return results


def print_ablation_summary_table(results: list):
    """Imprime tabela formatada com todos os resultados da ablação (Eixo 1)."""
    SEP = "=" * 105
    print("\n" + SEP)
    print("TABELA COMPLETA — ABLAÇÃO EIXO 1 (Célula Recorrente e Janela BPTT T)")
    print(SEP)
    header = "{:<5} | {:<3} | {:<7} | {:<16} | {:<14} | {:<8} | {:<8} | {:<8} | {:<8}".format(
        "Célula", "T", "Params", "IDF1 (Média±Std)", "Val Smooth-L1", "ADE@1", "ADE@5", "ADE@15", "ADE@30"
    )
    print(header)
    print("-" * 105)

    for r in results:
        idf_str = f"{r['idf1_mean']:.4f} ± {r['idf1_std']:.4f}"
        l1_str = f"{r['val_smooth_l1_mean']:.4f}"
        print("{:<5} | {:<3d} | {:<7d} | {:<16} | {:<14} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
            r["cell"].upper(), r["T"], r["num_params"], idf_str, l1_str,
            r.get("ade_1", 0.0), r.get("ade_5", 0.0), r.get("ade_15", 0.0), r.get("ade_30", 0.0)
        ))
    print(SEP + "\n")


if __name__ == "__main__":
    res = run_ablation_axis1(data_dir="data/MOT17/train", eval_seq_id="11", force_recompute=True, epochs=10)
    print_ablation_summary_table(res)
