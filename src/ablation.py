import sys, os
import json
import torch
import numpy as np
from src.models import MotionPredictor
from src.training import build_trajectory_dataloaders, train_motion_model
from src.tracker import RNNMotionTracker, NaiveTracker
from src.evaluation import run_part2_comparison
from src.utils import set_seed

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_hidden_dim_for_budget(cell_type: str):
    if cell_type == "lstm": return 64
    if cell_type == "gru": return 74
    if cell_type == "rnn": return 128
    return 64

def run_ablation_axis1(data_dir="data/MOT17/train", force_recompute=False, results_path="results/ablation_eixo1.json"):
    """
    Executa a ablação do Eixo 1 (Célula Recorrente e Janela de BPTT).
    Treina RNN, GRU e LSTM para T em [4, 8, 16, 32] com 3 seeds diferentes.
    """
    if not force_recompute and os.path.exists(results_path):
        print(f"Resultados já encontrados em {results_path}. Usando cache (force_recompute=False).")
        with open(results_path, "r") as f:
            return json.load(f)
            
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Iniciando cálculo da Ablação Eixo 1 no {device} (Isso pode levar de 5 a 10 minutos)...")
    
    cells = ["rnn", "gru", "lstm"]
    windows = [4, 8, 16, 32]
    seeds = [42, 100, 2026]
    
    results = []
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    os.makedirs("checkpoints/ablation", exist_ok=True)
    
    for cell in cells:
        h_dim = get_hidden_dim_for_budget(cell)
        for T in windows:
            idf1_list = []
            print(f"Treinando {cell.upper()} | Janela T={T:02d} | Params aprox: 35k", end="")
            
            train_loader, val_loader = build_trajectory_dataloaders(
                data_dir=data_dir, det_suffix="SDP", seq_len=T, batch_size=256, stride=4
            )
            
            for seed in seeds:
                set_seed(seed)
                model = MotionPredictor(cell_type=cell, hidden_dim=h_dim, num_layers=1, predict_uncertainty=True)
                save_path = f"checkpoints/ablation/model_{cell}_T{T}_seed{seed}.pt"
                
                # Desativa prints internos para manter o notebook limpo
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                try:
                    train_motion_model(
                        model, train_loader, val_loader,
                        epochs=5, lr=1e-3, gradient_clip=1.0,
                        teacher_forcing_ratio=1.0, scheduled_sampling_decay=0.0,
                        save_path=save_path, device=device, verbose=False
                    )
                    tracker = RNNMotionTracker(save_path, iou_threshold=0.3, max_lost_frames=15, device=device)
                    baseline = NaiveTracker(iou_threshold=0.3, max_lost_frames=15)
                    comp = run_part2_comparison(data_dir, baseline, tracker, seq_names=["09"])
                finally:
                    sys.stdout = old_stdout
                
                idf1 = comp[0]["rnn"]["idf1"]
                idf1_list.append(idf1)
                print(".", end="")
                sys.stdout.flush()
            
            mean_idf1 = float(np.mean(idf1_list))
            std_idf1 = float(np.std(idf1_list))
            print(f" Concluído => IDF1 Médio: {mean_idf1:.4f} ± {std_idf1:.4f}")
            
            results.append({
                "cell": cell,
                "T": T,
                "hidden_dim": h_dim,
                "idf1_mean": mean_idf1,
                "idf1_std": std_idf1,
                "raw_idf1s": idf1_list
            })
            
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
                
    print("Cálculo da ablação concluído!")
    return results
