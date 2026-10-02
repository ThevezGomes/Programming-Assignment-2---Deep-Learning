import argparse
import torch
import os

from src.models import MotionPredictor
from src.training import build_trajectory_dataloaders, train_motion_model
from src.tracker import RNNMotionTracker, NaiveTracker
from src.evaluation import run_part2_comparison

def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Iniciando treinamento da LSTM no dispositivo: {device}")
    print(f"Configurações: Janela T={args.seq_len}, Épocas={args.epochs}")
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    train_loader, val_loader = build_trajectory_dataloaders(
        data_dir=args.data_dir, det_suffix="SDP", seq_len=args.seq_len, batch_size=256, stride=4
    )
    
    # Modelo Final (LSTM com cabeça de incerteza)
    model = MotionPredictor(cell_type="lstm", hidden_dim=64, num_layers=1, predict_uncertainty=True)
    
    # Treinamento forçando o Teacher Forcing para evitar divergência inicial
    train_motion_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=1e-3, gradient_clip=1.0,
        teacher_forcing_ratio=1.0, scheduled_sampling_decay=0.0,
        save_path=args.save_path, device=device, verbose=True
    )
    print(f"\n[OK] Treinamento concluído. Modelo salvo em {args.save_path}")

def run_detailed_evaluation(data_dir, tracker, seq_names, det_suffix="SDP"):
    """Roda a avaliação apenas do modelo escolhido, mostrando uma tabela completa de métricas."""
    from src.data import load_mot17_sequence, compute_sequence_density
    from src.metrics import evaluate_tracking, compute_map_per_frame
    
    SEP = "=" * 105
    print(SEP)
    print("AVALIAÇÃO DETALHADA DO MODELO (LSTM Tracker)")
    print(SEP)
    print("{:<10} | {:<10} | {:<8} | {:<6} | {:<6} | {:<8} | {:<10} | {:<12}".format(
        "Sequência", "Densidade", "IDF1", "IDSW", "Frag", "IDs Pred", "Razão IDs", "mAP (Deteção)"
    ))
    print("-" * 105)
    
    results = []
    
    for s_name in seq_names:
        seq_p = os.path.join(data_dir, f"MOT17-{s_name}-{det_suffix}")
        if not os.path.exists(seq_p):
            print(f"MOT17-{s_name} não encontrado. Pulando...")
            continue
            
        gt, dets, info = load_mot17_sequence(seq_p)
        w = float(info.get("imWidth", 1920))
        h = float(info.get("imHeight", 1080))
        density = compute_sequence_density(gt)
        
        map_score = compute_map_per_frame(gt, dets)
        
        tracker.reset()
        preds = tracker.track_sequence(dets, im_width=w, im_height=h)
        m = evaluate_tracking(gt, preds)
        
        print("{:<10} | {:<10.1f} | {:<8.3f} | {:<6d} | {:<6d} | {:<8d} | {:<9.2f}x | {:<12.3f}".format(
            f"MOT17-{s_name}", density, m["idf1"], m["id_switches"], m["fragmentations"], 
            m["unique_pred_ids"], m["ratio_ids"], map_score
        ))
        results.append(m)
        
    print(SEP + "\n")
    return results

def evaluate(args):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Iniciando avaliação no dispositivo: {device}")
    
    if not os.path.exists(args.model_path):
        print(f"Erro: Modelo não encontrado em {args.model_path}.")
        print("Dica: Rode o comando 'python3 main.py train' primeiro.")
        return
        
    # Tracker com a nossa LSTM e as intervenções da Parte 4 ligadas (damping e inflação)
    print(f"Carregando modelo e configurando Rastreador com Intervenções de Oclusão...")
    tracker = RNNMotionTracker(
        args.model_path, 
        iou_threshold=0.3, 
        max_lost_frames=15, 
        use_adaptive_gating=True,
        velocity_damping=0.85,    # Intervenção Parte 4
        sigma_inflation=0.05,     # Intervenção Parte 4
        device=device
    )
    
    seqs = args.seqs.split(",")
    print(f"Avaliando sequências: {seqs}\n")
    
    run_detailed_evaluation(args.data_dir, tracker, seqs, det_suffix="SDP")

def main():
    parser = argparse.ArgumentParser(description="Pipeline Final de Rastreamento (MOT) - PA2")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Comando a ser executado")
    
    # Comando 1: Treinar
    parser_train = subparsers.add_parser("train", help="Treina a LSTM a partir do zero")
    parser_train.add_argument("--data_dir", type=str, default="data/MOT17/train", help="Diretório dos dados MOT17")
    parser_train.add_argument("--epochs", type=int, default=5, help="Número de épocas de treino")
    parser_train.add_argument("--seq_len", type=int, default=16, help="Tamanho da janela (T)")
    parser_train.add_argument("--save_path", type=str, default="checkpoints/final_model.pt", help="Onde salvar o checkpoint")
    
    # Comando 2: Avaliar
    parser_eval = subparsers.add_parser("eval", help="Avalia o modelo treinado contra o Baseline")
    parser_eval.add_argument("--data_dir", type=str, default="data/MOT17/train", help="Diretório dos dados MOT17")
    parser_eval.add_argument("--model_path", type=str, default="checkpoints/final_model.pt", help="Caminho do modelo")
    parser_eval.add_argument("--seqs", type=str, default="09,11,05", help="Sequências separadas por vírgula")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)

if __name__ == "__main__":
    main()
