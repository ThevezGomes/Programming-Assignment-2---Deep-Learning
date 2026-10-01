"""
src/stress.py
Parte 5 — Teste de Estresse: Degradação da Qualidade do Detector.

Avalia o comportamento do rastreador sob perturbações controladas nas detecções
públicas do MOT17 (descarte p%, ruído nas caixas e injeção de falsos positivos)
em 3 intensidades (mais o controle original), medindo simultaneamente mAP e IDF1
para determinar se a recorrência temporal absorve ou amplifica as falhas do detector.
"""

import os
import json
import numpy as np

from src.data import load_mot17_sequence
from src.tracker import NaiveTracker, RNNMotionTracker
from src.metrics import evaluate_tracking, compute_map_per_frame


DEFAULT_STRESS_LEVELS = [
    {
        "level": "Controle",
        "name": "0. Controle (Original)",
        "drop_prob": 0.00,
        "noise_std": 0.0,
        "fp_rate": 0.00,
        "description": "Detecções SDP originais sem alteração"
    },
    {
        "level": "Leve",
        "name": "1. Leve",
        "drop_prob": 0.10,
        "noise_std": 5.0,
        "fp_rate": 0.05,
        "description": "Oclusões breves e leve imprecisão de borda"
    },
    {
        "level": "Moderada",
        "name": "2. Moderada",
        "drop_prob": 0.25,
        "noise_std": 12.0,
        "fp_rate": 0.12,
        "description": "Detector oscilante com perda frequente de frames"
    },
    {
        "level": "Severa",
        "name": "3. Severa",
        "drop_prob": 0.45,
        "noise_std": 25.0,
        "fp_rate": 0.25,
        "description": "Baixa confiança, jitter elevado e múltiplos falsos alarmes"
    },
]


def degrade_mot17_detections(
    det_by_frame: dict,
    im_w: int = 1920,
    im_h: int = 1080,
    drop_prob: float = 0.0,
    noise_std: float = 0.0,
    fp_rate: float = 0.0,
    seed: int = 42
) -> dict:
    """
    Aplica perturbações realistas nas detecções de pedestres do MOT17:
    1. Descarte de detecções (False Negatives) com probabilidade drop_prob.
    2. Ruído gaussiano nas coordenadas [x, y, w, h] proporcional à resolução.
    3. Injeção de falsos positivos com proporções condizentes com pedestres.
    """
    if drop_prob == 0.0 and noise_std == 0.0 and fp_rate == 0.0:
        return {f: [list(d) for d in dets] for f, dets in det_by_frame.items()}

    rng = np.random.default_rng(seed)
    degraded = {}

    all_frames = sorted(list(det_by_frame.keys()))
    for f in all_frames:
        orig_dets = det_by_frame.get(f, [])
        new_dets = []

        # 1 & 2: Descarte e ruído nas detecções reais
        for d in orig_dets:
            if rng.random() < drop_prob:
                continue

            x, y, w, h = float(d[0]), float(d[1]), float(d[2]), float(d[3])
            conf = float(d[4]) if len(d) > 4 else 1.0

            if noise_std > 0.0:
                # Perturba centroide e dimensões
                x += rng.normal(0, noise_std)
                y += rng.normal(0, noise_std)
                w += rng.normal(0, noise_std * 0.5)
                h += rng.normal(0, noise_std * 0.5)

            # Garante limites válidos dentro da resolução
            x = float(np.clip(x, 0, im_w - 2))
            y = float(np.clip(y, 0, im_h - 2))
            w = float(np.clip(w, 4, im_w - x))
            h = float(np.clip(h, 8, im_h - y))

            new_dets.append([x, y, w, h, conf])

        # 3: Injeção de falsos positivos aleatórios
        if fp_rate > 0.0 and len(orig_dets) > 0:
            n_fp = rng.poisson(fp_rate * len(orig_dets))
            for _ in range(int(n_fp)):
                # Dimensões realistas de pedestres em MOT17
                fp_w = float(rng.uniform(25, 75))
                aspect_ratio = float(rng.uniform(2.0, 3.2))
                fp_h = float(min(im_h - 2, fp_w * aspect_ratio))
                fp_x = float(rng.uniform(0, max(1, im_w - fp_w)))
                fp_y = float(rng.uniform(0, max(1, im_h - fp_h)))
                fp_conf = float(rng.uniform(0.30, 0.60))
                new_dets.append([fp_x, fp_y, fp_w, fp_h, fp_conf])

        degraded[f] = new_dets

    return degraded


def run_detector_stress_experiment(
    seq_path: str = "data/MOT17/train/MOT17-09-SDP",
    model_path: str = "checkpoints/motion_lstm_best.pt",
    levels: list = None,
    velocity_damping: float = 0.85,
    sigma_inflation: float = 0.30,
    iou_threshold: float = 0.30,
    max_lost_frames: int = 15,
    seed: int = 42,
    save_results_path: str = "results/stress_detector.json",
    save_plot_path: str = "docs/parte5_stress_detector.png"
) -> dict:
    """
    Executa o teste de estresse da Parte 5 sobre uma sequência MOT17:
    - Avalia NaiveTracker vs RNNMotionTracker (LSTM) em 4 patamares de qualidade.
    - Mede mAP por quadro e métricas temporais (IDF1, IDSW, Frag, Ratio).
    - Salva os resultados estruturados e renderiza os gráficos.
    """
    if levels is None:
        levels = DEFAULT_STRESS_LEVELS

    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Sequência não encontrada: {seq_path}")

    gt_by_frame, det_by_frame, info = load_mot17_sequence(seq_path)
    im_w = int(info.get("imWidth", 1920))
    im_h = int(info.get("imHeight", 1080))
    seq_name = info.get("name", os.path.basename(seq_path))

    SEP = "=" * 94
    print(SEP)
    print(f"PARTE 5: TESTE DE ESTRESSE — DEGRADAÇÃO DA QUALIDADE DO DETECTOR")
    print(f"Sequência: {seq_name} ({im_w}x{im_h}) | Checkpoint: {os.path.basename(model_path)}")
    print(f"Configuração LSTM: velocity_damping={velocity_damping}, sigma_inflation={sigma_inflation}")
    print(SEP)
    header = f"{'Nível':<10} | {'mAP (Det)':<9} | {'IDF1 (Naive)':<12} | {'IDF1 (LSTM)':<11} | {'Delta IDF1':<10} | {'IDSW (N/L)':<10} | {'Diagnóstico'}"
    print(header)
    print("-" * 94)

    results_table = []

    for lvl in levels:
        degraded_dets = degrade_mot17_detections(
            det_by_frame=det_by_frame,
            im_w=im_w,
            im_h=im_h,
            drop_prob=lvl["drop_prob"],
            noise_std=lvl["noise_std"],
            fp_rate=lvl["fp_rate"],
            seed=seed
        )

        # 1. mAP do detector degradado
        map_score = float(compute_map_per_frame(gt_by_frame, degraded_dets, iou_threshold=0.5))

        # 2. Avaliação do Baseline Ingênuo
        naive = NaiveTracker(
            iou_threshold=iou_threshold,
            max_lost_frames=max_lost_frames,
            matching_method="hungarian"
        )
        preds_naive = naive.track_sequence(degraded_dets)
        m_naive = evaluate_tracking(gt_by_frame, preds_naive, iou_threshold=iou_threshold)

        # 3. Avaliação do Modelo Recorrente (Trilha A com correção da Parte 4)
        lstm = RNNMotionTracker(
            model=model_path,
            iou_threshold=iou_threshold,
            max_lost_frames=max_lost_frames,
            velocity_damping=velocity_damping,
            sigma_inflation=sigma_inflation,
            use_adaptive_gating=True
        )
        preds_lstm = lstm.track_sequence(degraded_dets)
        m_lstm = evaluate_tracking(gt_by_frame, preds_lstm, iou_threshold=iou_threshold)

        delta_idf1 = m_lstm["idf1"] - m_naive["idf1"]
        delta_idsw = m_lstm["id_switches"] - m_naive["id_switches"]

        if delta_idf1 >= 0.01:
            diag = "Absorção clara (LSTM amortece a perda)"
        elif delta_idf1 >= 0:
            diag = "Absorção moderada (LSTM estável)"
        else:
            diag = "Amplificação do erro"

        idsw_str = f"{m_naive['id_switches']}/{m_lstm['id_switches']}"
        print(f"{lvl['level']:<10} | {map_score:<9.3f} | {m_naive['idf1']:<12.3f} | {m_lstm['idf1']:<11.3f} | {delta_idf1:+10.3f} | {idsw_str:<10} | {diag}")

        entry = {
            "level": lvl["level"],
            "name": lvl["name"],
            "description": lvl["description"],
            "drop_prob": lvl["drop_prob"],
            "noise_std": lvl["noise_std"],
            "fp_rate": lvl["fp_rate"],
            "map_score": map_score,
            "naive": {
                "idf1": float(m_naive["idf1"]),
                "id_switches": int(m_naive["id_switches"]),
                "fragmentations": int(m_naive["fragmentations"]),
                "ratio_ids": float(m_naive["ratio_ids"])
            },
            "lstm": {
                "idf1": float(m_lstm["idf1"]),
                "id_switches": int(m_lstm["id_switches"]),
                "fragmentations": int(m_lstm["fragmentations"]),
                "ratio_ids": float(m_lstm["ratio_ids"])
            },
            "delta_idf1": float(delta_idf1),
            "delta_idsw": int(delta_idsw),
            "diagnostico": diag
        }
        results_table.append(entry)

    print(SEP)

    # Salva o arquivo de resultados JSON
    if save_results_path:
        os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
        payload = {
            "sequence": seq_name,
            "model_path": model_path,
            "velocity_damping": velocity_damping,
            "sigma_inflation": sigma_inflation,
            "results": results_table
        }
        with open(save_results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Resultados salvos em: {save_results_path}")

    # Gera visualização gráfica se especificado
    if save_plot_path:
        from src.visualization import plot_stress_detector_results
        plot_stress_detector_results(results_table, seq_name=seq_name, save_path=save_plot_path)

    return {"sequence": seq_name, "results": results_table}


if __name__ == "__main__":
    run_detector_stress_experiment()
