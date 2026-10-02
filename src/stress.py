"""
src/stress.py
Parte 5 — Teste de Estresse: Degradação da Qualidade do Detector.

Avalia o comportamento do rastreador sob perturbações controladas nas detecções
públicas do MOT17 (descarte p%, ruído nas caixas e injeção de falsos positivos)
em 4 patamares (Controle, Leve, Moderada, Severa) com 3 sementes aleatórias por nível,
avaliado na sequência MOT17-09 e também na MOT17-11.

Inclui controle justo: NaiveTracker com a mesma relaxação adaptativa de limiar,
separando o efeito da recorrência do efeito do limiar relaxado.
Métricas avaliadas no protocolo unificado EVAL_IOU = 0.5.
"""

import os
import json
import numpy as np

from src.config import EVAL_IOU, DEFAULT_TRACKER_IOU, DEFAULT_MAX_LOST_FRAMES, FINAL_CHECKPOINT_PATH
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
                x += rng.normal(0, noise_std)
                y += rng.normal(0, noise_std)
                w += rng.normal(0, noise_std * 0.5)
                h += rng.normal(0, noise_std * 0.5)

            x = float(np.clip(x, 0, im_w - 2))
            y = float(np.clip(y, 0, im_h - 2))
            w = float(np.clip(w, 4, im_w - x))
            h = float(np.clip(h, 8, im_h - y))

            new_dets.append([x, y, w, h, conf])

        # 3: Injeção de falsos positivos aleatórios
        if fp_rate > 0.0 and len(orig_dets) > 0:
            n_fp = rng.poisson(fp_rate * len(orig_dets))
            for _ in range(int(n_fp)):
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
    seq_path: str or list = "data/MOT17/train/MOT17-09-SDP",
    model_path: str = FINAL_CHECKPOINT_PATH,
    levels: list = None,
    seeds: list = None,
    velocity_damping: float = 0.85,
    sigma_inflation: float = 0.30,
    iou_threshold: float = DEFAULT_TRACKER_IOU,
    max_lost_frames: int = DEFAULT_MAX_LOST_FRAMES,
    save_results_path: str = "results/stress_detector.json",
    save_plot_path: str = "docs/parte5_stress_detector.png"
) -> dict:
    """
    Executa o teste de estresse da Parte 5 com múltiplas sementes e controle justo:
    - Compara Naive Tracker (padrão), Naive Tracker Relaxado (controle de limiar) e LSTM.
    - Avalia na sequência fornecida (ou lista de sequências MOT17-09 e MOT17-11).
    - Métricas avaliadas com EVAL_IOU = 0.5.
    - Retorna médias ± desvios, retenção relativa e veredito derivado dos números.
    """
    if levels is None:
        levels = DEFAULT_STRESS_LEVELS
    if seeds is None:
        seeds = [42, 100, 2026]

    if isinstance(seq_path, str):
        target_seqs = [seq_path]
    else:
        target_seqs = list(seq_path)

    all_sequences_results = {}

    for s_path in target_seqs:
        if not os.path.exists(s_path):
            alt = os.path.join(os.path.dirname(s_path), "data", "MOT17", "train", os.path.basename(s_path))
            if os.path.exists(alt):
                s_path = alt
            else:
                print(f"[SKIP] Sequência não encontrada: {s_path}")
                continue

        gt_by_frame, det_by_frame, info = load_mot17_sequence(s_path)
        im_w = int(info.get("imWidth", 1920))
        im_h = int(info.get("imHeight", 1080))
        seq_name = info.get("name", os.path.basename(s_path))

        SEP = "=" * 104
        print("\n" + SEP)
        print(f"PARTE 5: TESTE DE ESTRESSE DO DETECTOR — {seq_name} (3 seeds por nível)")
        print(f"Protocolo de Avaliação: EVAL_IOU = {EVAL_IOU} | Checkpoint: {os.path.basename(model_path)}")
        print(SEP)
        print("{:<10} | {:<12} | {:<12} | {:<13} | {:<12} | {:<11} | {}".format(
            "Nível", "mAP(0.5)", "IDF1 Naive", "IDF1 NaiveRel", "IDF1 LSTM", "Delta IDF1", "Veredito Derivado"
        ))
        print("-" * 104)

        results_table = []
        control_lstm_idf1 = None

        for lvl in levels:
            lvl_seeds = [seeds[0]] if (lvl["drop_prob"] == 0 and lvl["noise_std"] == 0 and lvl["fp_rate"] == 0) else seeds

            map_list = []
            naive_idf1_list = []
            naive_rel_idf1_list = []
            lstm_idf1_list = []
            naive_idsw_list = []
            lstm_idsw_list = []
            delta_idf1_list = []

            for s in lvl_seeds:
                degraded_dets = degrade_mot17_detections(
                    det_by_frame=det_by_frame,
                    im_w=im_w,
                    im_h=im_h,
                    drop_prob=lvl["drop_prob"],
                    noise_std=lvl["noise_std"],
                    fp_rate=lvl["fp_rate"],
                    seed=s
                )

                mp = compute_map_per_frame(gt_by_frame, degraded_dets, iou_threshold=EVAL_IOU)
                map_list.append(mp)

                # 1. Baseline Naive Padrão
                naive = NaiveTracker(iou_threshold=iou_threshold, max_lost_frames=max_lost_frames)
                preds_naive = naive.track_sequence(degraded_dets)
                m_naive = evaluate_tracking(gt_by_frame, preds_naive, iou_threshold=EVAL_IOU)
                naive_idf1_list.append(m_naive["idf1"])
                naive_idsw_list.append(m_naive["id_switches"])

                # 2. Controle Justo: Naive com a mesma relaxação de limiar
                naive_rel = NaiveTracker(
                    iou_threshold=iou_threshold,
                    max_lost_frames=max_lost_frames,
                    adaptive_decay=0.01
                )
                preds_naive_rel = naive_rel.track_sequence(degraded_dets)
                m_naive_rel = evaluate_tracking(gt_by_frame, preds_naive_rel, iou_threshold=EVAL_IOU)
                naive_rel_idf1_list.append(m_naive_rel["idf1"])

                # 3. Trilha A (LSTM com portão adaptativo e incerteza)
                lstm = RNNMotionTracker(
                    model=model_path,
                    iou_threshold=iou_threshold,
                    max_lost_frames=max_lost_frames,
                    velocity_damping=velocity_damping,
                    sigma_inflation=sigma_inflation,
                    use_adaptive_gating=True
                )
                preds_lstm = lstm.track_sequence(degraded_dets, im_width=im_w, im_height=im_h)
                m_lstm = evaluate_tracking(gt_by_frame, preds_lstm, iou_threshold=EVAL_IOU)
                lstm_idf1_list.append(m_lstm["idf1"])
                lstm_idsw_list.append(m_lstm["id_switches"])

                delta_idf1_list.append(m_lstm["idf1"] - m_naive["idf1"])

            mean_map = float(np.mean(map_list))
            std_map = float(np.std(map_list))
            mean_naive = float(np.mean(naive_idf1_list))
            std_naive = float(np.std(naive_idf1_list))
            mean_naive_rel = float(np.mean(naive_rel_idf1_list))
            std_naive_rel = float(np.std(naive_rel_idf1_list))
            mean_lstm = float(np.mean(lstm_idf1_list))
            std_lstm = float(np.std(lstm_idf1_list))
            mean_delta = float(np.mean(delta_idf1_list))
            std_delta = float(np.std(delta_idf1_list))

            mean_idsw_naive = float(np.mean(naive_idsw_list))
            mean_idsw_lstm = float(np.mean(lstm_idsw_list))

            if lvl["level"] == "Controle":
                control_lstm_idf1 = mean_lstm
                retention = 1.0
                abs_loss = 0.0
            else:
                retention = (mean_lstm / control_lstm_idf1) if control_lstm_idf1 else 0.0
                abs_loss = (control_lstm_idf1 - mean_lstm) if control_lstm_idf1 else 0.0

            # Veredito derivado matematicamente dos números medidos
            if mean_delta > std_delta and mean_delta >= 0.008:
                veredito = "Absorve (ganho além da variância)"
            elif abs(mean_delta) <= (std_delta + 1e-4):
                veredito = "Neutro (dentro da variância das seeds)"
            else:
                veredito = "Amplifica erro"

            map_str = f"{mean_map:.3f}" if std_map < 1e-3 else f"{mean_map:.3f}±{std_map:.3f}"
            naive_str = f"{mean_naive:.3f}" if std_naive < 1e-3 else f"{mean_naive:.3f}±{std_naive:.3f}"
            naive_rel_str = f"{mean_naive_rel:.3f}" if std_naive_rel < 1e-3 else f"{mean_naive_rel:.3f}±{std_naive_rel:.3f}"
            lstm_str = f"{mean_lstm:.3f}" if std_lstm < 1e-3 else f"{mean_lstm:.3f}±{std_lstm:.3f}"
            delta_str = f"{mean_delta:+.3f}" if std_delta < 1e-3 else f"{mean_delta:+.3f}±{std_delta:.3f}"

            print("{:<10} | {:<12} | {:<12} | {:<13} | {:<12} | {:<11} | {}".format(
                lvl["level"], map_str, naive_str, naive_rel_str, lstm_str, delta_str, veredito
            ))

            results_table.append({
                "level": lvl["level"],
                "name": lvl["name"],
                "description": lvl["description"],
                "drop_prob": lvl["drop_prob"],
                "noise_std": lvl["noise_std"],
                "fp_rate": lvl["fp_rate"],
                "map_mean": mean_map,
                "map_std": std_map,
                "map_score": mean_map,
                "naive": {
                    "idf1": mean_naive,
                    "idf1_std": std_naive,
                    "id_switches": int(round(mean_idsw_naive)),
                },
                "naive_relaxed": {
                    "idf1": mean_naive_rel,
                    "idf1_std": std_naive_rel,
                },
                "lstm": {
                    "idf1": mean_lstm,
                    "idf1_std": std_lstm,
                    "id_switches": int(round(mean_idsw_lstm)),
                },
                "delta_idf1": mean_delta,
                "delta_idf1_std": std_delta,
                "retention_relative": retention,
                "absolute_loss": abs_loss,
                "veredito": veredito,
                "diagnostico": veredito
            })

        print(SEP)
        all_sequences_results[seq_name] = results_table

    # Se apenas uma sequência foi executada, mantém formato de retorno plano para retrocompatibilidade
    primary_results = list(all_sequences_results.values())[0] if all_sequences_results else []
    primary_seq = list(all_sequences_results.keys())[0] if all_sequences_results else ""

    if save_results_path:
        os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
        payload = {
            "model_path": model_path,
            "velocity_damping": velocity_damping,
            "sigma_inflation": sigma_inflation,
            "eval_iou": EVAL_IOU,
            "sequences": all_sequences_results,
            "results": primary_results
        }
        with open(save_results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Resultados do teste de estresse salvos em: {save_results_path}")

    if save_plot_path and primary_results:
        from src.visualization import plot_stress_detector_results
        plot_stress_detector_results(primary_results, seq_name=primary_seq, save_path=save_plot_path)

    return {
        "sequence": primary_seq,
        "results": primary_results,
        "all_sequences": all_sequences_results
    }


if __name__ == "__main__":
    run_detector_stress_experiment()
