"""
src/evaluation.py
Funções de avaliação de alto nível para Parte 0 e Parte 1.
O notebook chama apenas essas funções — toda a lógica de loop,
formatação de tabela e coleta de métricas fica aqui.
"""

import os
import glob

import numpy as np
from PIL import Image

from src.data import (
    generate_synthetic_video,
    generate_controlled_occlusion_sequence,
    degrade_detections,
    load_mot17_sequence,
    compute_sequence_density,
)
from src.tracker import NaiveTracker, RNNMotionTracker
from src.metrics import evaluate_tracking, compute_map_per_frame
from src.inference import get_torchvision_person_detector, infer_torchvision_frame


# ===========================================================================
# PARTE 0 — Gerador sintético e curva de quebra
# ===========================================================================

_DEFAULT_LEVELS = [
    {"label": "Perfeito",  "drop": 0.00, "noise": 0.0,  "fp": 0.00},
    {"label": "Leve",      "drop": 0.10, "noise": 1.5,  "fp": 0.05},
    {"label": "Moderado",  "drop": 0.20, "noise": 3.0,  "fp": 0.10},
    {"label": "Severo",    "drop": 0.35, "noise": 6.0,  "fp": 0.20},
]


def run_synthetic_breakdown(
    gt_by_frame: dict,
    levels: list = None,
    iou_threshold: float = 0.3,
    max_lost_frames: int = 5,
    frame_size: int = 128,
    seed: int = 0,
) -> list:
    """
    Parte 0 (Item 4) — Roda o NaiveTracker em cenários de degradação progressiva
    e retorna lista de dicts com os resultados de cada nível.

    Cada dict contém: label, drop, noise, fp, map_score, idf1, id_switches,
    fragmentations, ratio_ids, switches_per_gt.
    """
    if levels is None:
        levels = _DEFAULT_LEVELS

    SEP = "=" * 75
    print(SEP)
    print("TABELA 0: BASELINE NO CENARIO SINTETICO — Curva de Quebra")
    print(SEP)
    print("{:<12} | {:<10} | {:<8} | {:<6} | {:<10}".format(
        "Degradacao", "mAP (Det)", "IDF1", "IDSW", "Razao IDs"
    ))
    print("-" * 75)

    results = []
    tracker = NaiveTracker(iou_threshold=iou_threshold, max_lost_frames=max_lost_frames)

    for lvl in levels:
        det = degrade_detections(
            gt_by_frame,
            drop_prob=lvl["drop"],
            noise_std=lvl["noise"],
            fp_rate=lvl["fp"],
            frame_size=frame_size,
            seed=seed,
        )
        tracker.reset()
        preds = tracker.track_sequence(det)
        m = evaluate_tracking(gt_by_frame, preds)
        mp = compute_map_per_frame(gt_by_frame, det)

        print("{:<12} | {:<10.3f} | {:<8.3f} | {:<6d} | {:<9.2f}x".format(
            lvl["label"], mp, m["idf1"], m["id_switches"], m["ratio_ids"]
        ))

        results.append({
            **lvl,
            "map_score": mp,
            "idf1": m["idf1"],
            "id_switches": m["id_switches"],
            "fragmentations": m["fragmentations"],
            "ratio_ids": m["ratio_ids"],
            "switches_per_gt": m["switches_per_gt"],
        })

    print(SEP)
    return results


def run_generator_knobs_breakdown(
    tracker: NaiveTracker = None,
    iou_threshold: float = 0.3,
    max_lost_frames: int = 10,
    seed: int = 42,
) -> dict:
    """
    Parte 0 (Item 4) — Gira os botões do gerador para avaliar o baseline no piso fácil
    e revelar onde a associação ingênua começa a quebrar:
    - Botão 1: Velocidade típica (1.0, 2.5, 5.0, 8.0, 12.0 px/frame)
    - Botão 2: Duração da oclusão (2, 5, 10, 16, 22 quadros) com max_lost_frames=10
    - Botão 3: Densidade de objetos (3, 6, 9, 13, 18 objetos em 128x128)
    """
    if tracker is None:
        tracker = NaiveTracker(iou_threshold=iou_threshold, max_lost_frames=max_lost_frames)

    SEP = "=" * 78
    print(SEP)
    print("PARTE 0 (Item 4) — ENSAIO DA PARTE 1: GIRANDO OS BOTOES DO GERADOR")
    print(SEP)

    # 1. BOTAO: VELOCIDADE
    print("\n--- 1. BOTAO: VELOCIDADE TYPICAL (num_objects=5, sem oclusao prolongada) ---")
    print("{:<12} | {:<8} | {:<6} | {:<8} | {:<10}".format(
        "Vel (px/f)", "IDF1", "IDSW", "Frag", "Razao IDs"
    ))
    print("-" * 78)
    vel_results = []
    velocities = [1.0, 2.5, 5.0, 8.0, 12.0]
    for v in velocities:
        frames, gt = generate_synthetic_video(
            num_frames=40, num_objects=5, frame_size=128, typical_velocity=v,
            occlusion_duration=0, seed=seed
        )
        tracker.reset()
        preds = tracker.track_sequence(gt)
        m = evaluate_tracking(gt, preds, iou_threshold=iou_threshold)
        print("{:<12.1f} | {:<8.3f} | {:<6d} | {:<8d} | {:<9.2f}x".format(
            v, m["idf1"], m["id_switches"], m["fragmentations"], m["ratio_ids"]
        ))
        vel_results.append({"velocity": v, **m})

    # 2. BOTAO: DURACAO DA OCLUSAO
    print(f"\n--- 2. BOTAO: DURACAO DA OCLUSAO (k_max_lost={max_lost_frames} quadros) ---")
    print("{:<12} | {:<8} | {:<6} | {:<8} | {:<10}".format(
        "Oclusao (f)", "IDF1", "IDSW", "Frag", "Razao IDs"
    ))
    print("-" * 78)
    occ_results = []
    occlusions = [2, 5, 10, 16, 22]
    for occ in occlusions:
        frames, gt, info = generate_controlled_occlusion_sequence(
            num_frames=45, occlusion_duration=occ, frame_size=128
        )
        tracker.reset()
        preds = tracker.track_sequence(gt)
        m = evaluate_tracking(gt, preds, iou_threshold=iou_threshold)
        print("{:<12d} | {:<8.3f} | {:<6d} | {:<8d} | {:<9.2f}x".format(
            occ, m["idf1"], m["id_switches"], m["fragmentations"], m["ratio_ids"]
        ))
        occ_results.append({"occlusion": occ, **m})

    # 3. BOTAO: NUMERO DE OBJETOS / DENSIDADE
    print("\n--- 3. BOTAO: NUMERO DE OBJETOS / DENSIDADE (vel=2.0 px/frame) ---")
    print("{:<12} | {:<8} | {:<6} | {:<8} | {:<10}".format(
        "Objetos", "IDF1", "IDSW", "Frag", "Razao IDs"
    ))
    print("-" * 78)
    obj_results = []
    n_objs = [3, 6, 9, 13, 18]
    for n in n_objs:
        frames, gt = generate_synthetic_video(
            num_frames=40, num_objects=n, frame_size=128, typical_velocity=2.0,
            occlusion_duration=0, seed=seed
        )
        tracker.reset()
        preds = tracker.track_sequence(gt)
        m = evaluate_tracking(gt, preds, iou_threshold=iou_threshold)
        print("{:<12d} | {:<8.3f} | {:<6d} | {:<8d} | {:<9.2f}x".format(
            n, m["idf1"], m["id_switches"], m["fragmentations"], m["ratio_ids"]
        ))
        obj_results.append({"num_objects": n, **m})

    print(SEP + "\n")
    return {
        "velocity": vel_results,
        "occlusion": occ_results,
        "density": obj_results
    }



# ===========================================================================
# PARTE 1 — Tabelas e avaliação das duas fontes
# ===========================================================================

def run_synthetic_table(
    gt_by_frame: dict,
    tracker: NaiveTracker,
    levels: list = None,
    frame_size: int = 128,
    seed: int = 0,
) -> list:
    """
    Parte 1 (Tabela 1.0) — Valida o tracker ingênuo no cenário sintético com
    degradação progressiva. Retorna lista de dicts com os resultados.
    """
    if levels is None:
        levels = _DEFAULT_LEVELS

    SEP = "=" * 75
    print(SEP)
    print("TABELA 1.0: BASELINE NO CENARIO SINTETICO (Parte 0 -> Parte 1)")
    print(SEP)
    print("{:<12} | {:<10} | {:<8} | {:<6} | {:<10}".format(
        "Degradacao", "mAP (Det)", "IDF1", "IDSW", "Razao IDs"
    ))
    print("-" * 75)

    results = []
    for lvl in levels:
        det = degrade_detections(
            gt_by_frame,
            drop_prob=lvl["drop"],
            noise_std=lvl["noise"],
            fp_rate=lvl["fp"],
            frame_size=frame_size,
            seed=seed,
        )
        tracker.reset()
        preds = tracker.track_sequence(det)
        m = evaluate_tracking(gt_by_frame, preds)
        mp = compute_map_per_frame(gt_by_frame, det)

        print("{:<12} | {:<10.3f} | {:<8.3f} | {:<6d} | {:<9.2f}x".format(
            lvl["label"], mp, m["idf1"], m["id_switches"], m["ratio_ids"]
        ))

        results.append({
            **lvl,
            "map_score": mp,
            "idf1": m["idf1"],
            "id_switches": m["id_switches"],
            "fragmentations": m["fragmentations"],
            "ratio_ids": m["ratio_ids"],
            "switches_per_gt": m["switches_per_gt"],
        })

    print(SEP + "\n")
    return results


def run_detector_comparison(
    data_dir: str,
    tracker: NaiveTracker,
    seq_id: str = "09",
    det_names: list = None,
) -> list:
    """
    Parte 1 (Tabela 1.1) — Compara os três detectores públicos do MOT17
    (DPM, FRCNN, SDP) numa sequência de referência. Imprime a tabela e
    retorna lista de dicts com os resultados.
    """
    if det_names is None:
        det_names = ["DPM", "FRCNN", "SDP"]

    SEP = "=" * 75
    print(SEP)
    print(f"TABELA 1.1: FONTES PUBLICAS NO MOT17-{seq_id} (ESCOLHA DO DETECTOR PADRAO)")
    print(SEP)
    print("{:<8} | {:<10} | {:<8} | {:<12} | {:<9}".format(
        "Fonte", "mAP (Det)", "IDF1", "ID Switches", "Pred IDs"
    ))
    print("-" * 75)

    results = []
    for det_name in det_names:
        seq_p = os.path.join(data_dir, f"MOT17-{seq_id}-{det_name}")
        if not os.path.exists(seq_p):
            alt = os.path.join(os.path.dirname(data_dir), "data", "MOT17", "train",
                               f"MOT17-{seq_id}-{det_name}")
            if os.path.exists(alt):
                seq_p = alt
            else:
                print(f"  [SKIP] {seq_p} nao encontrado")
                continue

        gt, dets, _ = load_mot17_sequence(seq_p)
        tracker.reset()
        preds = tracker.track_sequence(dets)
        m = evaluate_tracking(gt, preds)
        mp = compute_map_per_frame(gt, dets)

        if det_name == "SDP":
            status = "★ Padrao Escolhido (Maior mAP)"
        elif det_name == "DPM":
            status = "Ruidoso / Obsoleto"
        else:
            status = "Conservador"

        print("{:<8} | {:<10.3f} | {:<8.3f} | {:<12d} | {:<9d}".format(
            det_name, mp, m["idf1"], m["id_switches"], m["unique_pred_ids"]
        ))

        results.append({"det_name": det_name, "map_score": mp, **m})

    return results


def run_mot17_baseline(
    data_dir: str,
    tracker: NaiveTracker,
    det_suffix: str = "SDP",
    iou_threshold: float = 0.5,
) -> list:
    """
    Parte 1 (Tabela 1.2) — Roda o NaiveTracker em todas as sequências MOT17
    com o detector padrão e retorna lista de dicts com os resultados.
    """
    sequences = sorted(glob.glob(os.path.join(data_dir, f"MOT17-*-{det_suffix}")))

    SEP = "=" * 88
    print(SEP)
    print(f"TABELA 1.2: BASELINE INGENUO COM DETECTOR PADRAO ({det_suffix}) NAS SEQUENCIAS DO MOT17")
    print(SEP)
    print("{:<10} | {:<9} | {:<9} | {:<7} | {:<6} | {:<6} | {:<9} | {}".format(
        "Seq", "Densidade", "mAP (Det)", "IDF1", "IDSW", "Frag", "Razao IDs", "Switches/GT"
    ))
    print("-" * 88)

    results = []
    for seq in sequences:
        seq_name = os.path.basename(seq).replace(f"-{det_suffix}", "")
        gt_by_frame, det_by_frame, _ = load_mot17_sequence(seq)
        tracker.reset()
        preds = tracker.track_sequence(det_by_frame)
        metrics = evaluate_tracking(gt_by_frame, preds, iou_threshold=iou_threshold)
        map_score = compute_map_per_frame(gt_by_frame, det_by_frame, iou_threshold=iou_threshold)
        density = compute_sequence_density(gt_by_frame)

        res = {
            "seq_name": seq_name,
            "density": density,
            "map_score": map_score,
            "idf1": metrics["idf1"],
            "id_switches": metrics["id_switches"],
            "fragmentations": metrics["fragmentations"],
            "ratio_ids": metrics["ratio_ids"],
            "switches_per_gt": metrics["switches_per_gt"],
        }
        results.append(res)

        print("{:<10} | {:<9.1f} | {:<9.3f} | {:<7.3f} | {:<6d} | {:<6d} | {:<8.2f}x | {:.2f}".format(
            res["seq_name"], res["density"], res["map_score"], res["idf1"],
            res["id_switches"], res["fragmentations"], res["ratio_ids"], res["switches_per_gt"]
        ))

    print(SEP)
    return results


def compare_detection_sources(
    frames: np.ndarray,
    gt_by_frame: dict,
    device: str = "cpu",
    num_frames: int = None,
    min_score: float = 0.3,
    noise_std: float = 0.5,
) -> dict:
    """
    Parte 1 (Item 1) — Compara as duas fontes de detecção no domínio sintético:

    - Fonte 1: det.txt público simulado via degrade_detections (ruído mínimo)
    - Fonte 2: Faster R-CNN ResNet-50 FPN pré-treinado no COCO (torchvision)

    Retorna dict com:
      frames, gt_by_frame, det_fonte1, det_fonte2,
      n_fonte1, n_fonte2
    """
    n = num_frames if num_frames is not None else len(frames)

    # Fonte 1
    det_fonte1 = degrade_detections(gt_by_frame, drop_prob=0.0,
                                     noise_std=noise_std, fp_rate=0.0, seed=0)

    # Fonte 2 — Faster R-CNN
    print("Carregando Faster R-CNN ResNet-50 FPN (COCO)...")
    model = get_torchvision_person_detector(device=device)

    det_fonte2 = {}
    for i in range(n):
        frame_id = i + 1
        pil_img = Image.fromarray(frames[i])
        dets = infer_torchvision_frame(model, pil_img, min_score=min_score, device=device)
        det_fonte2[frame_id] = dets if dets else []

    n1 = sum(len(v) for v in det_fonte1.values())
    n2 = sum(len(v) for v in det_fonte2.values())

    SEP = "=" * 65
    print("\n" + SEP)
    print("COMPARACAO DAS DUAS FONTES NO DOMINIO SINTETICO")
    print(SEP)
    print("{:<40} | {:<12}".format("Fonte", "Deteccoes"))
    print("-" * 65)
    print("{:<40} | {:<12}".format(
        "Fonte 1 — Publicas/Simul. (SDP-like)", n1))
    print("{:<40} | {:<12}".format(
        "Fonte 2 — Torchvision Faster R-CNN", n2))

    return {
        "frames": frames,
        "gt_by_frame": gt_by_frame,
        "det_fonte1": det_fonte1,
        "det_fonte2": det_fonte2,
        "n_fonte1": n1,
        "n_fonte2": n2,
    }


def compare_real_detection_sources(
    seq_path: str,
    num_frames: int = 50,
    device: str = "cuda",
    min_score: float = 0.5,
) -> dict:
    """
    Parte 1 (Item 1) — Avalia as duas fontes de detecção em quadros REAIS do MOT17:
    - Fonte 1: Detecções públicas SDP (det/det.txt)
    - Fonte 2: Detector Faster R-CNN ResNet-50 FPN pré-treinado no COCO com custom_nms.

    Compara mAP, IDF1 e ID switches com NaiveTracker nos mesmos quadros.
    """
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    gt_full, sdp_full, seq_info = load_mot17_sequence(seq_path)
    img_dir = os.path.join(seq_path, "img1")

    # Limita aos primeiros num_frames
    target_frames = list(range(1, num_frames + 1))
    gt_sub = {f: gt_full.get(f, {}) for f in target_frames}
    sdp_sub = {f: sdp_full.get(f, []) for f in target_frames}

    print(f"Carregando Faster R-CNN ResNet-50 FPN no dispositivo {device}...")
    model = get_torchvision_person_detector(device=device)

    print(f"Executando inferência com custom_nms em {num_frames} quadros de {os.path.basename(seq_path)}...")
    rcnn_dets = {}
    sample_images = {}

    for f in target_frames:
        img_name = f"{f:06d}.jpg"
        img_p = os.path.join(img_dir, img_name)
        if not os.path.exists(img_p):
            continue
        pil_img = Image.open(img_p).convert("RGB")
        dets = infer_torchvision_frame(model, pil_img, min_score=min_score, device=device)
        rcnn_dets[f] = dets
        if f in [1, 10, 25]:
            sample_images[f] = pil_img

    # Avaliação de mAP
    map_sdp = compute_map_per_frame(gt_sub, sdp_sub)
    map_rcnn = compute_map_per_frame(gt_sub, rcnn_dets)

    # Avaliação de Tracking ingênuo
    tracker = NaiveTracker(iou_threshold=0.3, max_lost_frames=15)
    
    tracker.reset()
    pred_sdp = tracker.track_sequence(sdp_sub)
    m_sdp = evaluate_tracking(gt_sub, pred_sdp)

    tracker.reset()
    pred_rcnn = tracker.track_sequence(rcnn_dets)
    m_rcnn = evaluate_tracking(gt_sub, pred_rcnn)

    SEP = "=" * 80
    print("\n" + SEP)
    print(f"TABELA 1.1b: COMPARACAO DAS DUAS FONTES EM DADOS REAIS ({os.path.basename(seq_path)})")
    print(SEP)
    print("{:<28} | {:<9} | {:<8} | {:<6} | {:<10}".format(
        "Fonte de Detecção", "mAP (Det)", "IDF1", "IDSW", "Razao IDs"
    ))
    print("-" * 80)
    print("{:<28} | {:<9.3f} | {:<8.3f} | {:<6d} | {:<9.2f}x".format(
        "Fonte 1 — Publica (SDP)", map_sdp, m_sdp["idf1"], m_sdp["id_switches"],
        m_sdp["ratio_ids"]
    ))
    print("{:<28} | {:<9.3f} | {:<8.3f} | {:<6d} | {:<9.2f}x".format(
        "Fonte 2 — Faster R-CNN (COCO)", map_rcnn, m_rcnn["idf1"], m_rcnn["id_switches"],
        m_rcnn["ratio_ids"]
    ))

    return {
        "seq_name": os.path.basename(seq_path),
        "gt": gt_sub,
        "sdp_dets": sdp_sub,
        "rcnn_dets": rcnn_dets,
        "map_sdp": map_sdp,
        "map_rcnn": map_rcnn,
        "m_sdp": m_sdp,
        "m_rcnn": m_rcnn,
        "sample_images": sample_images
    }


def run_part2_comparison(
    data_dir: str,
    tracker_naive: NaiveTracker,
    tracker_rnn: RNNMotionTracker,
    seq_names: list = None,
    det_suffix: str = "SDP"
) -> list:
    """
    Parte 2 — Comparação Lado a Lado: Baseline Ingênuo (Parte 1) vs Trilha A (Modelo de Movimento LSTM).
    Avalia nas sequências de validação (ou selecionadas) com as mesmas detecções congeladas.
    """
    if seq_names is None:
        seq_names = ["09", "11", "05"]

    SEP = "=" * 90
    print(SEP)
    print("PARTE 2 (TRILHA A): COMPARAÇÃO LADO A LADO — BASELINE INGÊNUO vs RNN MOTION MODEL")
    print(SEP)
    print("{:<10} | {:<18} | {:<8} | {:<8} | {:<8} | {}".format(
        "Sequência", "Modelo", "IDF1", "IDSW", "Frag", "Razão IDs"
    ))
    print("-" * 90)

    comparison_results = []

    for s_name in seq_names:
        seq_p = os.path.join(data_dir, f"MOT17-{s_name}-{det_suffix}")
        if not os.path.exists(seq_p):
            continue

        gt, dets, info = load_mot17_sequence(seq_p)
        w = float(info.get("imWidth", 1920))
        h = float(info.get("imHeight", 1080))
        density = compute_sequence_density(gt)

        # 1. Baseline Naive
        tracker_naive.reset()
        preds_n = tracker_naive.track_sequence(dets)
        mn = evaluate_tracking(gt, preds_n)

        # 2. Trilha A (RNN Motion)
        tracker_rnn.reset()
        preds_r = tracker_rnn.track_sequence(dets, im_width=w, im_height=h)
        mr = evaluate_tracking(gt, preds_r)

        delta_idf1 = mr["idf1"] - mn["idf1"]
        delta_sw = mr["id_switches"] - mn["id_switches"]

        print("{:<10} | {:<18} | {:<8.3f} | {:<8d} | {:<8d} | {:.2f}x".format(
            f"MOT17-{s_name}", "Baseline (Naive)", mn["idf1"], mn["id_switches"], mn["fragmentations"], mn["ratio_ids"]
        ))
        print("{:<10} | {:<18} | {:<8.3f} | {:<8d} | {:<8d} | {:.2f}x (Delta IDF1: {:+.3f}, Delta IDSW: {:+d})".format(
            "", "Trilha A (LSTM)", mr["idf1"], mr["id_switches"], mr["fragmentations"], mr["ratio_ids"], delta_idf1, delta_sw
        ))
        print("-" * 90)

        comparison_results.append({
            "seq_name": f"MOT17-{s_name}",
            "density": density,
            "naive": mn,
            "rnn": mr,
            "delta_idf1": delta_idf1,
            "delta_switches": delta_sw
        })

    print(SEP + "\n")
    return comparison_results


