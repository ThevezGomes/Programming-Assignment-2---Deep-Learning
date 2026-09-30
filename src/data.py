"""
src/data.py
Carregamento de dados do MOT17, divisão por sequências e utilitários.
"""

import os
import configparser
import numpy as np


def parse_seqinfo(seq_path: str):
    """Lê o arquivo seqinfo.ini da sequência MOT17."""
    ini_path = os.path.join(seq_path, "seqinfo.ini")
    info = {}
    if os.path.exists(ini_path):
        config = configparser.ConfigParser()
        config.read(ini_path)
        if "Sequence" in config:
            s = config["Sequence"]
            info["name"] = s.get("name", os.path.basename(seq_path))
            info["imDir"] = s.get("imDir", "img1")
            info["frameRate"] = float(s.get("frameRate", 30))
            info["seqLength"] = int(s.get("seqLength", 0))
            info["imWidth"] = int(s.get("imWidth", 1920))
            info["imHeight"] = int(s.get("imHeight", 1080))
            info["imExt"] = s.get("imExt", ".jpg")
    return info


def load_mot17_sequence(seq_path: str, min_det_conf: float = 0.0, filter_pedestrians: bool = True):
    """
    Carrega o ground truth (gt.txt) e as detecções públicas (det.txt) de uma sequência do MOT17.
    
    gt.txt: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    det.txt: frame, -1, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
    
    Retorna:
    - gt_by_frame: {frame: {id: [x, y, w, h]}}
    - det_by_frame: {frame: [[x, y, w, h, conf], ...]}
    - seq_info: dict com metadados da sequência
    """
    seq_info = parse_seqinfo(seq_path)

    # 1. Carrega Ground Truth
    gt_file = os.path.join(seq_path, "gt", "gt.txt")
    gt_by_frame = {}
    if os.path.exists(gt_file):
        with open(gt_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame = int(parts[0])
                tid = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                cls = int(parts[7]) if len(parts) > 7 else 1
                vis = float(parts[8]) if len(parts) > 8 else 1.0

                # Filtrar apenas pedestres válidos (classe 1 = pedestrian no MOT17)
                if filter_pedestrians:
                    if cls != 1 or vis <= 0.0:
                        continue

                if frame not in gt_by_frame:
                    gt_by_frame[frame] = {}
                gt_by_frame[frame][tid] = np.array([x, y, w, h], dtype=np.float32)

    # 2. Carrega Detecções Públicas (det.txt)
    det_file = os.path.join(seq_path, "det", "det.txt")
    det_by_frame = {}
    if os.path.exists(det_file):
        with open(det_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame = int(parts[0])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0

                if conf < min_det_conf:
                    continue

                if frame not in det_by_frame:
                    det_by_frame[frame] = []
                det_by_frame[frame].append([x, y, w, h, conf])

    return gt_by_frame, det_by_frame, seq_info


def get_mot17_splits(base_dir: str = "data/MOT17/train", det_suffix: str = "SDP"):
    """
    Define a divisão rigorosa por sequência inteira (conforme regra do enunciado).
    
    Justificativas para a apresentação:
    - MOT17-09: Câmera estática, visão frontal de rua, densidade moderada (excelente para validação pura).
    - MOT17-11: Câmera móvel, ambiente interno, visão em perspectiva (excelente para testar generalização com movimento de câmera).
    - Treino: sequências restantes (MOT17-02, 04, 05, 10, 13) abrangendo diferentes densidades e iluminações.
    """
    all_seq_names = ["02", "04", "05", "09", "10", "11", "13"]
    val_names = ["09", "11"]
    train_names = ["02", "04", "05", "10", "13"]

    def _build_paths(names):
        paths = []
        for n in names:
            p = os.path.join(base_dir, f"MOT17-{n}-{det_suffix}")
            if os.path.exists(p):
                paths.append(p)
        return paths

    return {
        "train": _build_paths(train_names),
        "val": _build_paths(val_names)
    }


def compute_sequence_density(gt_by_frame):
    """Calcula a densidade média de pedestres por quadro."""
    if not gt_by_frame:
        return 0.0
    return float(np.mean([len(boxes) for boxes in gt_by_frame.values()]))
