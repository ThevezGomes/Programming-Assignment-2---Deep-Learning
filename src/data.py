"""
src/data.py
Carregamento e geração de dados:
- Parte 0: Gerador de vídeos sintéticos (128x128, elipses com z-order para oclusão real)
- Parte 0 & 5: Simulador de detector (descarte p%, ruído gaussiano, falsos positivos)
- Parte 1 & 2: Parser do MOT17 (gt.txt, det.txt) e splits por sequência
"""

import os
import configparser
import numpy as np


# ==============================================================================
# PARTE 0 — GERADOR DE VÍDEOS SINTÉTICOS
# ==============================================================================

def generate_synthetic_video(
    num_frames: int = 45,
    num_objects: int = 8,
    frame_size: int = 128,
    typical_velocity: float = 3.0,
    occlusion_duration: int = 10,
    noise_level: float = 0.05,
    seed: int = 42
):
    """
    Gera um vídeo sintético 128x128 com elipses em movimento e z-order para oclusão real.

    Parâmetros
    ----------
    num_frames        : número de quadros do vídeo (30 a 60)
    num_objects       : número de objetos elipse (5 a 15)
    frame_size        : lado do frame quadrado em pixels
    typical_velocity  : velocidade típica em pixels/frame
    occlusion_duration: duração média de oclusão forçada entre dois objetos
    noise_level       : desvio padrão do ruído nos tamanhos das elipses (fraction)
    seed              : semente aleatória para reprodutibilidade

    Retorna
    -------
    frames     : np.ndarray (num_frames, frame_size, frame_size, 3) uint8 RGB
    gt_by_frame: dict {frame_id: {object_id: np.array([x, y, w, h])}}
                 x,y = canto superior esquerdo da bounding box, w,h = largura e altura
    """
    rng = np.random.default_rng(seed)

    # --- Inicializa os objetos ---
    semi_a = rng.integers(6, 14, size=num_objects).astype(float)   # semi-eixo horizontal
    semi_b = rng.integers(6, 14, size=num_objects).astype(float)   # semi-eixo vertical
    cx = rng.uniform(semi_a + 2, frame_size - semi_a - 2)           # centro X
    cy = rng.uniform(semi_b + 2, frame_size - semi_b - 2)           # centro Y
    angle = rng.uniform(0, 2 * np.pi, size=num_objects)             # ângulo de rotação (estético)

    # Velocidades (pixels/frame), com mudança de direção suave nas bordas
    vx = rng.uniform(-typical_velocity, typical_velocity, size=num_objects)
    vy = rng.uniform(-typical_velocity, typical_velocity, size=num_objects)
    # Garante que nenhum inicie com velocidade zero
    vx = np.where(np.abs(vx) < 0.5, 0.5 * np.sign(vx + 1e-6), vx)
    vy = np.where(np.abs(vy) < 0.5, 0.5 * np.sign(vy + 1e-6), vy)

    # Z-order fixo para toda a sequência: índice menor = mais fundo (atrás)
    z_order = np.arange(num_objects)  # objeto 0 fica atrás de todos os outros

    # Cores distintas por objeto (HSV espaçado uniformemente -> RGB)
    colors = []
    for i in range(num_objects):
        hue = i / num_objects
        # Conversão HSV simplificada
        h = hue * 6
        x_c = 1 - abs(h % 2 - 1)
        r, g, b = [(1, x_c, 0), (x_c, 1, 0), (0, 1, x_c),
                   (0, x_c, 1), (x_c, 0, 1), (1, 0, x_c)][int(h) % 6]
        colors.append((int(r * 200) + 55, int(g * 200) + 55, int(b * 200) + 55))

    frames = []
    gt_by_frame = {}

    # Pré-calcular grid de coordenadas do frame
    yy, xx = np.mgrid[0:frame_size, 0:frame_size]

    for f in range(num_frames):
        frame = np.full((frame_size, frame_size, 3), 30, dtype=np.uint8)  # Fundo escuro

        # Mapa de z-buffer: qual objeto está visível em cada pixel
        z_buffer = -np.ones((frame_size, frame_size), dtype=int)

        # Desenha os objetos em ordem de z (do mais fundo para o mais próximo)
        for z in range(num_objects):
            obj_idx = np.argsort(z_order)[z]

            # Equação da elipse girada
            cos_a = np.cos(angle[obj_idx])
            sin_a = np.sin(angle[obj_idx])
            dx = xx - cx[obj_idx]
            dy = yy - cy[obj_idx]
            x_rot = dx * cos_a + dy * sin_a
            y_rot = -dx * sin_a + dy * cos_a

            # Máscara da elipse
            mask = ((x_rot / semi_a[obj_idx]) ** 2 + (y_rot / semi_b[obj_idx]) ** 2) <= 1.0

            # Desenha no frame com z-buffer (objeto mais próximo sobrescreve)
            frame[mask] = colors[obj_idx]
            z_buffer[mask] = obj_idx

        frames.append(frame)

        # Ground Truth: bounding box axial de cada objeto
        # Apenas registra objetos que têm pelo menos 1 pixel visível (não totalmente ocluídos)
        gt_by_frame[f + 1] = {}
        for obj_idx in range(num_objects):
            # Pixels do objeto que são realmente visíveis (top no z-buffer)
            visible = (z_buffer == obj_idx)
            visible_frac = visible.sum() / np.pi / semi_a[obj_idx] / semi_b[obj_idx]

            # Inclui no GT mesmo que parcialmente ocluído (visibility >= 0.2)
            if visible_frac >= 0.1:
                # Bounding box da ELIPSE COMPLETA (não só parte visível — padrão MOT)
                cos_a = np.cos(angle[obj_idx])
                sin_a = np.sin(angle[obj_idx])
                # Bounding box da elipse girada (fórmula analítica)
                hw = np.sqrt((semi_a[obj_idx] * cos_a) ** 2 + (semi_b[obj_idx] * sin_a) ** 2)
                hh = np.sqrt((semi_a[obj_idx] * sin_a) ** 2 + (semi_b[obj_idx] * cos_a) ** 2)
                x_box = float(np.clip(cx[obj_idx] - hw, 0, frame_size))
                y_box = float(np.clip(cy[obj_idx] - hh, 0, frame_size))
                w_box = float(np.clip(2 * hw, 1, frame_size - x_box))
                h_box = float(np.clip(2 * hh, 1, frame_size - y_box))
                gt_by_frame[f + 1][obj_idx + 1] = np.array([x_box, y_box, w_box, h_box], dtype=np.float32)

        # Atualiza posições
        cx += vx
        cy += vy

        # Reflexão suave nas bordas
        for i in range(num_objects):
            if cx[i] - semi_a[i] < 0:
                cx[i] = semi_a[i]
                vx[i] = abs(vx[i])
            elif cx[i] + semi_a[i] > frame_size:
                cx[i] = frame_size - semi_a[i]
                vx[i] = -abs(vx[i])
            if cy[i] - semi_b[i] < 0:
                cy[i] = semi_b[i]
                vy[i] = abs(vy[i])
            elif cy[i] + semi_b[i] > frame_size:
                cy[i] = frame_size - semi_b[i]
                vy[i] = -abs(vy[i])

        # Perturbação leve de velocidade (movimento mais orgânico)
        vx += rng.uniform(-0.2, 0.2, size=num_objects)
        vy += rng.uniform(-0.2, 0.2, size=num_objects)
        vx = np.clip(vx, -typical_velocity * 1.5, typical_velocity * 1.5)
        vy = np.clip(vy, -typical_velocity * 1.5, typical_velocity * 1.5)

    frames = np.stack(frames, axis=0)
    return frames, gt_by_frame


# ==============================================================================
# PARTE 0 & 5 — SIMULADOR DE DETECTOR
# ==============================================================================

def degrade_detections(gt_by_frame: dict, drop_prob: float = 0.1,
                        noise_std: float = 2.0, fp_rate: float = 0.05,
                        frame_size: int = 128, seed: int = 0):
    """
    Simula um detector imperfeito a partir do ground truth:
    - Descarta drop_prob% das detecções (falsos negativos)
    - Adiciona ruído gaussiano nas coordenadas [x, y, w, h]
    - Injeta falsos positivos aleatórios (caixas aleatórias)

    Retorna det_by_frame: dict {frame: [[x, y, w, h, conf], ...]}
    """
    rng = np.random.default_rng(seed)
    det_by_frame = {}

    for frame_id, gt_boxes in gt_by_frame.items():
        dets = []
        for obj_id, box in gt_boxes.items():
            # 1. Descarte (falso negativo)
            if rng.random() < drop_prob:
                continue

            # 2. Ruído gaussiano nas coordenadas
            noisy_box = box.copy().astype(np.float32)
            noisy_box[:2] += rng.normal(0, noise_std, size=2)  # x, y
            noisy_box[2:] += rng.normal(0, noise_std * 0.5, size=2)  # w, h (menos ruidoso)

            # Clip para não sair do frame
            noisy_box[0] = np.clip(noisy_box[0], 0, frame_size - 1)
            noisy_box[1] = np.clip(noisy_box[1], 0, frame_size - 1)
            noisy_box[2] = np.clip(noisy_box[2], 2, frame_size - noisy_box[0])
            noisy_box[3] = np.clip(noisy_box[3], 2, frame_size - noisy_box[1])

            # Score de confiança próximo de 1 (detector "quase certo")
            conf = float(np.clip(rng.normal(0.85, 0.1), 0.5, 1.0))
            dets.append([*noisy_box.tolist(), conf])

        # 3. Falsos positivos aleatórios
        n_fp = rng.poisson(fp_rate * len(gt_boxes) + 0.1)
        for _ in range(int(n_fp)):
            x = float(rng.uniform(0, frame_size - 10))
            y = float(rng.uniform(0, frame_size - 10))
            w = float(rng.uniform(5, min(30, frame_size - x)))
            h = float(rng.uniform(5, min(30, frame_size - y)))
            conf = float(rng.uniform(0.3, 0.6))
            dets.append([x, y, w, h, conf])

        det_by_frame[frame_id] = dets

    return det_by_frame


# ==============================================================================
# PARTE 1 & 2 — PARSER MOT17
# ==============================================================================

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


def load_mot17_sequence(seq_path: str, min_det_conf: float = 0.0,
                         filter_pedestrians: bool = True):
    """
    Carrega gt.txt e det.txt de uma sequência do MOT17.

    gt.txt:  frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    det.txt: frame, -1, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1

    Retorna
    -------
    gt_by_frame : {frame: {id: [x, y, w, h]}}
    det_by_frame: {frame: [[x, y, w, h, conf], ...]}
    seq_info    : dict com metadados da sequência
    """
    seq_info = parse_seqinfo(seq_path)

    # Ground Truth
    gt_by_frame = {}
    gt_file = os.path.join(seq_path, "gt", "gt.txt")
    if os.path.exists(gt_file):
        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame = int(parts[0])
                tid   = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                cls   = int(parts[7]) if len(parts) > 7 else 1
                vis   = float(parts[8]) if len(parts) > 8 else 1.0
                if filter_pedestrians and (cls != 1 or vis <= 0.0):
                    continue
                gt_by_frame.setdefault(frame, {})[tid] = np.array([x, y, w, h], dtype=np.float32)

    # Detecções Públicas
    det_by_frame = {}
    det_file = os.path.join(seq_path, "det", "det.txt")
    if os.path.exists(det_file):
        with open(det_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame = int(parts[0])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                conf  = float(parts[6]) if len(parts) > 6 else 1.0
                if conf < min_det_conf:
                    continue
                det_by_frame.setdefault(frame, []).append([x, y, w, h, conf])

    return gt_by_frame, det_by_frame, seq_info


def get_mot17_splits(base_dir: str = "data/MOT17/train", det_suffix: str = "SDP"):
    """
    Define splits por sequência inteira:
    - Treino : MOT17-02, 04, 05, 10, 13
    - Validação: MOT17-09 (câmera estática), MOT17-11 (câmera móvel)
    """
    train_names = ["02", "04", "05", "10", "13"]
    val_names   = ["09", "11"]

    def _paths(names):
        return [os.path.join(base_dir, f"MOT17-{n}-{det_suffix}")
                for n in names
                if os.path.exists(os.path.join(base_dir, f"MOT17-{n}-{det_suffix}"))]

    return {"train": _paths(train_names), "val": _paths(val_names)}


def compute_sequence_density(gt_by_frame: dict) -> float:
    """Densidade média de objetos por quadro."""
    if not gt_by_frame:
        return 0.0
    return float(np.mean([len(boxes) for boxes in gt_by_frame.values()]))
