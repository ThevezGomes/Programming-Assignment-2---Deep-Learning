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
    contrast: float = 1.0,
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
    contrast          : fator de escala do contraste das elipses (padrão 1.0)
    seed              : semente aleatória para reprodutibilidade

    Limite de Visibilidade:
    Objetos com menos de 8% de área visível no z-buffer (visible_frac < 0.08) são
    considerados em oclusão total e removidos do ground truth naquele quadro.

    Retorna
    -------
    frames     : np.ndarray (num_frames, frame_size, frame_size, 3) uint8 RGB
    gt_by_frame: dict {frame_id: {object_id: np.array([x, y, w, h])}}
                 x,y = canto superior esquerdo da bounding box, w,h = largura e altura
    """
    rng = np.random.default_rng(seed)

    # --- Inicializa os objetos ---
    semi_a = rng.integers(7, 13, size=num_objects).astype(float)   # semi-eixo horizontal
    semi_b = rng.integers(7, 13, size=num_objects).astype(float)   # semi-eixo vertical
    cx = rng.uniform(semi_a + 4, frame_size - semi_a - 4)           # centro X
    cy = rng.uniform(semi_b + 4, frame_size - semi_b - 4)           # centro Y
    angle = rng.uniform(0, 2 * np.pi, size=num_objects)             # ângulo de rotação (estético)

    # Velocidades (pixels/frame)
    vx = rng.uniform(-typical_velocity, typical_velocity, size=num_objects)
    vy = rng.uniform(-typical_velocity, typical_velocity, size=num_objects)
    # Garante que nenhum inicie com velocidade zero
    vx = np.where(np.abs(vx) < 0.5, 0.5 * np.sign(vx + 1e-6), vx)
    vy = np.where(np.abs(vy) < 0.5, 0.5 * np.sign(vy + 1e-6), vy)

    # Z-order fixo para toda a sequência: índice menor = mais fundo (atrás)
    z_order = np.arange(num_objects)  # objeto 0 fica atrás de todos os outros

    # Se occlusion_duration for especificado e houver pelo menos 2 objetos,
    # programa o objeto 0 (fundo) para cruzar diretamente atrás do objeto 1 (frente)
    if occlusion_duration > 0 and num_objects >= 2:
        # Aumenta o oclusor (objeto 1) para garantir cobertura completa
        semi_a[1] = max(semi_a[1], semi_a[0] + 6.0)
        semi_b[1] = max(semi_b[1], semi_b[0] + 6.0)
        # Oclusor no centro
        cx[1] = frame_size / 2.0
        cy[1] = frame_size / 2.0
        vx[1] = 0.0
        vy[1] = 0.0

        # Alvo (objeto 0) atravessa na horizontal na mesma altura
        cy[0] = cy[1]
        vy[0] = 0.0
        # Velocidade calculada para ficar ocluído por aproximadamente occlusion_duration quadros
        eff_occ_dur = max(2, min(occlusion_duration, num_frames // 2))
        occ_vx = (2.0 * (semi_a[1] - semi_a[0])) / float(eff_occ_dur)
        occ_vx = max(0.8, min(occ_vx, float(typical_velocity * 1.5)))
        vx[0] = occ_vx
        # Posiciona para cruzar o centro no meio do vídeo
        mid_f = num_frames // 2
        cx[0] = cx[1] - vx[0] * mid_f

    # Cores distintas por objeto (HSV espaçado uniformemente -> RGB)
    colors = []
    for i in range(num_objects):
        hue = i / num_objects
        h = hue * 6
        x_c = 1 - abs(h % 2 - 1)
        r, g, b = [(1, x_c, 0), (x_c, 1, 0), (0, 1, x_c),
                   (0, x_c, 1), (x_c, 0, 1), (1, 0, x_c)][int(h) % 6]
        colors.append((int(r * 200) + 55, int(g * 200) + 55, int(b * 200) + 55))

    frames = []
    gt_by_frame = {}

    yy, xx = np.mgrid[0:frame_size, 0:frame_size]

    for f in range(num_frames):
        # Fundo escuro com leve textura/ruído
        frame = np.full((frame_size, frame_size, 3), 30, dtype=np.uint8)
        if noise_level > 0:
            bg_noise = rng.normal(0, noise_level * 50, (frame_size, frame_size, 3))
            frame = np.clip(frame.astype(float) + bg_noise, 0, 255).astype(np.uint8)

        z_buffer = -np.ones((frame_size, frame_size), dtype=int)

        # Desenha os objetos em ordem de z (do mais fundo para o mais próximo)
        for z in range(num_objects):
            obj_idx = np.argsort(z_order)[z]

            cos_a = np.cos(angle[obj_idx])
            sin_a = np.sin(angle[obj_idx])
            dx = xx - cx[obj_idx]
            dy = yy - cy[obj_idx]
            x_rot = dx * cos_a + dy * sin_a
            y_rot = -dx * sin_a + dy * cos_a

            mask = ((x_rot / semi_a[obj_idx]) ** 2 + (y_rot / semi_b[obj_idx]) ** 2) <= 1.0

            frame[mask] = np.clip(np.array(colors[obj_idx]) * contrast, 0, 255).astype(np.uint8)
            z_buffer[mask] = obj_idx

        frames.append(frame)

        # Ground Truth: apenas se tiver visibilidade real no z-buffer
        gt_by_frame[f + 1] = {}
        for obj_idx in range(num_objects):
            visible = (z_buffer == obj_idx)
            area_elipse = np.pi * semi_a[obj_idx] * semi_b[obj_idx]
            visible_frac = visible.sum() / max(1.0, area_elipse)

            # Requisito de oclusão real: se ocluído quase totalmente (< 5% visível), desaparece do GT
            if visible_frac >= 0.08:
                cos_a = np.cos(angle[obj_idx])
                sin_a = np.sin(angle[obj_idx])
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

        # Reflexão suave nas bordas (exceto objeto 1 se for oclusor fixo programado)
        for i in range(num_objects):
            if occlusion_duration > 0 and i == 1:
                continue
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

    frames = np.stack(frames, axis=0)
    return frames, gt_by_frame


def generate_controlled_occlusion_sequence(
    num_frames: int = 35,
    occlusion_duration: int = 8,
    frame_size: int = 128
):
    """
    Gera um cenário determinístico para demonstrar o requisito verificável da Parte 0:
    Uma elipse (Objeto 1, alvo verde no fundo z=0) passa diretamente atrás de uma
    segunda elipse (Objeto 2, oclusor vermelho na frente z=1).

    O Objeto 1 fica ocluído no z-buffer (< 8% de área visível) por aproximadamente
    `occlusion_duration` quadros e reaparece em seguida com a mesma identidade.

    Retorna
    -------
    frames: np.ndarray (num_frames, 128, 128, 3)
    gt_by_frame: dict {frame: {id: [x, y, w, h]}}
    occlusion_info: dict com start_frame, end_frame, num_occluded_frames
    """
    yy, xx = np.mgrid[0:frame_size, 0:frame_size]

    # Objeto 1: Alvo (Verde esmeralda, menor, fundo z=0)
    target_semi_a = 7.0
    target_semi_b = 10.0
    color_target = (0, 230, 115)  # Verde

    # Objeto 2: Oclusor (Vermelho carmesim, maior, frente z=1)
    occluder_semi_a = 18.0
    occluder_semi_b = 22.0
    occluder_cx = frame_size / 2.0
    occluder_cy = frame_size / 2.0
    color_occluder = (235, 55, 55)  # Vermelho

    # Velocidade do alvo calculada para ficar completamente coberto por occlusion_duration quadros
    # Distância em que o alvo fica 100% contido no interior do oclusor:
    # (occluder_cx - occluder_semi_a + target_semi_a) até (occluder_cx + occluder_semi_a - target_semi_a)
    hidden_span = 2.0 * (occluder_semi_a - target_semi_a)
    target_vx = hidden_span / float(occlusion_duration)

    mid_frame = num_frames // 2
    # Define cx inicial do alvo para que o meio da oclusão ocorra no mid_frame
    target_cx_init = occluder_cx - target_vx * mid_frame

    frames = []
    gt_by_frame = {}
    occluded_frames = []

    target_cx = target_cx_init
    target_cy = occluder_cy

    for f in range(1, num_frames + 1):
        frame = np.full((frame_size, frame_size, 3), 32, dtype=np.uint8)
        z_buffer = -np.ones((frame_size, frame_size), dtype=int)

        # 1. Desenha Alvo (z=0, mais fundo)
        dx_t = xx - target_cx
        dy_t = yy - target_cy
        mask_target = ((dx_t / target_semi_a) ** 2 + (dy_t / target_semi_b) ** 2) <= 1.0
        frame[mask_target] = color_target
        z_buffer[mask_target] = 0

        # 2. Desenha Oclusor (z=1, frente, sobrescreve)
        dx_o = xx - occluder_cx
        dy_o = yy - occluder_cy
        mask_occluder = ((dx_o / occluder_semi_a) ** 2 + (dy_o / occluder_semi_b) ** 2) <= 1.0
        frame[mask_occluder] = color_occluder
        z_buffer[mask_occluder] = 1

        frames.append(frame)
        gt_by_frame[f] = {}

        # Oclusor sempre visível
        gt_by_frame[f][2] = np.array([
            occluder_cx - occluder_semi_a,
            occluder_cy - occluder_semi_b,
            2 * occluder_semi_a,
            2 * occluder_semi_b
        ], dtype=np.float32)

        # Alvo: visível se tiver pelo menos 8% da área visível no z-buffer
        target_vis_pixels = (z_buffer == 0).sum()
        target_area = np.pi * target_semi_a * target_semi_b
        target_vis_frac = target_vis_pixels / max(1.0, target_area)

        if target_vis_frac >= 0.08:
            gt_by_frame[f][1] = np.array([
                target_cx - target_semi_a,
                target_cy - target_semi_b,
                2 * target_semi_a,
                2 * target_semi_b
            ], dtype=np.float32)
        else:
            occluded_frames.append(f)

        target_cx += target_vx

    frames = np.stack(frames, axis=0)
    occlusion_info = {
        "start_frame": min(occluded_frames) if occluded_frames else -1,
        "end_frame": max(occluded_frames) if occluded_frames else -1,
        "num_occluded_frames": len(occluded_frames),
        "occluded_frames": occluded_frames
    }

    return frames, gt_by_frame, occlusion_info



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
    if drop_prob == 0.0 and noise_std == 0.0 and fp_rate == 0.0:
        return {f: [[*box.tolist(), 1.0] for box in boxes.values()] for f, boxes in gt_by_frame.items()}

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
        if fp_rate > 0.0 and len(gt_boxes) > 0:
            n_fp = rng.poisson(fp_rate * len(gt_boxes))
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
                conf_gt = float(parts[6]) if len(parts) > 6 else 1.0
                cls     = int(parts[7]) if len(parts) > 7 else 1
                vis     = float(parts[8]) if len(parts) > 8 else 1.0
                if filter_pedestrians and (cls != 1 or vis <= 0.0 or conf_gt == 0):
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
