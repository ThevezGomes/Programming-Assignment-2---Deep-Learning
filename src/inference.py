"""
src/inference.py
Inferência com modelos de detecção pré-treinados do torchvision e rastreador recorrente:
- Suporte a qualquer sequência: pasta MOT17, pasta de imagens genérica ou arquivo de vídeo (.mp4/.avi)
- Resolução dinâmica extraída de seqinfo.ini ou do primeiro quadro (sem assumir 1920x1080)
- Detecção sob demanda com Faster R-CNN (COCO) e custom_nms quando det.txt não estiver disponível
- Geração simultânea de GIF e MP4
- Contagem robusta de pedestres únicos com filtro de confirmação min_hits
"""

import os
import cv2
import torch
import torchvision
from PIL import Image, ImageDraw
import numpy as np

from src.metrics import custom_nms
from src.tracker import RNNMotionTracker
from src.config import FINAL_CHECKPOINT_PATH, DEFAULT_TRACKER_IOU, DEFAULT_MAX_LOST_FRAMES


def get_torchvision_person_detector(device="cpu"):
    """
    Carrega o Faster R-CNN ResNet-50 FPN pré-treinado do torchvision em modo de avaliação.
    """
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    return model


def infer_torchvision_frame(model, image, min_score: float = 0.5, nms_iou: float = 0.5, device="cpu"):
    """
    Executa inferência em uma imagem (PIL Image ou path), filtra a classe 'person' (ID 1 no COCO)
    e aplica o custom_nms de autoria própria (respeitando a proibição de torchvision.ops.nms).
    
    Retorna: detecções no formato [[x, y, w, h, score], ...]
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    transform = torchvision.transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)[0]

    boxes = preds["boxes"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()

    # Filtra classe 1 (person no COCO) e score mínimo
    person_mask = (labels == 1) & (scores >= min_score)
    person_boxes = boxes[person_mask]
    person_scores = scores[person_mask]

    if len(person_boxes) == 0:
        return []

    # Converte de [x1, y1, x2, y2] para [x, y, w, h]
    xywh_boxes = np.zeros_like(person_boxes)
    xywh_boxes[:, 0] = person_boxes[:, 0]
    xywh_boxes[:, 1] = person_boxes[:, 1]
    xywh_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    xywh_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    # Aplicação obrigatória do NMS próprio
    keep_indices = custom_nms(xywh_boxes, person_scores, iou_threshold=nms_iou)

    filtered_dets = []
    for idx in keep_indices:
        x, y, w, h = xywh_boxes[idx]
        sc = person_scores[idx]
        filtered_dets.append([float(x), float(y), float(w), float(h), float(sc)])

    return filtered_dets


def get_distinct_color(track_id: int):
    """Gera uma cor RGB viva e consistente baseada no hash/id da track."""
    import colorsys
    golden_ratio_conjugate = 0.618033988749895
    h = (track_id * golden_ratio_conjugate) % 1.0
    s = 0.85
    v = 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def run_tracking_inference(
    seq_path: str,
    model_path: str = FINAL_CHECKPOINT_PATH,
    output_video_path: str = "results/inferencia_output.gif",
    max_frames: int = None,
    fps: int = 10,
    render_scale: float = 1.0,
    velocity_damping: float = 0.85,
    sigma_inflation: float = 0.30,
    iou_threshold: float = DEFAULT_TRACKER_IOU,
    max_lost_frames: int = DEFAULT_MAX_LOST_FRAMES,
    min_hits: int = 3,
    min_det_score: float = 0.5,
    device: str = "cpu"
) -> dict:
    """
    Executa inferência ponta a ponta sobre uma sequência qualquer sem retreino:
    - Suporta 3 tipos de entrada:
      (a) Pasta MOT17 (img1/ + det/det.txt opcional);
      (b) Pasta de imagens arbitrárias (ordena alfanumericamente);
      (c) Arquivo de vídeo (.mp4, .avi, etc.).
    - Dimensões extraídas dinamicamente de seqinfo.ini ou do primeiro quadro.
    - Detector Faster R-CNN + custom_nms quando det.txt não existir.
    - max_frames=None processa todos os quadros.
    - Exporta GIF e MP4 sincronizados.
    - Contabiliza apenas tracks com pelo menos min_hits confirmações.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Caminho não encontrado: {seq_path}")

    is_video_file = os.path.isfile(seq_path) and seq_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

    raw_frames = []
    det_by_frame = {}
    seq_name = os.path.basename(seq_path)
    im_w, im_h = None, None

    # Caso (c): Arquivo de vídeo
    if is_video_file:
        cap = cv2.VideoCapture(seq_path)
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        if fps_in > 0:
            fps = int(fps_in)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_frames.append(Image.fromarray(frame_rgb))
            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break
        cap.release()
        if raw_frames:
            im_w, im_h = raw_frames[0].size

    # Caso (a) ou (b): Pasta no disco
    else:
        from src.data import parse_seqinfo
        seq_info = parse_seqinfo(seq_path)
        im_dir = os.path.join(seq_path, seq_info.get("imDir", "img1"))
        if not os.path.exists(im_dir):
            im_dir = seq_path  # Pasta com imagens diretamente

        im_exts = (".jpg", ".jpeg", ".png", ".bmp")
        image_files = sorted([
            f for f in os.listdir(im_dir) if f.lower().endswith(im_exts)
        ])

        if max_frames is not None and max_frames > 0:
            image_files = image_files[:max_frames]

        for fname in image_files:
            fpath = os.path.join(im_dir, fname)
            raw_frames.append(Image.open(fpath).convert("RGB"))

        # Lê dimensões de seqinfo.ini ou do primeiro frame
        if "imWidth" in seq_info and "imHeight" in seq_info:
            im_w = int(seq_info["imWidth"])
            im_h = int(seq_info["imHeight"])
        elif raw_frames:
            im_w, im_h = raw_frames[0].size

        # Carrega det.txt público se existir
        det_file = os.path.join(seq_path, "det", "det.txt")
        if os.path.exists(det_file):
            with open(det_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 6:
                        continue
                    fr = int(parts[0])
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    conf = float(parts[6]) if len(parts) > 6 else 1.0
                    det_by_frame.setdefault(fr, []).append([x, y, w, h, conf])

    total_frames = len(raw_frames)
    if total_frames == 0:
        raise ValueError(f"Nenhum quadro válido encontrado para inferência em: {seq_path}")

    if im_w is None or im_h is None:
        im_w, im_h = raw_frames[0].size

    print(f"Iniciando inferência em {seq_name} ({total_frames} quadros | Resolução: {im_w}x{im_h})...")

    # Se não houver det.txt, executa Faster R-CNN + custom_nms
    if not det_by_frame:
        print("Detecções públicas não encontradas. Executando Faster R-CNN (COCO) com custom_nms...")
        detector = get_torchvision_person_detector(device=device)
        for i, pil_img in enumerate(raw_frames, start=1):
            dets = infer_torchvision_frame(detector, pil_img, min_score=min_det_score, device=device)
            det_by_frame[i] = dets

    # Inicializa rastreador com dimensões corretas
    tracker = RNNMotionTracker(
        model=model_path,
        iou_threshold=iou_threshold,
        max_lost_frames=max_lost_frames,
        velocity_damping=velocity_damping,
        sigma_inflation=sigma_inflation,
        use_adaptive_gating=True,
        device=device
    )

    track_hit_counts = {}
    rendered_frames = []
    trail_history = {}

    for f_idx, pil_img in enumerate(raw_frames, start=1):
        dets = det_by_frame.get(f_idx, [])
        active_tracks = tracker.step(dets, f_idx, im_width=float(im_w), im_height=float(im_h))

        for tid in active_tracks.keys():
            track_hit_counts[tid] = track_hit_counts.get(tid, 0) + 1

        # Desenho sobre o quadro
        img_draw = pil_img.copy()
        draw = ImageDraw.Draw(img_draw)

        confirmed_active = 0
        for tid, box in active_tracks.items():
            if track_hit_counts.get(tid, 0) < min_hits:
                # Track ainda em período de confirmação
                continue

            confirmed_active += 1
            color = get_distinct_color(tid)
            x, y, w, h = box
            cx, cy = x + w / 2.0, y + h / 2.0

            trail_history.setdefault(tid, []).append((cx, cy))
            if len(trail_history[tid]) > 20:
                trail_history[tid] = trail_history[tid][-20:]

            pts = trail_history[tid]
            if len(pts) > 1:
                draw.line(pts, fill=color, width=3)

            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

            tag_text = f"ID:{tid}"
            tag_w = len(tag_text) * 10 + 6
            tag_h = 18
            tag_top = max(0, y - tag_h - 2)
            draw.rectangle([x, tag_top, x + tag_w, tag_top + tag_h], fill=color)
            draw.text((x + 3, tag_top + 1), tag_text, fill=(255, 255, 255))

        # Cabeçalho
        confirmed_total_unique = len([tid for tid, count in track_hit_counts.items() if count >= min_hits])
        header_h = 40
        draw.rectangle([0, 0, im_w, header_h], fill=(20, 24, 32))
        info_text = (
            f"Seq: {seq_name} ({im_w}x{im_h}) | Quadro: {f_idx:03d}/{total_frames:03d} | "
            f"Ativas: {confirmed_active:02d} | Objetos Únicos Confirmados (min_hits={min_hits}): {confirmed_total_unique:02d}"
        )
        draw.text((12, 11), info_text, fill=(0, 240, 180))

        if render_scale < 1.0:
            new_size = (int(im_w * render_scale), int(im_h * render_scale))
            img_draw = img_draw.resize(new_size, Image.Resampling.BILINEAR)

        rendered_frames.append(img_draw)

    # Identidades confirmadas
    confirmed_ids = sorted([tid for tid, count in track_hit_counts.items() if count >= min_hits])

    # 1. Salva GIF animado
    gif_path = output_video_path if output_video_path.endswith(".gif") else os.path.splitext(output_video_path)[0] + ".gif"
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    duration_ms = int(1000.0 / fps)
    rendered_frames[0].save(
        gif_path,
        save_all=True,
        append_images=rendered_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True
    )
    print(f"GIF salvo em: {gif_path}")

    # 2. Salva vídeo MP4
    mp4_path = os.path.splitext(output_video_path)[0] + ".mp4"
    frame_w, frame_h = rendered_frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(mp4_path, fourcc, float(fps), (frame_w, frame_h))
    for r_img in rendered_frames:
        frame_cv = cv2.cvtColor(np.array(r_img), cv2.COLOR_RGB2BGR)
        out_writer.write(frame_cv)
    out_writer.release()
    print(f"Vídeo MP4 salvo em: {mp4_path}")

    summary = {
        "sequence": seq_name,
        "resolution": f"{im_w}x{im_h}",
        "total_frames": total_frames,
        "unique_objects_count": len(confirmed_ids),
        "unique_ids": confirmed_ids,
        "gif_path": gif_path,
        "mp4_path": mp4_path,
        "output_video_path": gif_path
    }

    print("=" * 65)
    print("RESUMO DA INFERÊNCIA CONCLUÍDA")
    print(f"Sequência processada        : {seq_name}")
    print(f"Resolução real utilizada    : {im_w}x{im_h}")
    print(f"Quadros analisados          : {total_frames}")
    print(f"Pedestres Únicos Confirmados: {len(confirmed_ids)} (min_hits={min_hits})")
    print(f"Arquivos gerados            : {gif_path} e {mp4_path}")
    print("=" * 65)

    return summary
