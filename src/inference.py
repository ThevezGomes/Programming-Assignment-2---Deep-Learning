"""
src/inference.py
Inferência com modelos de detecção pré-treinados do torchvision e rastreador:
- infer_torchvision_frame: Detecção de pessoas usando Faster R-CNN do torchvision com NMS próprio
- run_sequence_inference: Execução completa sobre uma sequência ou vídeo
"""

import os
import torch
import torchvision
from PIL import Image
import numpy as np
from src.metrics import custom_nms
from src.tracker import NaiveTracker


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
    # Espaçamento uniforme pela proporção áurea
    golden_ratio_conjugate = 0.618033988749895
    h = (track_id * golden_ratio_conjugate) % 1.0
    s = 0.85
    v = 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def run_tracking_inference(
    seq_path: str,
    model_path: str = "checkpoints/motion_lstm_best.pt",
    output_video_path: str = "results/inferencia_output.gif",
    max_frames: int = 60,
    fps: int = 8,
    render_scale: float = 0.5,
    velocity_damping: float = 0.85,
    sigma_inflation: float = 0.30,
    iou_threshold: float = 0.30,
    max_lost_frames: int = 15,
    device: str = "cpu"
) -> dict:
    """
    Executa inferência ponta a ponta sobre uma sequência qualquer sem retreino:
    - Carrega a sequência MOT17 (imagens em img1 e detecções em det/det.txt);
    - Se det.txt não existir, usa Faster R-CNN do torchvision com NMS próprio;
    - Executa o RNNMotionTracker com memória recorrente e amortecimento;
    - Produz vídeo (GIF animado) com caixas coloridas consistentemente por ID;
    - Contabiliza o total de identidades únicas observadas no vídeo.

    Retorna dict com:
    - sequence: nome da sequência
    - total_frames: total de quadros processados
    - unique_objects_count: contagem de pedestres únicos identificados
    - output_video_path: caminho do vídeo gerado
    """
    from PIL import ImageDraw, ImageFont
    from src.data import parse_seqinfo
    from src.tracker import RNNMotionTracker

    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Caminho da sequência não encontrado: {seq_path}")

    seq_info = parse_seqinfo(seq_path)
    seq_name = seq_info.get("name", os.path.basename(seq_path))
    im_w = int(seq_info.get("imWidth", 1920))
    im_h = int(seq_info.get("imHeight", 1080))
    im_dir = os.path.join(seq_path, seq_info.get("imDir", "img1"))
    im_ext = seq_info.get("imExt", ".jpg")

    # Identifica arquivos de imagem disponíveis
    image_files = sorted([
        f for f in os.listdir(im_dir) if f.lower().endswith(im_ext.lower())
    ]) if os.path.exists(im_dir) else []

    if max_frames is not None and max_frames > 0:
        image_files = image_files[:max_frames]

    # Carrega detecções se det.txt existir
    det_file = os.path.join(seq_path, "det", "det.txt")
    det_by_frame = {}
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
    else:
        print("det.txt não encontrado. Utilizando detector Faster R-CNN (COCO) com custom_nms...")
        detector = get_torchvision_person_detector(device=device)
        for i, fname in enumerate(image_files, start=1):
            fpath = os.path.join(im_dir, fname)
            dets = infer_torchvision_frame(detector, fpath, min_score=0.5, nms_iou=0.5, device=device)
            det_by_frame[i] = dets

    # Inicializa o rastreador com modelo pré-treinado
    tracker = RNNMotionTracker(
        model=model_path,
        iou_threshold=iou_threshold,
        max_lost_frames=max_lost_frames,
        velocity_damping=velocity_damping,
        sigma_inflation=sigma_inflation,
        use_adaptive_gating=True,
        device=device
    )

    all_unique_ids = set()
    rendered_frames = []
    trail_history = {}  # {track_id: [(cx, cy), ...]}

    print(f"Executando rastreamento recorrente em {len(image_files)} quadros de {seq_name}...")

    for f_idx, fname in enumerate(image_files, start=1):
        fpath = os.path.join(im_dir, fname)
        img = Image.open(fpath).convert("RGB")
        orig_w, orig_h = img.size

        dets = det_by_frame.get(f_idx, [])
        active_tracks = tracker.step(dets, f_idx)

        # Atualiza conjunto de identidades únicas
        for tid in active_tracks.keys():
            all_unique_ids.add(tid)

        # Renderização visual sobre a imagem
        draw = ImageDraw.Draw(img)

        # Desenha rastro (trajetória histórica) e caixas
        for tid, box in active_tracks.items():
            color = get_distinct_color(tid)
            x, y, w, h = box
            cx, cy = x + w / 2.0, y + h / 2.0

            trail_history.setdefault(tid, []).append((cx, cy))
            if len(trail_history[tid]) > 20:
                trail_history[tid] = trail_history[tid][-20:]

            # Rastro
            pts = trail_history[tid]
            if len(pts) > 1:
                draw.line(pts, fill=color, width=3)

            # Bounding Box
            draw.rectangle([x, y, x + w, y + h], outline=color, width=4)

            # Tag com ID
            tag_text = f"ID:{tid}"
            tag_w = len(tag_text) * 11 + 8
            tag_h = 20
            tag_top = max(0, y - tag_h - 2)
            draw.rectangle([x, tag_top, x + tag_w, tag_top + tag_h], fill=color)
            draw.text((x + 4, tag_top + 2), tag_text, fill=(255, 255, 255))

        # Cabeçalho Informativo no topo do vídeo
        header_h = 44
        draw.rectangle([0, 0, orig_w, header_h], fill=(20, 24, 32))
        info_text = (
            f"Sequência: {seq_name} | Quadro: {f_idx:03d}/{len(image_files):03d} | "
            f"Tracks Ativas: {len(active_tracks):02d} | Objetos Únicos Totais: {len(all_unique_ids):02d}"
        )
        draw.text((16, 12), info_text, fill=(0, 240, 180))

        # Redimensiona para formato mais leve se render_scale < 1.0
        if render_scale < 1.0:
            new_size = (int(orig_w * render_scale), int(orig_h * render_scale))
            img = img.resize(new_size, Image.Resampling.BILINEAR)

        rendered_frames.append(img)

    # Salva o arquivo de vídeo (GIF animado de alta qualidade)
    if rendered_frames and output_video_path:
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        duration_ms = int(1000.0 / fps)
        rendered_frames[0].save(
            output_video_path,
            save_all=True,
            append_images=rendered_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True
        )
        print(f"Vídeo de inferência gerado com sucesso: {output_video_path}")

    summary = {
        "sequence": seq_name,
        "total_frames": len(image_files),
        "unique_objects_count": len(all_unique_ids),
        "unique_ids": sorted(list(all_unique_ids)),
        "output_video_path": output_video_path
    }

    print("=" * 60)
    print("RESUMO DA INFERÊNCIA")
    print(f"Sequência processada     : {seq_name}")
    print(f"Quadros analisados       : {len(image_files)}")
    print(f"Contagem de Objetos Únicos: {len(all_unique_ids)} pedestres")
    print(f"Arquivo de saída         : {output_video_path}")
    print("=" * 60)

    return summary

