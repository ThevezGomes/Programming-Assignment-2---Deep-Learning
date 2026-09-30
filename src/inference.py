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
