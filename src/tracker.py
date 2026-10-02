"""
src/tracker.py
Rastreador de Linha de Base (Baseline por Quadro - Parte 1) e Modelo Recorrente (Trilha A - Parte 2):
- NaiveTracker: Associação espacial frame-a-frame por IoU (Hungarian ou Guloso)
- RNNMotionTracker: Modelo recorrente de movimento com projeção autoregressiva (rollout) e incerteza
- Suporte a min_conf para filtragem de detecções de baixa confiança (P0-7)
- Protocolo unificado de constantes e rastreamento de predições perdidas para visualização (P1-4)
"""

import os
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.metrics import box_iou_matrix, calculate_iou
from src.models import MotionPredictor
from src.config import DEFAULT_TRACKER_IOU, DEFAULT_MAX_LOST_FRAMES, FINAL_CHECKPOINT_PATH


class Track:
    """Representa o estado de uma trajetória ao longo do tempo (Baseline Naive)."""
    def __init__(self, track_id: int, initial_box: np.ndarray, frame_id: int):
        self.track_id = track_id
        self.last_box = np.asarray(initial_box, dtype=np.float32)
        self.history = {frame_id: self.last_box}
        self.lost_predictions = {}
        self.time_since_update = 0
        self.state = "active"
        self.age = 1

    def update(self, box: np.ndarray, frame_id: int):
        """Atualiza a trajetória com uma nova observação detectada."""
        self.last_box = np.asarray(box, dtype=np.float32)
        self.history[frame_id] = self.last_box
        self.time_since_update = 0
        self.state = "active"
        self.age += 1

    def mark_missed(self, frame_id: int = None):
        """Registra ausência de detecção no quadro atual."""
        self.time_since_update += 1
        self.state = "lost"
        self.age += 1
        if frame_id is not None:
            self.lost_predictions[frame_id] = self.last_box.copy()


class NaiveTracker:
    """
    Rastreador ingênuo baseado em sobreposição espacial de caixas (IoU) quadro a quadro.
    """
    def __init__(
        self,
        iou_threshold: float = DEFAULT_TRACKER_IOU,
        max_lost_frames: int = DEFAULT_MAX_LOST_FRAMES,
        matching_method: str = "hungarian",
        min_conf: float = 0.0,
        adaptive_decay: float = 0.0,
    ):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.matching_method = matching_method
        self.min_conf = min_conf
        self.adaptive_decay = adaptive_decay
        self.next_id = 1
        self.tracks = []

    def reset(self):
        """Reinicia o estado do rastreador para uma nova sequência."""
        self.next_id = 1
        self.tracks = []

    def _get_threshold(self, track: Track) -> float:
        """Limiar de IoU com relaxamento opcional para controle justo de estresse."""
        if self.adaptive_decay > 0.0:
            lost = getattr(track, "time_since_update", 0)
            decay = min(0.20, self.adaptive_decay * lost)
            return max(0.10, self.iou_threshold - decay)
        return self.iou_threshold

    def step(self, detections: list, frame_id: int):
        """
        Processa um único quadro:
        detections: lista ou array de [x, y, w, h] (ou [x, y, w, h, conf])
        frame_id: índice temporal do quadro
        Retorna: dict {track_id: box} das tracks ativas no quadro
        """
        if isinstance(detections, dict):
            detections = list(detections.values())

        det_boxes = []
        for d in detections:
            conf = float(d[4]) if len(d) > 4 else 1.0
            if conf >= self.min_conf:
                det_boxes.append(d[:4])
        det_boxes = np.array(det_boxes, dtype=np.float32) if len(det_boxes) > 0 else np.empty((0, 4), dtype=np.float32)

        candidate_tracks = [t for t in self.tracks if t.state in ["active", "lost"]]
        matched_tracks = set()
        matched_dets = set()

        if len(candidate_tracks) > 0 and len(det_boxes) > 0:
            track_boxes = np.array([t.last_box for t in candidate_tracks], dtype=np.float32)
            ious = box_iou_matrix(track_boxes, det_boxes)

            if self.matching_method == "hungarian":
                row_ind, col_ind = linear_sum_assignment(-ious)
                for r, c in zip(row_ind, col_ind):
                    thresh = self._get_threshold(candidate_tracks[r])
                    if ious[r, c] >= thresh:
                        candidate_tracks[r].update(det_boxes[c], frame_id)
                        matched_tracks.add(r)
                        matched_dets.add(c)
            else:
                flat_order = np.argsort(-ious, axis=None)
                for idx in flat_order:
                    r, c = np.unravel_index(idx, ious.shape)
                    thresh = self._get_threshold(candidate_tracks[r])
                    if ious[r, c] < thresh:
                        break
                    if r not in matched_tracks and c not in matched_dets:
                        candidate_tracks[r].update(det_boxes[c], frame_id)
                        matched_tracks.add(r)
                        matched_dets.add(c)

        # 1. Atualiza tracks que não tiveram match
        for i, track in enumerate(candidate_tracks):
            if i not in matched_tracks:
                track.mark_missed(frame_id=frame_id)
                if track.time_since_update > self.max_lost_frames:
                    track.state = "dead"

        # 2. Regra de Nascimento: novas detecções iniciam tracks
        for j in range(len(det_boxes)):
            if j not in matched_dets:
                new_track = Track(self.next_id, det_boxes[j], frame_id)
                self.next_id += 1
                self.tracks.append(new_track)

        current_active = {}
        for t in self.tracks:
            if t.time_since_update == 0:
                current_active[t.track_id] = t.last_box

        return current_active

    def track_sequence(self, det_by_frame: dict):
        """Rastreia uma sequência completa dada em det_by_frame."""
        self.reset()
        all_frames = sorted(list(det_by_frame.keys()))
        pred_by_frame = {}

        for f in all_frames:
            dets = det_by_frame[f]
            active_tracks = self.step(dets, f)
            pred_by_frame[f] = active_tracks

        return pred_by_frame


# ==============================================================================
# PARTE 2 — TRILHA A: MODELO DE MOVIMENTO RECORRENTE (RNN / LSTM / GRU)
# ==============================================================================

class RNNTrack:
    """
    Representa o estado de uma trajetória governada por uma Rede Recorrente (Trilha A).
    Mantém estado oculto, caixa predita, incerteza sigma e histórico de predição em oclusão.
    """
    def __init__(self, track_id: int, initial_box: np.ndarray, frame_id: int,
                 model, im_w: float = 1920.0, im_h: float = 1080.0, device="cpu"):
        self.track_id = track_id
        self.last_box = np.asarray(initial_box, dtype=np.float32)
        self.predicted_box = self.last_box.copy()
        self.predicted_sigma = np.zeros(4, dtype=np.float32)
        self.history = {frame_id: self.last_box}
        self.lost_predictions = {}
        self.prev_box = self.last_box.copy()
        self.time_since_update = 0
        self.state = "active"
        self.age = 1

        self.im_w = float(im_w)
        self.im_h = float(im_h)
        self.device = torch.device(device)

        # Inicializa estado oculto
        self.hidden = model.init_hidden(1, self.device)
        self._predict_next(model, dt=1.0)

    def _box_to_normalized_feat(self, box, prev_box, dt=1.0):
        """Converte caixa [x, y, w, h] para vetor normalizado [cx, cy, w, h, vx, vy, dt]."""
        cx = (box[0] + box[2] / 2.0) / self.im_w
        cy = (box[1] + box[3] / 2.0) / self.im_h
        w = box[2] / self.im_w
        h = box[3] / self.im_h

        prev_cx = (prev_box[0] + prev_box[2] / 2.0) / self.im_w
        prev_cy = (prev_box[1] + prev_box[3] / 2.0) / self.im_h
        vx = cx - prev_cx
        vy = cy - prev_cy

        feat = np.array([cx, cy, w, h, vx, vy, dt], dtype=np.float32)
        box_norm = np.array([cx, cy, w, h], dtype=np.float32)
        return feat, box_norm

    def _unnormalize_box(self, box_norm):
        """Converte de [cx, cy, w, h] normalizado para [x, y, w, h] em pixels."""
        cx = float(box_norm[0]) * self.im_w
        cy = float(box_norm[1]) * self.im_h
        w = float(box_norm[2]) * self.im_w
        h = float(box_norm[3]) * self.im_h
        x = cx - w / 2.0
        y = cy - h / 2.0
        return np.array([x, y, w, h], dtype=np.float32)

    def _predict_next(self, model, dt=1.0):
        """Executa um passo da RNN para prever a caixa do próximo quadro."""
        feat, box_norm = self._box_to_normalized_feat(self.last_box, self.prev_box, dt=dt)
        x_t = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        curr_box = torch.tensor(box_norm, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            pred_box_norm, log_sigma, self.hidden = model.step(x_t, curr_box, self.hidden)

        pred_box_np = pred_box_norm.squeeze(0).cpu().numpy()
        self.predicted_box = self._unnormalize_box(pred_box_np)

        if log_sigma is not None:
            self.predicted_sigma = torch.exp(log_sigma).squeeze(0).cpu().numpy()

    def update(self, box: np.ndarray, frame_id: int, model, dt=1.0):
        """Atualiza a trajetória com nova observação e prevê o próximo passo."""
        self.prev_box = self.last_box.copy()
        self.last_box = np.asarray(box, dtype=np.float32)
        self.history[frame_id] = self.last_box
        self.time_since_update = 0
        self.state = "active"
        self.age += 1
        self._predict_next(model, dt=dt)

    def mark_missed(self, model, dt=1.0, velocity_damping=1.0, sigma_inflation=0.0, frame_id: int = None):
        """
        Sob oclusão: roda a RNN para frente em modo autoregressivo (free-running),
        mantendo a estimativa de movimento e incerteza no espaço.
        """
        self.time_since_update += 1
        self.state = "lost"
        self.age += 1

        if frame_id is not None:
            self.lost_predictions[frame_id] = self.predicted_box.copy()

        # Alimenta a própria previsão anterior na recorrência
        self.prev_box = self.last_box.copy()
        self.last_box = self.predicted_box.copy()

        if velocity_damping != 1.0:
            cx = (self.last_box[0] + self.last_box[2] / 2.0)
            cy = (self.last_box[1] + self.last_box[3] / 2.0)
            pcx = (self.prev_box[0] + self.prev_box[2] / 2.0)
            pcy = (self.prev_box[1] + self.prev_box[3] / 2.0)
            vx = (cx - pcx) * velocity_damping
            vy = (cy - pcy) * velocity_damping
            self.prev_box[0] = cx - vx - self.last_box[2] / 2.0
            self.prev_box[1] = cy - vy - self.last_box[3] / 2.0
            self.prev_box[2:] = self.last_box[2:]

        self._predict_next(model, dt=dt)
        if sigma_inflation > 0.0:
            self.predicted_sigma += (sigma_inflation * 0.1)


class RNNMotionTracker:
    """
    Rastreador com Memória Temporal Recorrente (Trilha A):
    - Associação via IoU entre caixa PREVISTA pelo modelo recorrente e detecções observadas
    - Portão adaptativo baseado na incerteza gaussiana sigma prevista
    - Sob oclusão: rollout autoregressivo mantendo velocidade e momento
    """
    def __init__(
        self,
        model: str or MotionPredictor = FINAL_CHECKPOINT_PATH,
        iou_threshold: float = DEFAULT_TRACKER_IOU,
        max_lost_frames: int = DEFAULT_MAX_LOST_FRAMES,
        matching_method: str = "hungarian",
        use_adaptive_gating: bool = True,
        velocity_damping: float = 1.0,
        sigma_inflation: float = 0.0,
        min_conf: float = 0.0,
        emit_predicted_boxes_on_lost: bool = False,
        device: str = "cpu"
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.device = torch.device(device)

        if isinstance(model, str):
            ckpt = torch.load(model, map_location=self.device)
            cell_type = ckpt.get("cell_type", "lstm")
            hidden_dim = ckpt.get("hidden_dim", 128)
            num_layers = ckpt.get("num_layers", 2)
            predict_uncertainty = ckpt.get("predict_uncertainty", True)
            self.model = MotionPredictor(
                cell_type=cell_type,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                predict_uncertainty=predict_uncertainty
            )
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()

        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.matching_method = matching_method
        self.use_adaptive_gating = use_adaptive_gating
        self.velocity_damping = velocity_damping
        self.sigma_inflation = sigma_inflation
        self.min_conf = min_conf
        self.emit_predicted_boxes_on_lost = emit_predicted_boxes_on_lost

        self.next_id = 1
        self.tracks = []

    def reset(self):
        self.next_id = 1
        self.tracks = []

    def _adaptive_threshold(self, track) -> float:
        """
        Portão adaptativo calibrado pela incerteza prevista pelo modelo:
        - Sem intervenção (sigma_inflation=0): reduz até 0.10 proporcionalmente à incerteza sigma prevista.
        - Com sigma_inflation > 0: escala a incerteza aprendida da rede:
          thresh = max(0.10, base - sigma_inflation * mean_sigma).
        """
        if not self.use_adaptive_gating:
            return self.iou_threshold

        base = self.iou_threshold
        mean_sigma = float(np.mean(getattr(track, "predicted_sigma", np.zeros(4))))

        if self.sigma_inflation > 0.0:
            decay = min(0.20, self.sigma_inflation * mean_sigma)
            return max(0.10, base - decay)
        else:
            return max(0.15, base - 0.10 * min(1.0, mean_sigma))

    def step(self, detections: list or dict, frame_id: int, im_width: float = 1920.0,
             im_height: float = 1080.0, dt: float = 1.0):
        """Processa um quadro com predição recorrente."""
        if isinstance(detections, dict):
            detections = list(detections.values())

        det_boxes = []
        for d in detections:
            conf = float(d[4]) if len(d) > 4 else 1.0
            if conf >= self.min_conf:
                det_boxes.append(d[:4])
        det_boxes = np.array(det_boxes, dtype=np.float32) if len(det_boxes) > 0 else np.empty((0, 4), dtype=np.float32)

        candidate_tracks = [t for t in self.tracks if t.state in ["active", "lost"]]
        matched_tracks = set()
        matched_dets = set()

        if len(candidate_tracks) > 0 and len(det_boxes) > 0:
            pred_boxes = np.array([t.predicted_box for t in candidate_tracks], dtype=np.float32)
            ious = box_iou_matrix(pred_boxes, det_boxes)

            if self.matching_method == "hungarian":
                row_ind, col_ind = linear_sum_assignment(-ious)
                for r, c in zip(row_ind, col_ind):
                    track = candidate_tracks[r]
                    thresh = self._adaptive_threshold(track)
                    if ious[r, c] >= thresh:
                        track.update(det_boxes[c], frame_id, self.model, dt=dt)
                        matched_tracks.add(r)
                        matched_dets.add(c)
            else:
                flat_order = np.argsort(-ious, axis=None)
                for idx in flat_order:
                    r, c = np.unravel_index(idx, ious.shape)
                    track = candidate_tracks[r]
                    thresh = self._adaptive_threshold(track)
                    if ious[r, c] < thresh:
                        break
                    if r not in matched_tracks and c not in matched_dets:
                        track.update(det_boxes[c], frame_id, self.model, dt=dt)
                        matched_tracks.add(r)
                        matched_dets.add(c)

        # 1. Tracks não associadas (sob oclusão): rodam a RNN autoregressivamente
        for i, track in enumerate(candidate_tracks):
            if i not in matched_tracks:
                track.mark_missed(
                    self.model,
                    dt=dt,
                    velocity_damping=self.velocity_damping,
                    sigma_inflation=self.sigma_inflation,
                    frame_id=frame_id
                )
                if track.time_since_update > self.max_lost_frames:
                    track.state = "dead"

        # 2. Novas detecções iniciam tracks
        for j in range(len(det_boxes)):
            if j not in matched_dets:
                new_track = RNNTrack(
                    self.next_id, det_boxes[j], frame_id, self.model,
                    im_w=im_width, im_h=im_height, device=self.device
                )
                self.next_id += 1
                self.tracks.append(new_track)

        # Emissão de caixas ativas
        current_active = {}
        for t in self.tracks:
            if t.time_since_update == 0:
                current_active[t.track_id] = t.last_box
            elif self.emit_predicted_boxes_on_lost and t.time_since_update <= self.max_lost_frames:
                current_active[t.track_id] = t.predicted_box

        return current_active

    def track_sequence(self, det_by_frame: dict, im_width: float = 1920.0,
                       im_height: float = 1080.0, dt: float = 1.0):
        """Rastreia uma sequência completa utilizando predição recorrente."""
        self.reset()
        all_frames = sorted(list(det_by_frame.keys()))
        pred_by_frame = {}

        for f in all_frames:
            dets = det_by_frame[f]
            active_tracks = self.step(dets, f, im_width=im_width, im_height=im_height, dt=dt)
            pred_by_frame[f] = active_tracks

        return pred_by_frame
