"""
src/tracker.py
Rastreador de Linha de Base (Baseline por Quadro - Parte 1):
- Associação frame-a-frame ingênua por IoU
- Matching ótimo via Algoritmo Húngaro (linear_sum_assignment) ou Guloso
- Gestão do ciclo de vida:
  * Nascimento: Deteção sem match inicia nova track com ID único sequencial
  * Oclusão / Lost: Track sem match acumula contador time_since_update
  * Morte: Track é extinta após k quadros consecutivos sem observação
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from src.metrics import box_iou_matrix, calculate_iou


class Track:
    """Representa o estado de uma trajetória ao longo do tempo."""
    def __init__(self, track_id: int, initial_box: np.ndarray, frame_id: int):
        self.track_id = track_id
        self.last_box = np.asarray(initial_box, dtype=np.float32)
        self.history = {frame_id: self.last_box}
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

    def mark_missed(self):
        """Registra ausência de detecção no quadro atual."""
        self.time_since_update += 1
        self.state = "lost"
        self.age += 1


class NaiveTracker:
    """
    Rastreador ingênuo baseado em sobreposição espacial de caixas (IoU) quadro a quadro.
    """
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 15,
        matching_method: str = "hungarian"
    ):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.matching_method = matching_method
        self.next_id = 1
        self.tracks = []

    def reset(self):
        """Reinicia o estado do rastreador para uma nova sequência."""
        self.next_id = 1
        self.tracks = []

    def step(self, detections: list, frame_id: int):
        """
        Processa um único quadro:
        detections: lista ou array de [x, y, w, h] (ou [x, y, w, h, conf])
        frame_id: índice temporal do quadro
        Retorna: dict {track_id: box} das tracks ativas no quadro
        """
        # Extrai coordenadas [x, y, w, h] (suporta tanto list de caixas quanto dict {id: box})
        if isinstance(detections, dict):
            detections = list(detections.values())

        det_boxes = []
        for d in detections:
            det_boxes.append(d[:4])
        det_boxes = np.array(det_boxes, dtype=np.float32) if len(det_boxes) > 0 else np.empty((0, 4))

        # Seleciona tracks vivas (ativas ou perdidas temporariamente)
        candidate_tracks = [t for t in self.tracks if t.state in ["active", "lost"]]

        matched_tracks = set()
        matched_dets = set()

        if len(candidate_tracks) > 0 and len(det_boxes) > 0:
            track_boxes = np.array([t.last_box for t in candidate_tracks], dtype=np.float32)
            ious = box_iou_matrix(track_boxes, det_boxes)

            if self.matching_method == "hungarian":
                # Algoritmo Húngaro (maximizar IoU = minimizar -IoU)
                row_ind, col_ind = linear_sum_assignment(-ious)
                for r, c in zip(row_ind, col_ind):
                    if ious[r, c] >= self.iou_threshold:
                        candidate_tracks[r].update(det_boxes[c], frame_id)
                        matched_tracks.add(r)
                        matched_dets.add(c)
            else:
                # Matching Guloso
                flat_order = np.argsort(-ious, axis=None)
                for idx in flat_order:
                    r, c = np.unravel_index(idx, ious.shape)
                    if ious[r, c] < self.iou_threshold:
                        break
                    if r not in matched_tracks and c not in matched_dets:
                        candidate_tracks[r].update(det_boxes[c], frame_id)
                        matched_tracks.add(r)
                        matched_dets.add(c)

        # 1. Atualiza tracks que não tiveram match (incrementa contador de oclusão)
        for i, track in enumerate(candidate_tracks):
            if i not in matched_tracks:
                track.mark_missed()
                # Regra de Morte: ultrapassou k quadros sem observação
                if track.time_since_update > self.max_lost_frames:
                    track.state = "dead"

        # 2. Regra de Nascimento: detecções não associadas iniciam novas tracks
        for j in range(len(det_boxes)):
            if j not in matched_dets:
                new_track = Track(self.next_id, det_boxes[j], frame_id)
                self.next_id += 1
                self.tracks.append(new_track)

        # Retorna as tracks associadas neste frame
        current_active = {}
        for t in self.tracks:
            if t.time_since_update == 0:
                current_active[t.track_id] = t.last_box

        return current_active

    def track_sequence(self, det_by_frame: dict):
        """
        Rastreia uma sequência completa dada em det_by_frame = {frame: [[x, y, w, h, conf], ...]}.
        Retorna: pred_by_frame = {frame: {track_id: box}}
        """
        self.reset()
        all_frames = sorted(list(det_by_frame.keys()))
        pred_by_frame = {}

        for f in all_frames:
            dets = det_by_frame[f]
            active_tracks = self.step(dets, f)
            pred_by_frame[f] = active_tracks

        return pred_by_frame
