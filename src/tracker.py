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

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.metrics import box_iou_matrix, calculate_iou
from src.models import MotionPredictor


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


# ==============================================================================
# PARTE 2 — TRILHA A: MODELO DE MOVIMENTO RECORRENTE (RNN / LSTM / GRU)
# ==============================================================================

class RNNTrack:
    """
    Representa o estado de uma trajetória governada por uma Rede Recorrente (Trilha A).
    Mantém o estado oculto da RNN, caixa predita e incerteza associada.
    """
    def __init__(self, track_id: int, initial_box: np.ndarray, frame_id: int,
                 model, im_w: float = 1920.0, im_h: float = 1080.0, device="cpu"):
        self.track_id = track_id
        self.last_box = np.asarray(initial_box, dtype=np.float32)
        self.predicted_box = self.last_box.copy()
        self.predicted_sigma = np.zeros(4, dtype=np.float32)
        self.history = {frame_id: self.last_box}
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
        w = max(1.0, float(box_norm[2]) * self.im_w)
        h = max(1.0, float(box_norm[3]) * self.im_h)
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

    def mark_missed(self, model, dt=1.0, velocity_damping=1.0, sigma_inflation=0.0):
        """
        Sob oclusão: roda a RNN para frente em modo autoregressivo (free-running),
        mantendo a estimativa de movimento no espaço.
        """
        self.time_since_update += 1
        self.state = "lost"
        self.age += 1

        # Alimenta a própria previsão anterior na recorrência
        self.prev_box = self.last_box.copy()
        self.last_box = self.predicted_box.copy()
        # Intervenção (Parte 4): Amortecimento de velocidade e inflação de incerteza sob oclusão
        cx = (self.last_box[0] + self.last_box[2]/2)
        cy = (self.last_box[1] + self.last_box[3]/2)
        pcx = (self.prev_box[0] + self.prev_box[2]/2)
        pcy = (self.prev_box[1] + self.prev_box[3]/2)
        vx = (cx - pcx) * velocity_damping
        vy = (cy - pcy) * velocity_damping
        self.prev_box[0] = cx - vx - self.last_box[2]/2
        self.prev_box[1] = cy - vy - self.last_box[3]/2
        self.prev_box[2:] = self.last_box[2:]
        self._predict_next(model, dt=dt)
        self.predicted_sigma += sigma_inflation


class RNNMotionTracker:
    """
    Rastreador com Memória Temporal Recorrente (Trilha A):
    - Associação via IoU entre caixa PREVISTA pelo modelo recorrente e detecções observadas
    - Portão adaptativo baseado na incerteza gaussiana sigma
    - Sob oclusão: rollout autoregressivo mantendo velocidade e momento
    """
    def __init__(
        self,
        model: str or MotionPredictor,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 15,
        matching_method: str = "hungarian",
        use_adaptive_gating: bool = True,
        velocity_damping: float = 1.0,
        sigma_inflation: float = 0.0,
        device: str = "cuda"
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        if isinstance(model, str):
            # Carrega de checkpoint
            ckpt = torch.load(model, map_location=self.device)
            cell_type = ckpt.get("cell_type", "lstm")
            hidden_dim = ckpt.get("hidden_dim", 128)
            num_layers = ckpt.get("num_layers", 2)
            self.model = MotionPredictor(cell_type=cell_type, hidden_dim=hidden_dim, num_layers=num_layers)
            self.model.load_state_dict(ckpt["model_state_dict"])
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

        self.next_id = 1
        self.tracks = []

    def reset(self):
        self.next_id = 1
        self.tracks = []

    def _adaptive_threshold(self, track) -> float:
        """
        Portão adaptativo: relaxa o limiar de IoU proporcionalmente ao tempo de oclusão.
        Sem correção (sigma_inflation=0): mantém o limiar base.
        Com correção: reduz o limiar 0.01 por frame perdido (mínimo 0.10).
        Isso permite que tracks ocluídas por muitos frames reacitem detecções mesmo
        que a caixa prevista pela RNN tenha divergido ligeiramente da posição real.
        """
        if not self.use_adaptive_gating:
            return self.iou_threshold
        base = self.iou_threshold
        if self.sigma_inflation > 0.0:
            # Decai 0.01 por frame perdido, no máximo 0.20 de desconto
            lost = getattr(track, "time_since_update", 0)
            decay = min(0.20, 0.01 * lost * self.sigma_inflation * 10)
            return max(0.10, base - decay)
        else:
            # Comportamento original: usa incerteza gaussiana prevista
            uncertainty = float(np.mean(track.predicted_sigma))
            return max(0.15, base - 0.1 * min(1.0, uncertainty))

    def step(self, detections: list or dict, frame_id: int, im_width: float = 1920.0,
             im_height: float = 1080.0, dt: float = 1.0):
        """Processa um quadro com predição recorrente."""
        if isinstance(detections, dict):
            detections = list(detections.values())

        det_boxes = []
        for d in detections:
            det_boxes.append(d[:4])
        det_boxes = np.array(det_boxes, dtype=np.float32) if len(det_boxes) > 0 else np.empty((0, 4))

        candidate_tracks = [t for t in self.tracks if t.state in ["active", "lost"]]
        matched_tracks = set()
        matched_dets = set()

        if len(candidate_tracks) > 0 and len(det_boxes) > 0:
            # Ponto-chave da Trilha A: compara as caixas PREVISTAS pela RNN (e não as congeladas!)
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
                track.mark_missed(self.model, dt=dt, velocity_damping=self.velocity_damping, sigma_inflation=self.sigma_inflation)
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

        # Retorna apenas tracks com observação no quadro atual
        current_active = {}
        for t in self.tracks:
            if t.time_since_update == 0:
                current_active[t.track_id] = t.last_box

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

