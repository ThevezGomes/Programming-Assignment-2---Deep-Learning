"""
src/metrics.py
Implementação de métricas de tracking e NMS de autoria própria para o PA2:
- custom_nms: Implementação própria sem usar torchvision.ops.nms
- calculate_iou e box_iou_matrix: Cálculo de IoU entre caixas [x, y, w, h]
- compute_idf1: IDF1 oficial (Ristani et al., 2016 / TrackEval) com matriz de sobreposição global
- compute_id_switches_and_fragmentations: Contagem CLEAR-MOT (preserva par anterior antes do Hungarian)
- compute_map_per_frame: mAP de detecção por quadro usando confiança como score com ordenação estável
- evaluate_tracking: Avaliação consolidada com id_count_error
- run_unit_tests: Bateria de testes unitários com assert (casos a, b, c, d e invariância de AP)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from src.config import EVAL_IOU


def calculate_iou(box1, box2):
    """
    Calcula IoU entre duas caixas no formato [x, y, w, h] (top-left, width, height).
    """
    x1_min, y1_min = float(box1[0]), float(box1[1])
    x1_max, y1_max = float(box1[0] + box1[2]), float(box1[1] + box1[3])

    x2_min, y2_min = float(box2[0]), float(box2[1])
    x2_max, y2_max = float(box2[0] + box2[2]), float(box2[1] + box2[3])

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2])) * max(0.0, float(box1[3]))
    area2 = max(0.0, float(box2[2])) * max(0.0, float(box2[3]))

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def box_iou_matrix(boxes_a, boxes_b):
    """
    Calcula a matriz de IoU entre N caixas e M caixas.
    boxes_a: (N, 4) no formato [x, y, w, h]
    boxes_b: (M, 4) no formato [x, y, w, h]
    Retorna: (N, M)
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    a_x1, a_y1 = boxes_a[:, 0], boxes_a[:, 1]
    a_x2, a_y2 = boxes_a[:, 0] + boxes_a[:, 2], boxes_a[:, 1] + boxes_a[:, 3]

    b_x1, b_y1 = boxes_b[:, 0], boxes_b[:, 1]
    b_x2, b_y2 = boxes_b[:, 0] + boxes_b[:, 2], boxes_b[:, 1] + boxes_b[:, 3]

    inter_x1 = np.maximum(a_x1[:, None], b_x1[None, :])
    inter_y1 = np.maximum(a_y1[:, None], b_y1[None, :])
    inter_x2 = np.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = np.minimum(a_y2[:, None], b_y2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, boxes_a[:, 2]) * np.maximum(0.0, boxes_a[:, 3])
    area_b = np.maximum(0.0, boxes_b[:, 2]) * np.maximum(0.0, boxes_b[:, 3])
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    iou = np.where(union_area > 0, inter_area / np.maximum(union_area, 1e-7), 0.0)
    return iou.astype(np.float32)


def custom_nms(boxes, scores, iou_threshold=0.5):
    """
    Implementação própria de NMS sem torchvision.ops.nms.
    boxes: array-like (N, 4) no formato [x, y, w, h]
    scores: array-like (N,)
    iou_threshold: limiar de corte
    Retorna: lista de índices mantidos
    """
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = np.maximum(0.0, boxes[:, 2]) * np.maximum(0.0, boxes[:, 3])

    order = scores.argsort(kind="mergesort")[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def _to_trajectory_dict(d):
    """Converte dados para dict {frame: {id: [x, y, w, h]}}."""
    if isinstance(d, dict):
        out = {}
        for f, objs in d.items():
            out[int(f)] = {}
            if isinstance(objs, dict):
                for tid, b in objs.items():
                    out[int(f)][int(tid)] = np.asarray(b[:4], dtype=np.float32)
            else:
                for row in objs:
                    out[int(f)][int(row[1])] = np.asarray(row[2:6], dtype=np.float32)
        return out
    out = {}
    for row in d:
        f, tid = int(row[0]), int(row[1])
        box = np.asarray(row[2:6], dtype=np.float32)
        if f not in out:
            out[f] = {}
        out[f][tid] = box
    return out


def compute_idf1(gt_data, pred_data, iou_threshold=EVAL_IOU):
    """
    Calcula IDF1 oficial (Ristani et al., 2016 / TrackEval):
    - C[i, j] acumula o número de quadros onde o IoU entre GT i e Pred j é >= iou_threshold,
      SEM bipartite matching intermediário por quadro.
    - Matching húngaro global único entre trajetórias sobre C.
    """
    gt_dict = _to_trajectory_dict(gt_data)
    pred_dict = _to_trajectory_dict(pred_data)

    gt_ids = sorted(list({tid for f in gt_dict.values() for tid in f.keys()}))
    pred_ids = sorted(list({tid for f in pred_dict.values() for tid in f.keys()}))

    n_gt = len(gt_ids)
    n_pred = len(pred_ids)

    total_gt = sum(len(f) for f in gt_dict.values())
    total_pred = sum(len(f) for f in pred_dict.values())

    if total_gt == 0 and total_pred == 0:
        return {"idf1": 1.0, "idp": 1.0, "idr": 1.0, "idtp": 0, "idfp": 0, "idfn": 0}
    if total_gt == 0 or total_pred == 0 or n_gt == 0 or n_pred == 0:
        return {"idf1": 0.0, "idp": 0.0, "idr": 0.0, "idtp": 0, "idfp": total_pred, "idfn": total_gt}

    gt_to_idx = {gid: i for i, gid in enumerate(gt_ids)}
    pred_to_idx = {pid: j for j, pid in enumerate(pred_ids)}

    cost_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)

    all_frames = set(gt_dict.keys()).intersection(set(pred_dict.keys()))
    for f in all_frames:
        gts = gt_dict[f]
        preds = pred_dict[f]
        if not gts or not preds:
            continue

        f_gt_ids = list(gts.keys())
        f_pred_ids = list(preds.keys())
        f_gt_boxes = np.array([gts[gid] for gid in f_gt_ids], dtype=np.float32)
        f_pred_boxes = np.array([preds[pid] for pid in f_pred_ids], dtype=np.float32)

        ious = box_iou_matrix(f_gt_boxes, f_pred_boxes)
        for r_idx, gid in enumerate(f_gt_ids):
            for c_idx, pid in enumerate(f_pred_ids):
                if ious[r_idx, c_idx] >= iou_threshold:
                    cost_matrix[gt_to_idx[gid], pred_to_idx[pid]] += 1.0

    # Matching global entre identidades de trajetórias
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    idtp = int(sum(cost_matrix[r, c] for r, c in zip(row_ind, col_ind)))

    idfp = total_pred - idtp
    idfn = total_gt - idtp
    denom = 2 * idtp + idfp + idfn
    idf1 = float(2 * idtp / denom) if denom > 0 else 0.0
    idp = float(idtp / total_pred) if total_pred > 0 else 0.0
    idr = float(idtp / total_gt) if total_gt > 0 else 0.0

    return {
        "idf1": idf1,
        "idp": idp,
        "idr": idr,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn
    }


def compute_id_switches_and_fragmentations(gt_data, pred_data, iou_threshold=EVAL_IOU):
    """
    Contagem de trocas de identidade (ID switches) e fragmentações conforme CLEAR-MOT:
    Em cada quadro:
    1. Mantém pares (GT, Pred) ativos no quadro imediatamente anterior se IoU >= limiar;
    2. Roda matching húngaro apenas entre os GTs e detecções restantes;
    3. Registra ID switches e fragmentações sobre a associação temporal.
    """
    gt_dict = _to_trajectory_dict(gt_data)
    pred_dict = _to_trajectory_dict(pred_data)

    all_frames = sorted(list(set(gt_dict.keys()).union(set(pred_dict.keys()))))

    last_pred = {}
    last_frame = {}
    id_switches = 0
    fragmentations = 0

    prev_frame_matches = {}

    for f in all_frames:
        gts = gt_dict.get(f, {})
        preds = pred_dict.get(f, {})
        curr_frame_matches = {}

        if gts and preds:
            f_gt_ids = list(gts.keys())
            f_pred_ids = list(preds.keys())

            matched_gts = set()
            matched_preds = set()

            # Passo 1: Manter associações anteriores válidas (CLEAR-MOT rule)
            for gid, pid in prev_frame_matches.items():
                if gid in gts and pid in preds:
                    iou = calculate_iou(gts[gid], preds[pid])
                    if iou >= iou_threshold:
                        curr_frame_matches[gid] = pid
                        matched_gts.add(gid)
                        matched_preds.add(pid)

            # Passo 2: Hungarian apenas sobre os elementos não casados no Passo 1
            rem_gt_ids = [gid for gid in f_gt_ids if gid not in matched_gts]
            rem_pred_ids = [pid for pid in f_pred_ids if pid not in matched_preds]

            if rem_gt_ids and rem_pred_ids:
                rem_gt_boxes = np.array([gts[gid] for gid in rem_gt_ids], dtype=np.float32)
                rem_pred_boxes = np.array([preds[pid] for pid in rem_pred_ids], dtype=np.float32)
                ious = box_iou_matrix(rem_gt_boxes, rem_pred_boxes)
                r_ind, c_ind = linear_sum_assignment(-ious)
                for r, c in zip(r_ind, c_ind):
                    if ious[r, c] >= iou_threshold:
                        gid = rem_gt_ids[r]
                        pid = rem_pred_ids[c]
                        curr_frame_matches[gid] = pid

        # Passo 3: Identificar switches e fragmentações
        for gid in gts.keys():
            if gid in curr_frame_matches:
                curr_pid = curr_frame_matches[gid]
                if gid in last_pred:
                    if curr_pid != last_pred[gid]:
                        id_switches += 1
                    if f > last_frame[gid] + 1:
                        fragmentations += 1
                last_pred[gid] = curr_pid
                last_frame[gid] = f

        prev_frame_matches = curr_frame_matches

    return {
        "id_switches": id_switches,
        "fragmentations": fragmentations
    }


def compute_map_per_frame(gt_dict, det_dict, iou_threshold=EVAL_IOU):
    """
    Calcula Average Precision (AP) por quadro para avaliar a qualidade pura do detector:
    - Ordena detecções de forma estável por score/confiança decrescente.
    - Associa cada detecção ao GT de maior IoU; se o GT já foi casado, gera FP.
    - Invariante à ordem inicial de caixas com scores idênticos ou permutados.
    """
    all_frames = sorted(list(set(gt_dict.keys()).union(set(det_dict.keys()))))
    frame_aps = []

    for f in all_frames:
        gts = gt_dict.get(f, {})
        dets = det_dict.get(f, [])

        # Formata GTs para array (N, 4)
        if isinstance(gts, dict):
            gt_boxes = np.array([gts[k][:4] for k in gts], dtype=np.float32) if gts else np.empty((0, 4), dtype=np.float32)
        else:
            gt_boxes = np.array([g[2:6] for g in gts], dtype=np.float32) if len(gts) > 0 else np.empty((0, 4), dtype=np.float32)

        n_gt = len(gt_boxes)
        if n_gt == 0:
            continue

        if len(dets) == 0:
            frame_aps.append(0.0)
            continue

        # Extrai [x, y, w, h] e score
        boxes_list = []
        scores_list = []

        if isinstance(dets, dict):
            for k, v in dets.items():
                boxes_list.append(v[:4])
                scores_list.append(float(v[4]) if len(v) > 4 else 1.0)
        else:
            for d in dets:
                # Pode vir como [frame, id, x, y, w, h, conf] ou [x, y, w, h, conf] ou [x, y, w, h]
                if len(d) >= 7:
                    boxes_list.append(d[2:6])
                    scores_list.append(float(d[6]))
                elif len(d) == 5:
                    boxes_list.append(d[:4])
                    scores_list.append(float(d[4]))
                elif len(d) == 6:
                    boxes_list.append(d[2:6])
                    scores_list.append(1.0)
                else:
                    boxes_list.append(d[:4])
                    scores_list.append(1.0)

        det_boxes = np.asarray(boxes_list, dtype=np.float32)
        det_scores = np.asarray(scores_list, dtype=np.float32)

        # Ordenação estável decrescente por score
        order = np.argsort(-det_scores, kind="mergesort")
        det_boxes = det_boxes[order]
        det_scores = det_scores[order]

        matched_gt = set()
        tp = np.zeros(len(det_boxes), dtype=np.float32)
        fp = np.zeros(len(det_boxes), dtype=np.float32)

        ious = box_iou_matrix(det_boxes, gt_boxes)

        for i in range(len(det_boxes)):
            best_iou = 0.0
            best_gt = -1
            for j in range(n_gt):
                if ious[i, j] > best_iou:
                    best_iou = ious[i, j]
                    best_gt = j

            if best_iou >= iou_threshold and best_gt != -1:
                if best_gt not in matched_gt:
                    tp[i] = 1.0
                    matched_gt.add(best_gt)
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        prec = cum_tp / (cum_tp + cum_fp + 1e-7)
        rec = cum_tp / float(n_gt)

        # Interpolação de 11 pontos oficial VOC
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            mask = rec >= (t - 1e-6)
            p = np.max(prec[mask]) if np.any(mask) else 0.0
            ap += p / 11.0

        frame_aps.append(float(ap))

    return float(np.mean(frame_aps)) if frame_aps else 0.0


def evaluate_tracking(gt_data, pred_data, iou_threshold=EVAL_IOU):
    """
    Avaliação consolidada de uma sequência de rastreamento no protocolo unificado EVAL_IOU.
    Retorna IDF1, IDP, IDR, ID Switches, Fragmentações, contagem de IDs e id_count_error.
    """
    idf1_res = compute_idf1(gt_data, pred_data, iou_threshold=iou_threshold)
    idsw_res = compute_id_switches_and_fragmentations(gt_data, pred_data, iou_threshold=iou_threshold)

    def _get_unique(d):
        if isinstance(d, dict):
            return len({tid for f in d.values() for tid in f.keys()})
        return len(np.unique([int(r[1]) for r in d])) if len(d) > 0 else 0

    unique_gt = _get_unique(gt_data)
    unique_pred = _get_unique(pred_data)
    ratio_ids = (unique_pred / unique_gt) if unique_gt > 0 else 0.0
    switches_per_gt = (idsw_res["id_switches"] / unique_gt) if unique_gt > 0 else 0.0
    id_count_error = (unique_pred - unique_gt) / unique_gt if unique_gt > 0 else 0.0
    abs_id_count_error = abs(unique_pred - unique_gt) / unique_gt if unique_gt > 0 else 0.0

    return {
        "idf1": idf1_res["idf1"],
        "idp": idf1_res["idp"],
        "idr": idf1_res["idr"],
        "id_switches": idsw_res["id_switches"],
        "fragmentations": idsw_res["fragmentations"],
        "unique_gt_ids": unique_gt,
        "unique_pred_ids": unique_pred,
        "ratio_ids": ratio_ids,
        "switches_per_gt": switches_per_gt,
        "id_count_error": id_count_error,
        "abs_id_count_error": abs_id_count_error,
    }


def run_unit_tests(verbose: bool = True):
    """
    Testes unitários rigorosos das métricas com assert:
    (a) Predição == Ground Truth => IDF1 == 1.0 e switches == 0
    (b) Duas identidades trocadas a partir de k=11 (N=20) => IDF1 == 0.5, switches == 2, IDs pred == 2
    (c) Dois objetos, um quebrado em dois IDs no meio (N=20) => IDF1 == 0.75, switches == 1, IDs pred == 3
    (d) Objeto partido em dois IDs com gap de oclusão => Fragmentations == 1
    (e) Teste obrigatório mAP: 5 acertos (conf 0.9) + 3 FPs (conf 0.4), FPs no início e fim dão o mesmo mAP 1.0
    """
    SEP = "=" * 70
    if verbose:
        print(SEP)
        print("VALIDAÇÃO UNITÁRIA DAS MÉTRICAS DE RASTREAMENTO E DETECÇÃO")
        print(SEP)

    # (a) Predição idêntica ao GT
    gt_a, pred_a = {}, {}
    for f in range(1, 21):
        gt_a[f] = {
            1: np.array([10 + f, 10 + f, 20, 40], dtype=np.float32),
            2: np.array([80 - f, 60 - f, 20, 40], dtype=np.float32),
        }
        pred_a[f] = {k: v.copy() for k, v in gt_a[f].items()}

    res_a = evaluate_tracking(gt_a, pred_a, iou_threshold=0.5)
    assert np.isclose(res_a["idf1"], 1.0), f"Esperado IDF1=1.0, obtido {res_a['idf1']}"
    assert res_a["id_switches"] == 0, f"Esperado 0 switches, obtido {res_a['id_switches']}"
    assert res_a["fragmentations"] == 0, f"Esperado 0 frag, obtido {res_a['fragmentations']}"
    if verbose:
        print("[PASS] Caso (a) — Predição == GT: IDF1=1.0000, IDSW=0")

    # (b) Dois objetos trocados em k=11 (N=20)
    gt_b = {f: {k: v.copy() for k, v in gt_a[f].items()} for f in gt_a}
    pred_b = {}
    for f in range(1, 21):
        if f < 11:
            pred_b[f] = {1: gt_b[f][1].copy(), 2: gt_b[f][2].copy()}
        else:
            pred_b[f] = {1: gt_b[f][2].copy(), 2: gt_b[f][1].copy()}

    res_b = evaluate_tracking(gt_b, pred_b, iou_threshold=0.5)
    assert np.isclose(res_b["idf1"], 0.5, atol=1e-3), f"Esperado IDF1=0.5, obtido {res_b['idf1']}"
    assert res_b["id_switches"] == 2, f"Esperado 2 switches, obtido {res_b['id_switches']}"
    assert res_b["unique_pred_ids"] == 2, f"Esperado 2 IDs pred, obtido {res_b['unique_pred_ids']}"
    if verbose:
        print("[PASS] Caso (b) — 2 objetos trocados: IDF1=0.5000, IDSW=2, Pred IDs=2")

    # (c) Dois objetos, um quebrado em dois IDs no meio (N=20)
    gt_c = {f: {k: v.copy() for k, v in gt_a[f].items()} for f in gt_a}
    pred_c = {}
    for f in range(1, 21):
        if f < 11:
            pred_c[f] = {1: gt_c[f][1].copy(), 2: gt_c[f][2].copy()}
        else:
            pred_c[f] = {1: gt_c[f][1].copy(), 3: gt_c[f][2].copy()}

    res_c = evaluate_tracking(gt_c, pred_c, iou_threshold=0.5)
    assert np.isclose(res_c["idf1"], 0.75, atol=1e-3), f"Esperado IDF1=0.75, obtido {res_c['idf1']}"
    assert res_c["id_switches"] == 1, f"Esperado 1 switch, obtido {res_c['id_switches']}"
    assert res_c["unique_pred_ids"] == 3, f"Esperado 3 IDs pred, obtido {res_c['unique_pred_ids']}"
    if verbose:
        print("[PASS] Caso (c) — 2 objetos, 1 quebrado ao meio: IDF1=0.7500, IDSW=1, Pred IDs=3")

    # (d) Objeto partido em dois IDs com gap de oclusão
    gt_d = {f: {1: np.array([20 + f, 30 + f, 20, 40], dtype=np.float32)} for f in range(1, 21)}
    pred_d = {}
    for f in range(1, 9):
        pred_d[f] = {1: gt_d[f][1].copy()}
    # frames 9, 10, 11 sem detecção (gap de oclusão)
    for f in range(12, 21):
        pred_d[f] = {2: gt_d[f][1].copy()}

    res_d = evaluate_tracking(gt_d, pred_d, iou_threshold=0.5)
    assert res_d["fragmentations"] == 1, f"Esperado Frag=1, obtido {res_d['fragmentations']}"
    assert res_d["id_switches"] == 1, f"Esperado IDSW=1, obtido {res_d['id_switches']}"
    if verbose:
        print(f"[PASS] Caso (d) — Objeto partido com oclusão: Frag={res_d['fragmentations']}, IDSW={res_d['id_switches']}")

    # (e) Teste obrigatório de mAP (P0-6): 5 acertos (conf 0.9) + 3 FPs (conf 0.4)
    gt_map = {
        1: {
            1: np.array([10, 10, 20, 20], dtype=np.float32),
            2: np.array([40, 10, 20, 20], dtype=np.float32),
            3: np.array([70, 10, 20, 20], dtype=np.float32),
            4: np.array([100, 10, 20, 20], dtype=np.float32),
            5: np.array([130, 10, 20, 20], dtype=np.float32),
        }
    }
    tps_correct = [
        [10, 10, 20, 20, 0.9],
        [40, 10, 20, 20, 0.9],
        [70, 10, 20, 20, 0.9],
        [100, 10, 20, 20, 0.9],
        [130, 10, 20, 20, 0.9],
    ]
    fps_noise = [
        [200, 200, 20, 20, 0.4],
        [230, 200, 20, 20, 0.4],
        [260, 200, 20, 20, 0.4],
    ]

    # Teste 1: FPs no início da lista
    det_map_fps_first = {1: fps_noise + tps_correct}
    map1 = compute_map_per_frame(gt_map, det_map_fps_first, iou_threshold=0.5)

    # Teste 2: FPs no final da lista
    det_map_fps_last = {1: tps_correct + fps_noise}
    map2 = compute_map_per_frame(gt_map, det_map_fps_last, iou_threshold=0.5)

    assert np.isclose(map1, 1.0), f"Esperado mAP=1.0 com FPs no início, obtido {map1}"
    assert np.isclose(map2, 1.0), f"Esperado mAP=1.0 com FPs no fim, obtido {map2}"
    assert np.isclose(map1, map2), f"mAP não é invariante à ordem: {map1} vs {map2}"
    if verbose:
        print(f"[PASS] Caso (e) — mAP independente da ordem das caixas: map1={map1:.4f}, map2={map2:.4f}")
        print(SEP)

    return True


if __name__ == "__main__":
    run_unit_tests(verbose=True)
