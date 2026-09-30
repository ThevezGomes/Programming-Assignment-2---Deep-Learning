"""
src/metrics.py
Implementação de métricas de tracking e NMS de autoria própria:
- custom_nms: Implementação própria sem usar torchvision.ops.nms
- calculate_iou e box_iou_matrix: Cálculo de IoU entre caixas [x, y, w, h]
- compute_idf1: IDF1 oficial (Ristani et al., 2016) via Hungarian matching global
- compute_id_switches_and_fragmentations: Contagem explícita de switches e fragmentações
- compute_map_per_frame: mAP de detecção por quadro para o gráfico de descolamento
- evaluate_tracking: Avaliação completa de trajetórias
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_iou(box1, box2):
    """
    Calcula IoU entre duas caixas no formato [x, y, w, h] (top-left, width, height).
    """
    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]

    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2]) * max(0.0, box1[3])
    area2 = max(0.0, box2[2]) * max(0.0, box2[3])

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

    area_a = boxes_a[:, 2] * boxes_a[:, 3]
    area_b = boxes_b[:, 2] * boxes_b[:, 3]
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    iou = np.where(union_area > 0, inter_area / np.maximum(union_area, 1e-7), 0.0)
    return iou


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

    order = scores.argsort()[::-1]
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


def compute_idf1(gt_data, pred_data, iou_threshold=0.5):
    """
    Calcula IDF1 oficial (Ristani et al., 2016) via Hungarian matching global.
    gt_data: dict {frame: {id: [x, y, w, h]}} ou list/array de [frame, id, x, y, w, h]
    pred_data: dict {frame: {id: [x, y, w, h]}} ou list/array de [frame, id, x, y, w, h]
    """
    def _to_dict(d):
        if isinstance(d, dict):
            return d
        out = {}
        for row in d:
            f, tid = int(row[0]), int(row[1])
            box = np.asarray(row[2:6], dtype=np.float32)
            if f not in out:
                out[f] = {}
            out[f][tid] = box
        return out

    gt_dict = _to_dict(gt_data)
    pred_dict = _to_dict(pred_data)

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

    # Matriz de sobreposição C[i, j] = frames onde GT i casa com Pred j com IoU >= iou_threshold
    cost_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)

    all_frames = set(gt_dict.keys()).union(set(pred_dict.keys()))
    for f in all_frames:
        gts = gt_dict.get(f, {})
        preds = pred_dict.get(f, {})
        if not gts or not preds:
            continue

        f_gt_ids = list(gts.keys())
        f_pred_ids = list(preds.keys())
        f_gt_boxes = np.array([gts[gid] for gid in f_gt_ids])
        f_pred_boxes = np.array([preds[pid] for pid in f_pred_ids])

        ious = box_iou_matrix(f_gt_boxes, f_pred_boxes)
        r_ind, c_ind = linear_sum_assignment(-ious)
        for r, c in zip(r_ind, c_ind):
            if ious[r, c] >= iou_threshold:
                gid = f_gt_ids[r]
                pid = f_pred_ids[c]
                cost_matrix[gt_to_idx[gid], pred_to_idx[pid]] += 1.0

    # Matching global entre trajetórias
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


def compute_id_switches_and_fragmentations(gt_data, pred_data, iou_threshold=0.5):
    """
    Contagem de trocas de identidade (ID switches) e fragmentações (quebras na track).
    """
    def _to_dict(d):
        if isinstance(d, dict):
            return d
        out = {}
        for row in d:
            f, tid = int(row[0]), int(row[1])
            box = np.asarray(row[2:6], dtype=np.float32)
            if f not in out:
                out[f] = {}
            out[f][tid] = box
        return out

    gt_dict = _to_dict(gt_data)
    pred_dict = _to_dict(pred_data)

    all_frames = sorted(list(set(gt_dict.keys()).union(set(pred_dict.keys()))))

    last_pred = {}
    last_frame = {}
    id_switches = 0
    fragmentations = 0

    for f in all_frames:
        gts = gt_dict.get(f, {})
        preds = pred_dict.get(f, {})
        f_matches = {}

        if gts and preds:
            f_gt_ids = list(gts.keys())
            f_pred_ids = list(preds.keys())
            f_gt_boxes = np.array([gts[gid] for gid in f_gt_ids])
            f_pred_boxes = np.array([preds[pid] for pid in f_pred_ids])

            ious = box_iou_matrix(f_gt_boxes, f_pred_boxes)
            r_ind, c_ind = linear_sum_assignment(-ious)
            for r, c in zip(r_ind, c_ind):
                if ious[r, c] >= iou_threshold:
                    f_matches[f_gt_ids[r]] = f_pred_ids[c]

        for gid in gts.keys():
            if gid in f_matches:
                curr_pred = f_matches[gid]
                if gid in last_pred:
                    if curr_pred != last_pred[gid]:
                        id_switches += 1
                    if f > last_frame[gid] + 1:
                        fragmentations += 1
                last_pred[gid] = curr_pred
                last_frame[gid] = f

    return {
        "id_switches": id_switches,
        "fragmentations": fragmentations
    }


def compute_map_per_frame(gt_dict, det_dict, iou_threshold=0.5):
    """
    Calcula Average Precision (AP) por quadro para avaliar a qualidade pura do detector
    (usado no Painel de Descolamento da Parte 1).
    """
    all_frames = sorted(list(set(gt_dict.keys()).union(set(det_dict.keys()))))
    frame_aps = []

    for f in all_frames:
        gts = gt_dict.get(f, {})
        dets = det_dict.get(f, [])

        n_gt = len(gts)
        if n_gt == 0:
            continue

        if len(dets) == 0:
            frame_aps.append(0.0)
            continue

        # dets: list of (box, score) ordenados por score decrescente
        if isinstance(dets, dict):
            det_boxes = [dets[k] for k in dets]
            det_scores = [1.0] * len(det_boxes)
        elif len(dets) > 0 and len(dets[0]) >= 6:
            # Formato [frame, id, x, y, w, h, conf]
            det_boxes = [d[2:6] for d in dets]
            det_scores = [d[6] if len(d) > 6 else 1.0 for d in dets]
        else:
            det_boxes = [d[:4] for d in dets]
            det_scores = [1.0] * len(det_boxes)

        order = np.argsort(det_scores)[::-1]
        det_boxes = np.array(det_boxes)[order]

        gt_boxes = np.array(list(gts.values()))
        matched_gt = set()

        tp = np.zeros(len(det_boxes))
        fp = np.zeros(len(det_boxes))

        ious = box_iou_matrix(det_boxes, gt_boxes)
        for i in range(len(det_boxes)):
            best_iou = 0.0
            best_gt = -1
            for j in range(len(gt_boxes)):
                if j not in matched_gt and ious[i, j] > best_iou:
                    best_iou = ious[i, j]
                    best_gt = j
            if best_iou >= iou_threshold and best_gt != -1:
                tp[i] = 1.0
                matched_gt.add(best_gt)
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        prec = cum_tp / (cum_tp + cum_fp + 1e-7)
        rec = cum_tp / n_gt

        # 11-point interpolation or area under curve
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = np.max(prec[rec >= t]) if np.sum(rec >= t) > 0 else 0.0
            ap += p / 11.0
        frame_aps.append(ap)

    return float(np.mean(frame_aps)) if frame_aps else 0.0


def evaluate_tracking(gt_data, pred_data, iou_threshold=0.5):
    """
    Avaliação consolidada de uma sequência de rastreamento.
    """
    idf1_res = compute_idf1(gt_data, pred_data, iou_threshold)
    idsw_res = compute_id_switches_and_fragmentations(gt_data, pred_data, iou_threshold)

    def _get_unique(d):
        if isinstance(d, dict):
            return len({tid for f in d.values() for tid in f.keys()})
        return len(np.unique([int(r[1]) for r in d])) if len(d) > 0 else 0

    unique_gt = _get_unique(gt_data)
    unique_pred = _get_unique(pred_data)
    ratio_ids = (unique_pred / unique_gt) if unique_gt > 0 else 0.0
    switches_per_gt = (idsw_res["id_switches"] / unique_gt) if unique_gt > 0 else 0.0

    return {
        "idf1": idf1_res["idf1"],
        "idp": idf1_res["idp"],
        "idr": idf1_res["idr"],
        "id_switches": idsw_res["id_switches"],
        "fragmentations": idsw_res["fragmentations"],
        "unique_gt_ids": unique_gt,
        "unique_pred_ids": unique_pred,
        "ratio_ids": ratio_ids,
        "switches_per_gt": switches_per_gt
    }


def run_unit_tests(verbose: bool = True):
    """
    Parte 0 (Item 3) - Testes unitarios construidos a mao:

    (a) Predicao == Ground Truth  =>  IDF1 = 1.0  e  ID Switches = 0
    (b) Duas identidades trocadas a partir do quadro k=11  =>  2 switches, IDF1 = 0.5
    (c) Uma track partida em dois IDs distintos (objeto fragmentado)
        =>  1 switch, IDF1 = 0.5, unique_pred_ids = 2 (GT tem 1)

    Nota: (b) e (c) tem IDF1 igual mas causas diferentes:
    (b) confusao entre dois objetos => contagem correta, identidades trocadas
    (c) fragmentacao de um objeto   => contagem inflada, um objeto virou dois IDs
    """
    SEP = "=" * 65
    if verbose:
        print(SEP)
        print("PARTE 0 (Item 3) - TESTES UNITARIOS DAS METRICAS")
        print(SEP)

    all_passed = True

    # --- CASO (a): Predicao identica ao GT ---
    gt_a, pred_a = {}, {}
    for f in range(1, 21):
        gt_a[f] = {
            1: np.array([10 + f, 10 + f, 20, 40], dtype=np.float32),
            2: np.array([80 - f, 60 - f, 20, 40], dtype=np.float32),
        }
        pred_a[f] = {k: v.copy() for k, v in gt_a[f].items()}

    res_a = evaluate_tracking(gt_a, pred_a)
    ok_a = np.isclose(res_a["idf1"], 1.0) and res_a["id_switches"] == 0
    all_passed = all_passed and ok_a
    if verbose:
        print(f"\n{'[PASS]' if ok_a else '[FAIL]'} Caso (a) -- Predicao == Ground Truth")
        print(f"       IDF1        = {res_a['idf1']:.4f}  (esperado: 1.0000)")
        print(f"       ID Switches = {res_a['id_switches']}  (esperado: 0)")

    # --- CASO (b): Duas identidades trocadas a partir do frame k=11 ---
    gt_b = {f: {k: v.copy() for k, v in gt_a[f].items()} for f in gt_a}
    pred_b = {}
    for f in range(1, 21):
        if f < 11:
            pred_b[f] = {k: v.copy() for k, v in gt_a[f].items()}
        else:
            pred_b[f] = {1: gt_a[f][2].copy(), 2: gt_a[f][1].copy()}

    res_b = evaluate_tracking(gt_b, pred_b)
    ok_b = np.isclose(res_b["idf1"], 0.5, atol=0.01) and res_b["id_switches"] == 2
    all_passed = all_passed and ok_b
    if verbose:
        print(f"\n{'[PASS]' if ok_b else '[FAIL]'} Caso (b) -- Troca de 2 identidades a partir de k=11")
        print(f"       IDF1        = {res_b['idf1']:.4f}  (esperado: ~0.5000)")
        print(f"       ID Switches = {res_b['id_switches']}  (esperado: 2)")
        print("       DIAGNOSTICO: tracker CONFUNDIU identidades dos dois objetos")

    # --- CASO (c): Track unica partida em dois IDs ---
    gt_c = {f: {1: np.array([10 + f, 20 + f, 20, 30], dtype=np.float32)}
            for f in range(1, 21)}
    pred_c = {}
    for f in range(1, 11):
        pred_c[f] = {1: gt_c[f][1].copy()}
    for f in range(11, 21):
        pred_c[f] = {2: gt_c[f][1].copy()}

    res_c = evaluate_tracking(gt_c, pred_c)
    ok_c = (np.isclose(res_c["idf1"], 0.5, atol=0.01)
            and res_c["id_switches"] == 1
            and res_c["unique_pred_ids"] == 2
            and res_c["unique_gt_ids"] == 1)
    all_passed = all_passed and ok_c
    if verbose:
        print(f"\n{'[PASS]' if ok_c else '[FAIL]'} Caso (c) -- Track unica fragmentada em 2 IDs")
        print(f"       IDF1           = {res_c['idf1']:.4f}  (esperado: ~0.5000)")
        print(f"       ID Switches    = {res_c['id_switches']}  (esperado: 1)")
        print(f"       IDs GT / Pred  = {res_c['unique_gt_ids']} GT / {res_c['unique_pred_ids']} Pred")
        print("       DIAGNOSTICO: contagem INFLADA (2 IDs pred para 1 objeto real)")
        print("\n       Diferenca entre (b) e (c):")
        print("       (b) 2 objetos trocados   => switches=2, qtd de IDs correta")
        print("       (c) 1 objeto fragmentado => switches=1, qtd de IDs inflada")

    if verbose:
        print(f"\n{SEP}")
        print("TODOS OS TESTES APROVADOS!" if all_passed else "ATENCAO: ALGUM TESTE FALHOU!")
        print(SEP)

    return all_passed
