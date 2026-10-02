"""
tests/test_metrics.py
Bateria de testes unitários com pytest para validação de métricas (P1-7).
"""

import numpy as np
import pytest
from src.metrics import (
    evaluate_tracking,
    compute_idf1,
    compute_id_switches_and_fragmentations,
    compute_map_per_frame,
    calculate_iou,
    custom_nms
)


def test_caso_a_predicao_identica_ao_gt():
    """(a) pred = GT: IDF1 == 1, IDSW == 0, Frag == 0."""
    gt, pred = {}, {}
    for f in range(1, 21):
        gt[f] = {
            1: np.array([10 + f, 10 + f, 20, 40], dtype=np.float32),
            2: np.array([80 - f, 60 - f, 20, 40], dtype=np.float32),
        }
        pred[f] = {k: v.copy() for k, v in gt[f].items()}

    res = evaluate_tracking(gt, pred, iou_threshold=0.5)
    assert np.isclose(res["idf1"], 1.0)
    assert res["id_switches"] == 0
    assert res["fragmentations"] == 0
    assert res["unique_pred_ids"] == 2
    assert res["unique_gt_ids"] == 2


def test_caso_b_dois_objetos_trocados_k11():
    """(b) 2 objetos trocados em k=11 (N=20): IDF1 == 0.5, IDSW == 2, IDs previstos == 2."""
    gt, pred = {}, {}
    for f in range(1, 21):
        gt[f] = {
            1: np.array([10 + f, 10 + f, 20, 40], dtype=np.float32),
            2: np.array([80 - f, 60 - f, 20, 40], dtype=np.float32),
        }
        if f < 11:
            pred[f] = {1: gt[f][1].copy(), 2: gt[f][2].copy()}
        else:
            pred[f] = {1: gt[f][2].copy(), 2: gt[f][1].copy()}

    res = evaluate_tracking(gt, pred, iou_threshold=0.5)
    assert np.isclose(res["idf1"], 0.5, atol=1e-3)
    assert res["id_switches"] == 2
    assert res["unique_pred_ids"] == 2
    assert res["unique_gt_ids"] == 2


def test_caso_c_dois_objetos_um_quebrado_ao_meio():
    """(c) 2 objetos, um deles quebrado em dois IDs no meio (N=20): IDF1 == 0.75, IDSW == 1, IDs previstos == 3."""
    gt, pred = {}, {}
    for f in range(1, 21):
        gt[f] = {
            1: np.array([10 + f, 10 + f, 20, 40], dtype=np.float32),
            2: np.array([80 - f, 60 - f, 20, 40], dtype=np.float32),
        }
        if f < 11:
            pred[f] = {1: gt[f][1].copy(), 2: gt[f][2].copy()}
        else:
            pred[f] = {1: gt[f][1].copy(), 3: gt[f][2].copy()}

    res = evaluate_tracking(gt, pred, iou_threshold=0.5)
    assert np.isclose(res["idf1"], 0.75, atol=1e-3)
    assert res["id_switches"] == 1
    assert res["unique_pred_ids"] == 3
    assert res["unique_gt_ids"] == 2


def test_caso_d_objeto_partido_com_buraco_oclusao():
    """(d) objeto partido em dois IDs com buraco: Frag == 1."""
    gt, pred = {}, {}
    for f in range(1, 21):
        gt[f] = {1: np.array([20 + f, 30 + f, 20, 40], dtype=np.float32)}

    for f in range(1, 9):
        pred[f] = {1: gt[f][1].copy()}
    # frames 9, 10, 11: gap de oclusão
    for f in range(12, 21):
        pred[f] = {2: gt[f][1].copy()}

    res = evaluate_tracking(gt, pred, iou_threshold=0.5)
    assert res["fragmentations"] == 1
    assert res["id_switches"] == 1
    assert res["unique_pred_ids"] == 2
    assert res["unique_gt_ids"] == 1


def test_map_invariancia_de_ordem_e_confianca():
    """(e) Teste obrigatório mAP (P0-6): 5 acertos (conf 0.9) + 3 FPs (conf 0.4), mAP == 1.0 independente da ordem."""
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

    det_fps_first = {1: fps_noise + tps_correct}
    det_fps_last = {1: tps_correct + fps_noise}

    map1 = compute_map_per_frame(gt_map, det_fps_first, iou_threshold=0.5)
    map2 = compute_map_per_frame(gt_map, det_fps_last, iou_threshold=0.5)

    assert np.isclose(map1, 1.0)
    assert np.isclose(map2, 1.0)
    assert np.isclose(map1, map2)
