"""
metrics.py (Raiz do projeto)
Implementação própria de IDF1, ID switches e fragmentações conforme requisitos do PA2.
"""

from src.metrics import (
    calculate_iou,
    box_iou_matrix,
    custom_nms,
    compute_idf1,
    compute_id_switches_and_fragmentations,
    compute_map_per_frame,
    evaluate_tracking
)

__all__ = [
    "calculate_iou",
    "box_iou_matrix",
    "custom_nms",
    "compute_idf1",
    "compute_id_switches_and_fragmentations",
    "compute_map_per_frame",
    "evaluate_tracking"
]

if __name__ == "__main__":
    print("Módulo de métricas carregado com sucesso.")
