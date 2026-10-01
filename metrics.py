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
    evaluate_tracking,
    run_unit_tests
)

__all__ = [
    "calculate_iou",
    "box_iou_matrix",
    "custom_nms",
    "compute_idf1",
    "compute_id_switches_and_fragmentations",
    "compute_map_per_frame",
    "evaluate_tracking",
    "run_unit_tests"
]

if __name__ == "__main__":
    print("Executando validação unitária das métricas de rastreamento (Parte 0 - Item 3)...")
    success = run_unit_tests(verbose=True)
    if not success:
        exit(1)
    print("\nImplementação de métricas validada com sucesso.")

