"""
src/visualization.py
Visualizações e gráficos exigidos pelo PA2:
- plot_decoupling_panel: Painel duplo obrigatório do descolamento (Parte 1, Item 5)
- plot_gradient_vanishing_curves: Curva analítica de ||dL_t / dh_{t-k}|| (Parte 4)
- render_tracking_preview: Visualização estática ou em tira de frames
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_decoupling_panel(results_list: list, save_path: str = None):
    """
    Parte 1 (Item 5) — Gráfico obrigatório do descolamento em dois painéis:
    - Painel Superior: mAP por quadro vs IDF1
    - Painel Inferior: IDs previstas / verdadeiras vs ID switches por identidade verdadeira
    
    As sequências são ordenadas pelo eixo de dificuldade escolhido: densidade de pedestres.
    """
    # Ordenar por densidade crescente
    sorted_results = sorted(results_list, key=lambda x: x["density"])

    seq_labels = [f"{r['seq_name']}\n(dens={r['density']:.1f})" for r in sorted_results]
    maps = [r["map_score"] for r in sorted_results]
    idf1s = [r["idf1"] for r in sorted_results]
    ratio_ids = [r["ratio_ids"] for r in sorted_results]
    switches_per_gt = [r["switches_per_gt"] for r in sorted_results]

    x = np.arange(len(seq_labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # --- Painel Superior: Detecção vs Tracking ---
    ax1.bar(x - width/2, maps, width, label="mAP por quadro (Detecção)", color="#2b5c8f", alpha=0.85)
    ax1.bar(x + width/2, idf1s, width, label="IDF1 (Tracking temporal)", color="#d95f02", alpha=0.85)
    ax1.set_ylabel("Score (0 a 1)", fontsize=11, fontweight="bold")
    ax1.set_title("Descolamento Parte 1: Qualidade da Detecção (mAP) vs Consistência Temporal (IDF1)",
                  fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Anotações dos valores nas barras superiores
    for i in range(len(x)):
        ax1.text(x[i] - width/2, maps[i] + 0.02, f"{maps[i]:.2f}", ha="center", fontsize=8)
        ax1.text(x[i] + width/2, idf1s[i] + 0.02, f"{idf1s[i]:.2f}", ha="center", fontsize=8, color="#a53600")

    # --- Painel Inferior: Explosão de IDs e Switches ---
    ax2.bar(x - width/2, ratio_ids, width, label="IDs Previstos / IDs Reais", color="#7570b3", alpha=0.85)
    ax2.bar(x + width/2, switches_per_gt, width, label="ID Switches por Pedestre Real", color="#e7298a", alpha=0.85)
    ax2.axhline(1.0, color="gray", linestyle=":", label="Razão Ideal (1.0)")
    ax2.set_ylabel("Razão / Frequência", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Sequências MOT17 (Ordenadas por densidade média de pedestres/quadro)",
                   fontsize=11, fontweight="bold", labelpad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(seq_labels, fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.9)

    # Anotações no painel inferior
    for i in range(len(x)):
        ax2.text(x[i] - width/2, ratio_ids[i] + 0.05, f"{ratio_ids[i]:.1f}x", ha="center", fontsize=8)
        ax2.text(x[i] + width/2, switches_per_gt[i] + 0.05, f"{switches_per_gt[i]:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Gráfico de descolamento salvo em: {save_path}")

    return fig
