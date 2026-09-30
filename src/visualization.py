"""
src/visualization.py
Todas as funções de visualização e gráficos do PA2.
O notebook apenas chama essas funções + plt.show().
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ===========================================================================
# PARTE 0 — Visualizações do gerador sintético
# ===========================================================================

def plot_synthetic_frames(
    frames: np.ndarray,
    gt_by_frame: dict,
    title: str = "Gerador Sintético",
    n_frames: int = 5,
    frame_indices: list = None,
    box_color: str = "lime",
):
    """
    Parte 0 (Item 1) — Exibe n_frames do vídeo sintético com as bounding boxes do GT.

    Parâmetros
    ----------
    frames        : array (T, H, W, 3)
    gt_by_frame   : {frame_id: {obj_id: [x, y, w, h]}}
    title         : título geral da figura
    n_frames      : quantos frames exibir (ignorado se frame_indices for fornecido)
    frame_indices : índices 0-based dos frames a exibir; se None, distribui uniformemente
    box_color     : cor das bounding boxes do GT
    """
    T = len(frames)
    if frame_indices is None:
        step = max(1, T // n_frames)
        frame_indices = list(range(0, min(T, n_frames * step), step))[:n_frames]

    fig, axes = plt.subplots(1, len(frame_indices), figsize=(3 * len(frame_indices), 3.5))
    if len(frame_indices) == 1:
        axes = [axes]

    fig.suptitle(title, fontweight="bold", fontsize=11)

    for ax, idx in zip(axes, frame_indices):
        frame_id = idx + 1
        ax.imshow(frames[idx])
        for obj_id, box in gt_by_frame.get(frame_id, {}).items():
            x, y, w, h = box[:4]
            ax.add_patch(plt.Rectangle((x, y), w, h, lw=1.5,
                                        edgecolor=box_color, facecolor="none"))
            ax.text(x, y - 2, str(obj_id), color=box_color, fontsize=7, fontweight="bold")
        ax.set_title(f"Frame {frame_id}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_breakdown_curve(
    synth_results: list,
    save_path: str = None,
):
    """
    Parte 0 (Item 4) — Curva de quebra: mAP vs IDF1 em degradação progressiva
    no cenário sintético. Evidencia o descolamento (IDF1 colapsa antes do mAP).

    Parâmetros
    ----------
    synth_results : lista retornada por run_synthetic_breakdown()
    save_path     : caminho para salvar a figura (None = não salva)
    """
    labels = [r["label"] for r in synth_results]
    maps   = [r["map_score"] for r in synth_results]
    idf1s  = [r["idf1"] for r in synth_results]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, maps,  "o-", color="#2b5c8f", lw=2, label="mAP (Detecção)")
    ax.plot(x, idf1s, "s-", color="#d95f02", lw=2, label="IDF1 (Tracking)")
    ax.fill_between(x, idf1s, maps, alpha=0.12, color="purple", label="Descolamento")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Parte 0 — Curva de Quebra: mAP vs IDF1 no Cenário Sintético\n"
        "(o IDF1 colapsa muito antes do mAP, evidenciando o descolamento)",
        fontsize=11, fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo em {save_path}")

    return fig


# ===========================================================================
# PARTE 1 — Visualizações das duas fontes e painel de descolamento
# ===========================================================================

def plot_source_comparison(
    comparison: dict,
    frame_idx: int = 4,
    save_path: str = None,
):
    """
    Parte 1 (Item 1) — Visualiza as duas fontes de detecção side-by-side:
    GT (referência), Fonte 1 (SDP-like) e Fonte 2 (Faster R-CNN).

    Parâmetros
    ----------
    comparison : dict retornado por compare_detection_sources()
    frame_idx  : índice 0-based do frame a exibir
    save_path  : caminho para salvar a figura (None = não salva)
    """
    frames     = comparison["frames"]
    gt         = comparison["gt_by_frame"]
    det_fonte1 = comparison["det_fonte1"]
    det_fonte2 = comparison["det_fonte2"]

    frame_id = frame_idx + 1

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, boxes, color) in zip(axes, [
        ("Ground Truth\n(referência)", gt.get(frame_id, {}), "lime"),
        ("Fonte 1: SDP-like\n(det. simulado s/ ruído)", det_fonte1.get(frame_id, []), "deepskyblue"),
        ("Fonte 2: Faster R-CNN\n(torchvision COCO)", det_fonte2.get(frame_id, []), "red"),
    ]):
        ax.imshow(frames[frame_idx])
        items = list(boxes.values()) if isinstance(boxes, dict) else [b[:4] for b in boxes]
        for box in items:
            x, y, w, h = box[:4]
            ax.add_patch(plt.Rectangle((x, y), w, h, lw=2, edgecolor=color, facecolor="none"))
        ax.set_title(f"{title}\n({len(items)} caixas)", fontsize=9, fontweight="bold")
        ax.axis("off")

    patches = [
        mpatches.Patch(color="lime",        label="GT (referência)"),
        mpatches.Patch(color="deepskyblue", label="Fonte 1 (SDP-like)"),
        mpatches.Patch(color="red",         label="Fonte 2 (Faster R-CNN)"),
    ]
    plt.legend(handles=patches, loc="lower right", fontsize=8)
    plt.suptitle(
        "Parte 1 — Item 1: Comparação das Duas Fontes de Detecção\n"
        "(Fonte 2 detecta 0 pessoas em elipses — gap de domínio esperado)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Figura salva em {save_path}")

    return fig


def plot_decoupling_panel(
    synth_results: list,
    mot17_results: list,
    save_path: str = None,
):
    """
    Parte 1 (Item 5) — Painel duplo obrigatório do descolamento:
    - Esquerdo : curva de quebra no cenário sintético (Parte 0 → Parte 1)
    - Direito  : barras do MOT17 real, sequências ordenadas por densidade

    Parâmetros
    ----------
    synth_results : lista retornada por run_synthetic_table()
    mot17_results : lista retornada por run_mot17_baseline()
    save_path     : caminho para salvar a figura (None = não salva)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # --- Painel esquerdo: curva sintética ---
    labels_s = [r["label"] for r in synth_results]
    maps_s   = [r["map_score"] for r in synth_results]
    idf1s_s  = [r["idf1"] for r in synth_results]
    x_s = np.arange(len(labels_s))

    axes[0].plot(x_s, maps_s,  "o-", color="#2b5c8f", lw=2, label="mAP (Detecção)")
    axes[0].plot(x_s, idf1s_s, "s-", color="#d95f02", lw=2, label="IDF1 (Tracking)")
    axes[0].fill_between(x_s, idf1s_s, maps_s, alpha=0.12, color="purple", label="Descolamento")
    axes[0].set_xticks(x_s)
    axes[0].set_xticklabels(labels_s)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(
        "Cenário Sintético (Parte 0 → Parte 1)\nmAP cai junto com IDF1 — validação controlada",
        fontweight="bold",
    )
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.4)

    # --- Painel direito: MOT17 real, barras por sequência ---
    sorted_r = sorted(mot17_results, key=lambda r: r["density"])
    labels_m = [f"{r['seq_name']}\n(d={r['density']:.0f})" for r in sorted_r]
    maps_m   = [r["map_score"] for r in sorted_r]
    idf1s_m  = [r["idf1"] for r in sorted_r]
    x_m = np.arange(len(labels_m))
    w = 0.35

    axes[1].bar(x_m - w/2, maps_m,  w, label="mAP (Det)",   color="#2b5c8f", alpha=0.85)
    axes[1].bar(x_m + w/2, idf1s_m, w, label="IDF1",        color="#d95f02", alpha=0.85)
    axes[1].set_xticks(x_m)
    axes[1].set_xticklabels(labels_m, fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(
        "MOT17 Real — SDP (Parte 1)\nmAP alto, IDF1 colapsa com densidade",
        fontweight="bold",
    )
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.4)
    for i, (mp, id1) in enumerate(zip(maps_m, idf1s_m)):
        axes[1].text(x_m[i] - w/2, mp  + 0.01, f"{mp:.2f}",  ha="center", fontsize=7)
        axes[1].text(x_m[i] + w/2, id1 + 0.01, f"{id1:.2f}", ha="center", fontsize=7, color="#a53600")

    plt.suptitle(
        "Parte 1 — Painel do Descolamento: Qualidade da Detecção vs Consistência Temporal",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Gráfico salvo em {save_path}")

    return fig
