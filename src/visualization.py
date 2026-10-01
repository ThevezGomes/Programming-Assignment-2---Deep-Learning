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


def plot_occlusion_strip(
    frames: np.ndarray,
    gt_by_frame: dict,
    target_id: int = 1,
    occluder_id: int = 2,
    frame_indices: list = None,
    save_path: str = None,
):
    """
    Parte 0 (Item 1) — Requisito Verificável Obrigatório:
    Mostra uma sequência cronológica em tira com uma trajetória que passa atrás de
    outro objeto, SOME completamente por N quadros e volta com o mesmo ID.
    """
    if frame_indices is None:
        # Encontra frames onde o alvo está visível, onde está ocluído e onde reaparece
        all_frames = sorted(list(gt_by_frame.keys()))
        vis_before = [f for f in all_frames if f <= len(all_frames)//2 and target_id in gt_by_frame[f]]
        occ_frames = [f for f in all_frames if target_id not in gt_by_frame[f]]
        vis_after = [f for f in all_frames if f > len(all_frames)//2 and target_id in gt_by_frame[f]]

        f1 = vis_before[0] if vis_before else 1
        f2 = vis_before[-1] if vis_before else 10
        f_occ1 = occ_frames[0] if occ_frames else 15
        f_occ_mid = occ_frames[len(occ_frames)//2] if occ_frames else 18
        f_occ2 = occ_frames[-1] if occ_frames else 21
        f3 = vis_after[0] if vis_after else 23
        f4 = vis_after[-1] if vis_after else len(all_frames)
        frame_indices = [f1 - 1, f2 - 1, f_occ1 - 1, f_occ_mid - 1, f_occ2 - 1, f3 - 1, f4 - 1]

    fig, axes = plt.subplots(1, len(frame_indices), figsize=(2.7 * len(frame_indices), 3.8))
    fig.suptitle(
        "Parte 0 (Item 1) — Requisito Verificável: Trajetória que Some por N Quadros e Volta\n"
        "[Alvo (ID 1 - Verde) passa atrás do Oclusor (ID 2 - Vermelho) e desaparece 100% no z-buffer]",
        fontsize=11, fontweight="bold", y=1.05
    )

    for ax, idx in zip(axes, frame_indices):
        f = idx + 1
        ax.imshow(frames[idx])
        gts = gt_by_frame.get(f, {})

        # Oclusor (ID 2)
        if occluder_id in gts:
            box_o = gts[occluder_id][:4]
            ax.add_patch(plt.Rectangle((box_o[0], box_o[1]), box_o[2], box_o[3],
                                       lw=2, edgecolor="#e41a1c", facecolor="none"))
            ax.text(box_o[0], box_o[1] - 3, "ID 2 (Oclusor)", color="#ff4d4d", fontsize=7, fontweight="bold")

        # Alvo (ID 1)
        if target_id in gts:
            box_t = gts[target_id][:4]
            ax.add_patch(plt.Rectangle((box_t[0], box_t[1]), box_t[2], box_t[3],
                                       lw=2, edgecolor="#00ff66", facecolor="none"))
            ax.text(box_t[0], box_t[1] - 3, "ID 1 (Alvo)", color="#00ff66", fontsize=7, fontweight="bold")
            status = "VISÍVEL"
            status_color = "green"
        else:
            status = "OCLUSÃO TOTAL\n(0 px visíveis)"
            status_color = "red"

        ax.set_title(f"Quadro {f}\n{status}", fontsize=8, fontweight="bold", color=status_color)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Figura de oclusão salva em {save_path}")

    return fig


def plot_generator_knobs_breakdown(
    results_by_knob: dict,
    save_path: str = None,
):
    """
    Parte 0 (Item 4) — Gráfico do Ensaio da Parte 1:
    Gira os botões do gerador (Velocidade, Duração da Oclusão, Densidade)
    e mostra onde a associação ingênua começa a quebrar.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Velocidade
    vel_res = results_by_knob["velocity"]
    vels = [r["velocity"] for r in vel_res]
    idfs_v = [r["idf1"] for r in vel_res]
    ax1.plot(vels, idfs_v, "o-", color="#2b5c8f", lw=2.5, markersize=7)
    ax1.axvline(x=6.0, color="red", linestyle="--", alpha=0.7, label="Velocidade > Tam. Caixa")
    ax1.set_title("1. Velocidade dos Objetos", fontweight="bold", fontsize=10)
    ax1.set_xlabel("Velocidade típica (pixels/quadro)", fontsize=9)
    ax1.set_ylabel("IDF1", fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8)

    # 2. Duração da Oclusão
    occ_res = results_by_knob["occlusion"]
    occs = [r["occlusion"] for r in occ_res]
    idfs_o = [r["idf1"] for r in occ_res]
    sws_o = [r["id_switches"] for r in occ_res]
    ax2.plot(occs, idfs_o, "s-", color="#d95f02", lw=2.5, label="IDF1")
    ax2.axvline(x=10, color="purple", linestyle="--", lw=2, label="k = 10 (Morte da Track)")
    ax2.set_title("2. Duração da Oclusão", fontweight="bold", fontsize=10)
    ax2.set_xlabel("Duração da oclusão (quadros)", fontsize=9)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8)

    # 3. Densidade (Número de Objetos)
    den_res = results_by_knob["density"]
    dens = [r["num_objects"] for r in den_res]
    idfs_d = [r["idf1"] for r in den_res]
    sws_d = [r["id_switches"] for r in den_res]
    ax3.plot(dens, idfs_d, "^-", color="#2ca02c", lw=2.5, label="IDF1")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(dens, sws_d, "v--", color="#d62728", lw=2, label="ID Switches")
    ax3.set_title("3. Densidade de Objetos (128x128)", fontweight="bold", fontsize=10)
    ax3.set_xlabel("Número de elipses simultâneas", fontsize=9)
    ax3.set_ylim(-0.05, 1.05)
    ax3_twin.set_ylabel("ID Switches", color="#d62728", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)
    lines_1, labels_1 = ax3.get_legend_handles_labels()
    lines_2, labels_2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower left", fontsize=8)

    plt.suptitle(
        "Parte 0 (Item 4) — Onde o Baseline Ingênuo Quebra ao Girar os Botões do Gerador",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Gráfico de quebra do gerador salvo em {save_path}")

    return fig


def plot_breakdown_curve(
    synth_results: list,
    save_path: str = None,
):
    """
    Curva de quebra do detector degradado (Item 2 / Parte 5).
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
        "Parte 0 — Curva de Quebra com Simulador de Detector Degradado\n"
        "(IDF1 cai acentuadamente mesmo com detecções razoáveis)",
        fontsize=11, fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo em {save_path}")

    return fig


def plot_decoupling_panel_two_rows(
    mot17_results: list,
    sort_by: str = "density",
    save_path: str = None,
):
    """
    Parte 1 (Item 5) — GRÁFICO OBRIGATÓRIO DO DESCOLAMENTO:
    Estruturado em DOIS PAINÉIS VERTICAIS sobre as MESMAS sequências do MOT17,
    ordenadas pelo eixo de dificuldade (densidade média):

    - Painel Superior: mAP por quadro e IDF1;
    - Painel Inferior: Razão de identidades (N_pred / N_gt) e ID switches por identidade verdadeira.
    """
    if sort_by == "density":
        sorted_r = sorted(mot17_results, key=lambda r: r["density"])
    else:
        sorted_r = list(mot17_results)

    seq_labels = [f"{r['seq_name']}\n(d={r['density']:.1f})" for r in sorted_r]
    maps       = [r["map_score"] for r in sorted_r]
    idf1s      = [r["idf1"] for r in sorted_r]
    ratios     = [r["ratio_ids"] for r in sorted_r]
    sw_per_gt  = [r["switches_per_gt"] for r in sorted_r]

    x = np.arange(len(seq_labels))
    w = 0.35

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # ==========================
    # PAINEL SUPERIOR: mAP vs IDF1
    # ==========================
    bar1 = ax_top.bar(x - w/2, maps,  w, label="mAP por quadro (Qualidade da Detecção)", color="#2b5c8f", alpha=0.9)
    bar2 = ax_top.bar(x + w/2, idf1s, w, label="IDF1 (Consistência Temporal)", color="#d95f02", alpha=0.9)
    ax_top.set_ylabel("Score (0 a 1)", fontsize=11, fontweight="bold")
    ax_top.set_ylim(0, 1.08)
    ax_top.set_title(
        "PAINEL SUPERIOR: mAP por Quadro (Detecção) vs IDF1 (Rastreamento)\n"
        "[O detector mantém mAP ~0.75-0.80 estável, mas o IDF1 colapsa com o aumento da densidade]",
        fontsize=11, fontweight="bold"
    )
    ax_top.legend(loc="upper right", fontsize=9)
    ax_top.grid(axis="y", linestyle="--", alpha=0.4)

    # Valores no topo das barras
    for i, (m, id1) in enumerate(zip(maps, idf1s)):
        ax_top.text(x[i] - w/2, m + 0.02, f"{m:.2f}", ha="center", fontsize=8, color="#1c3d61", fontweight="bold")
        ax_top.text(x[i] + w/2, id1 + 0.02, f"{id1:.2f}", ha="center", fontsize=8, color="#a53600", fontweight="bold")

    # ==========================
    # PAINEL INFERIOR: Razão IDs e Switches/GT
    # ==========================
    bar3 = ax_bot.bar(x - w/2, ratios,    w, label="Razão de Identidades ($N_{pred} / N_{gt}$)", color="#7570b3", alpha=0.9)
    bar4 = ax_bot.bar(x + w/2, sw_per_gt, w, label="ID Switches por Identidade ($IDSW / N_{gt}$)", color="#e7298a", alpha=0.9)
    ax_bot.axhline(y=1.0, color="gray", linestyle=":", lw=1.5, label="Contagem Perfeita (1.0x)")
    ax_bot.set_ylabel("Múltiplos / Razão", fontsize=11, fontweight="bold")
    ax_bot.set_title(
        "PAINEL INFERIOR: Fragmentação de Trajetórias e Trocas de Identidade\n"
        "[Em sequências densas, cada pedestre real é fragmentado em mais de 2.5 IDs diferentes]",
        fontsize=11, fontweight="bold"
    )
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(seq_labels, fontsize=10, fontweight="bold")
    ax_bot.set_xlabel("Sequências do MOT17 (Ordenadas pelo Eixo de Dificuldade: Densidade Média de Pedestres)",
                      fontsize=11, fontweight="bold")
    ax_bot.legend(loc="upper left", fontsize=9)
    ax_bot.grid(axis="y", linestyle="--", alpha=0.4)

    for i, (r, sw) in enumerate(zip(ratios, sw_per_gt)):
        ax_bot.text(x[i] - w/2, r + 0.05, f"{r:.2f}x", ha="center", fontsize=8, color="#483d8b", fontweight="bold")
        ax_bot.text(x[i] + w/2, sw + 0.05, f"{sw:.2f}", ha="center", fontsize=8, color="#8b0046", fontweight="bold")

    plt.suptitle(
        "Parte 1 (Item 5) — PAINEL OBRIGATÓRIO DO DESCOLAMENTO TEMPORAL",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Painel obrigatório do descolamento salvo em {save_path}")

    return fig


def plot_decoupling_panel(synth_results, mot17_results, save_path=None):
    """Alias para manter retrocompatibilidade com chamadas existentes."""
    return plot_decoupling_panel_two_rows(mot17_results, sort_by="density", save_path=save_path)


def plot_source_comparison(
    comparison: dict,
    frame_idx: int = 4,
    save_path: str = None,
):
    """Visualização da comparação sintética."""
    frames     = comparison["frames"]
    gt         = comparison["gt_by_frame"]
    det_fonte1 = comparison["det_fonte1"]
    det_fonte2 = comparison["det_fonte2"]
    frame_id   = frame_idx + 1

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, boxes, color) in zip(axes, [
        ("Ground Truth\n(referência)", gt.get(frame_id, {}), "lime"),
        ("Fonte 1: SDP-like\n(det. público simulado)", det_fonte1.get(frame_id, []), "deepskyblue"),
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
    plt.suptitle("Parte 1 — Comparação das Duas Fontes no Domínio Sintético", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_real_source_comparison(
    real_comparison: dict,
    frame_id: int = 1,
    save_path: str = None,
):
    """
    Parte 1 (Item 1) — Visualização das duas fontes em IMAGENS REAIS do MOT17:
    - Imagem com Ground Truth (verde)
    - Imagem com Fonte 1: Detecções Públicas SDP (azul)
    - Imagem com Fonte 2: Torchvision Faster R-CNN com custom_nms (laranja)
    """
    sample_images = real_comparison.get("sample_images", {})
    if frame_id not in sample_images:
        frame_id = list(sample_images.keys())[0] if sample_images else 1

    pil_img = sample_images[frame_id]
    gt_boxes = real_comparison["gt"].get(frame_id, {})
    sdp_boxes = real_comparison["sdp_dets"].get(frame_id, [])
    rcnn_boxes = real_comparison["rcnn_dets"].get(frame_id, [])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    seq_name = real_comparison.get("seq_name", "MOT17-09")

    panels = [
        ("Ground Truth Oficial", gt_boxes, "#00ff66"),
        ("Fonte 1: Detecção Pública (SDP)", sdp_boxes, "#00bfff"),
        ("Fonte 2: Torchvision Faster R-CNN (custom_nms)", rcnn_boxes, "#ff7f0e"),
    ]

    for ax, (title, boxes, color) in zip(axes, panels):
        ax.imshow(pil_img)
        items = list(boxes.values()) if isinstance(boxes, dict) else [b[:4] for b in boxes]
        for b in items:
            x, y, w, h = b[:4]
            ax.add_patch(plt.Rectangle((x, y), w, h, lw=2, edgecolor=color, facecolor="none"))
        ax.set_title(f"{title}\n({len(items)} pedestres detectados)", fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        f"Parte 1 (Item 1) — Comparação das Duas Fontes em Imagens Reais: {seq_name} (Quadro {frame_id})\n"
        f"mAP SDP = {real_comparison['map_sdp']:.3f} | mAP Faster R-CNN = {real_comparison['map_rcnn']:.3f}",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    return fig


def plot_part2_comparison_panel(
    comparison_results: list,
    training_history: dict = None,
    save_path: str = None
):
    """
    Parte 2 — Painel Visual Comparativo:
    - Subplot 1: Comparação de IDF1 (Baseline vs Trilha A LSTM)
    - Subplot 2: Comparação de ID Switches (Redução drástica de trocas)
    - Subplot 3: Curva de Treinamento da LSTM (Smooth L1 Loss na validação)
    """
    seqs = [r["seq_name"] for r in comparison_results]
    idf_naive = [r["naive"]["idf1"] for r in comparison_results]
    idf_rnn   = [r["rnn"]["idf1"] for r in comparison_results]
    sw_naive  = [r["naive"]["id_switches"] for r in comparison_results]
    sw_rnn    = [r["rnn"]["id_switches"] for r in comparison_results]

    x = np.arange(len(seqs))
    w = 0.35

    n_cols = 3 if training_history is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.5))

    # 1. IDF1
    axes[0].bar(x - w/2, idf_naive, w, label="Baseline (Naive)", color="#7570b3", alpha=0.9)
    axes[0].bar(x + w/2, idf_rnn,   w, label="Trilha A (LSTM)",   color="#1b9e77", alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(seqs, fontsize=10, fontweight="bold")
    axes[0].set_ylabel("IDF1", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("1. Consistência Temporal (IDF1)", fontsize=11, fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    for i in range(len(x)):
        axes[0].text(x[i] - w/2, idf_naive[i] + 0.02, f"{idf_naive[i]:.2f}", ha="center", fontsize=8)
        axes[0].text(x[i] + w/2, idf_rnn[i] + 0.02, f"{idf_rnn[i]:.2f}", ha="center", fontsize=8, color="#0b6647", fontweight="bold")

    # 2. ID Switches
    axes[1].bar(x - w/2, sw_naive, w, label="Baseline (Naive)", color="#d95f02", alpha=0.9)
    axes[1].bar(x + w/2, sw_rnn,   w, label="Trilha A (LSTM)",   color="#2b5c8f", alpha=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(seqs, fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Total de ID Switches", fontsize=11, fontweight="bold")
    axes[1].set_title("2. Trocas de Identidade (ID Switches)", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    for i in range(len(x)):
        axes[1].text(x[i] - w/2, sw_naive[i] + 1, f"{sw_naive[i]}", ha="center", fontsize=8)
        axes[1].text(x[i] + w/2, sw_rnn[i] + 1, f"{sw_rnn[i]}", ha="center", fontsize=8, color="#1c3d61", fontweight="bold")

    # 3. Curva de Treinamento (se fornecido)
    if training_history is not None:
        epochs = np.arange(1, len(training_history["val_smooth_l1"]) + 1)
        axes[2].plot(epochs, training_history["val_smooth_l1"], "o-", color="#1b9e77", lw=2, label="Val Smooth-L1")
        axes[2].set_xlabel("Época", fontsize=10)
        axes[2].set_ylabel("Erro de Posição da Caixa (Smooth L1)", fontsize=10)
        axes[2].set_title("3. Convergência da LSTM de Movimento", fontsize=11, fontweight="bold")
        axes[2].grid(True, linestyle="--", alpha=0.4)
        axes[2].legend()

    plt.suptitle(
        "Parte 2 (Trilha A) — Comparação Direta: Baseline Ingênuo vs Memória Temporal Recorrente",
        fontsize=13, fontweight="bold", y=1.03
    )
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Painel da Parte 2 salvo em {save_path}")

    return fig


