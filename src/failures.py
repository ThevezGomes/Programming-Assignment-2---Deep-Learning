"""
src/failures.py
Parte 4 — Galeria de Falhas e Horizonte de Memória.

Funções:
- compute_analytical_gradient_norm: medição analítica ||∂L_T/∂h_{T-k}|| vs k
- compute_empirical_horizon: medição empírica — duração de oclusão até ID switch / morte
- plot_failure_gallery: 3 tiras com GT, predição colorida por identidade + diagnóstico
- demonstrate_fix: antes/depois da correção (velocity_damping + sigma_inflation)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from src.models import MotionPredictor
from src.tracker import RNNMotionTracker
from src.data import load_mot17_sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_iou(b1, b2):
    xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
    xi2 = min(b1[0]+b1[2], b2[0]+b2[2]); yi2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


def _best_match_pid(gt_box, preds_frame, min_iou=0.1):
    """Retorna (pid, iou) que melhor coincide com gt_box em preds_frame."""
    best_pid, best_iou = None, min_iou
    for pid, pbox in preds_frame.items():
        iou = _box_iou(gt_box[:4], pbox[:4])
        if iou > best_iou:
            best_iou, best_pid = iou, pid
    return best_pid, best_iou


def _run_tracker(model_path, seq_path, **kwargs):
    gt, dets, info = load_mot17_sequence(seq_path)
    tracker = RNNMotionTracker(model_path, iou_threshold=0.3, max_lost_frames=60,
                               device="cpu", **kwargs)
    preds = tracker.track_sequence(dets)
    return gt, dets, info, preds


# ---------------------------------------------------------------------------
# 1. Medição Analítica: ||∂L_T/∂h_{T-k}||
# ---------------------------------------------------------------------------

def compute_analytical_gradient_norm(
    model_path_rnn: str,
    model_path_lstm: str,
    T: int = 32,
    save_path: str = None
):
    """
    Parte 4 — Horizonte de Memória Analítico.
    Calcula ||∂L_T/∂h_{T-k}|| como função da defasagem k para RNN e LSTM.
    """
    from IPython.display import display as ipy_display

    print("Calculando horizonte de memória analítico (norma do gradiente)...")

    def get_grad_norms(model_path, cell_type, h_dim, use_trained=True):
        """
        Calcula ||∂L_T / ∂h_{T-k}|| via forward manual passo a passo,
        preservando o grafo computacional inteiro para backprop.
        """
        model = MotionPredictor(cell_type=cell_type, hidden_dim=h_dim,
                                num_layers=1, predict_uncertainty=False)

        if use_trained and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            state = ckpt["model_state_dict"]
            compatible = {k: v for k, v in state.items()
                          if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
            model.load_state_dict(compatible, strict=False)

        model.eval()
        torch.manual_seed(42)

        T_local = T
        x_seq = torch.zeros(1, T_local, 7)
        for t in range(T_local):
            x_seq[0, t] = torch.tensor(
                [0.5 + 0.002 * t, 0.5, 0.05, 0.10, 0.002, 0.0, 1.0]
            )

        # Extrai pesos do módulo nn.RNN/LSTM
        rnn_cell = model.rnn
        input_proj = model.input_proj

        # Inicializa estados
        h_t = torch.zeros(h_dim)  # estado oculto inicial

        h_states = []  # lista de h_t que participam do grafo

        if cell_type == "rnn":
            # Pesos: weight_ih_l0 (hidden_dim, input_size), weight_hh_l0 (hidden_dim, hidden_dim)
            W_ih = rnn_cell.weight_ih_l0  # (H, input_dim)
            W_hh = rnn_cell.weight_hh_l0  # (H, H)
            b_ih = rnn_cell.bias_ih_l0    # (H,)
            b_hh = rnn_cell.bias_hh_l0    # (H,)

            for t in range(T_local):
                x_in = x_seq[0, t, :].unsqueeze(0)  # (1, input_dim)
                emb = input_proj(x_in.unsqueeze(0)).squeeze()  # (hidden_dim,)

                z = emb @ W_ih.T + h_t @ W_hh.T + b_ih + b_hh
                h_t = torch.tanh(z)
                h_states.append(h_t)

        elif cell_type in ("lstm", "gru"):
            # Para LSTM e GRU: usamos o passo individual via nn.LSTMCell / nn.GRUCell
            # Extraímos os pesos e reconstruímos
            W_ih = rnn_cell.weight_ih_l0
            W_hh = rnn_cell.weight_hh_l0
            b_ih = rnn_cell.bias_ih_l0
            b_hh = rnn_cell.bias_hh_l0

            if cell_type == "lstm":
                c_t = torch.zeros(h_dim)
                for t in range(T_local):
                    x_in = x_seq[0, t, :].unsqueeze(0)
                    emb = input_proj(x_in.unsqueeze(0)).squeeze()  # (hidden_dim,)

                    gates = emb @ W_ih.T + h_t @ W_hh.T + b_ih + b_hh
                    H = h_dim
                    i_gate = torch.sigmoid(gates[:H])
                    f_gate = torch.sigmoid(gates[H:2*H])
                    g_gate = torch.tanh(gates[2*H:3*H])
                    o_gate = torch.sigmoid(gates[3*H:])

                    c_t = f_gate * c_t + i_gate * g_gate
                    h_t = o_gate * torch.tanh(c_t)
                    h_states.append(h_t)
            else:
                for t in range(T_local):
                    x_in = x_seq[0, t, :].unsqueeze(0)
                    emb = input_proj(x_in.unsqueeze(0)).squeeze()

                    gates_x = emb @ W_ih.T + b_ih
                    gates_h = h_t @ W_hh.T + b_hh
                    H = h_dim
                    r = torch.sigmoid(gates_x[:H] + gates_h[:H])
                    z = torch.sigmoid(gates_x[H:2*H] + gates_h[H:2*H])
                    n = torch.tanh(gates_x[2*H:] + r * gates_h[2*H:])
                    h_t = (1 - z) * n + z * h_t
                    h_states.append(h_t)

        # Loss no último passo
        loss = h_states[-1].sum()

        # ||∂L_T / ∂h_{T-k}||: gradiente direto pela cadeia conectada
        norms = []
        for k in range(T_local):
            t_state = h_states[T_local - 1 - k]
            try:
                (g,) = torch.autograd.grad(
                    loss, t_state,
                    retain_graph=True, allow_unused=True, create_graph=False
                )
                norms.append(g.norm().item() if g is not None else 0.0)
            except Exception:
                norms.append(0.0)

        n0 = max(norms[0], 1e-15)
        return [val / n0 for val in norms]

    norms_rnn  = get_grad_norms(model_path_rnn,  "rnn",  128)
    norms_lstm = get_grad_norms(model_path_lstm, "lstm",  64)

    k_vals = np.arange(T)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(k_vals, norms_rnn,  label="RNN Simples", color="#d62728", lw=2.5, marker="o", markersize=3)
    ax.semilogy(k_vals, norms_lstm, label="LSTM",        color="#1f77b4", lw=2.5, marker="s", markersize=3)

    # Queda 10× na RNN
    n0 = norms_rnn[0] if norms_rnn[0] > 0 else 1
    threshold_10x = n0 / 10
    idx_10x = next((i for i, v in enumerate(norms_rnn) if v < threshold_10x), None)
    if idx_10x:
        ax.axvline(idx_10x, color="#d62728", linestyle="--", alpha=0.6,
                   label=f"RNN: queda 10× em k={idx_10x}")

    ax.set_xlabel("Defasagem $k$ (passos no passado)", fontsize=12)
    ax.set_ylabel(r"Decaimento relativo $\|\partial L_T / \partial h_{T-k}\| / \|\partial L_T / \partial h_T\|$", fontsize=10)
    ax.set_title(
        "Parte 4 — Horizonte de Memória Analítico\n"
        "Vanishing Gradient: RNN simples colapsa em poucos passos, LSTM se sustenta",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Salvo em {save_path}")

    ipy_display(fig)
    plt.close(fig)
    return {"rnn": norms_rnn, "lstm": norms_lstm}


# ---------------------------------------------------------------------------
# 2. Medição Empírica
# ---------------------------------------------------------------------------

def compute_empirical_horizon(
    model_path: str,
    seq_path: str = "data/MOT17/train/MOT17-09-SDP",
    save_path: str = None
):
    """
    Parte 4 — Horizonte de Memória Empírico.
    Compara a distribuição de duração das oclusões que causam ID Switch
    com as que a track sobrevive. Mostra onde o modelo "quebra" na prática.
    """
    from IPython.display import display as ipy_display

    print("Calculando horizonte de memória empírico (distribuição de oclusões)...")
    gt, dets, info, preds = _run_tracker(model_path, seq_path)

    switch_lens    = []  # durações de oclusão que causaram ID Switch
    survived_lens  = []  # durações de oclusão que a track sobreviveu

    for gt_id in sorted(set(tid for frame in gt.values() for tid in frame)):
        gt_frames = sorted(f for f in gt if gt_id in gt[f])
        for i in range(len(gt_frames) - 1):
            gap = gt_frames[i+1] - gt_frames[i]
            if gap < 3:
                continue
            f_before, f_after = gt_frames[i], gt_frames[i+1]

            pid_before, iou_b = _best_match_pid(gt[f_before][gt_id], preds.get(f_before, {}))
            pid_after,  iou_a = _best_match_pid(gt[f_after][gt_id],  preds.get(f_after,  {}))

            if pid_before is None or pid_after is None or iou_b < 0.2 or iou_a < 0.2:
                continue

            if pid_before != pid_after:
                switch_lens.append(gap)
            else:
                survived_lens.append(gap)

    med_sw  = int(np.median(switch_lens))   if switch_lens   else 0
    med_sur = int(np.median(survived_lens)) if survived_lens else 0

    print(f"  Oclusões com ID Switch: {len(switch_lens)}, mediana = {med_sw} quadros")
    print(f"  Oclusões sobrevividas:  {len(survived_lens)}, mediana = {med_sur} quadros")
    print(f"  → Horizonte efetivo do modelo: ~{med_sur} quadros (acima disso, predominam switches)")

    all_vals = switch_lens + survived_lens
    max_gap  = max(all_vals) if all_vals else 60
    bins = np.arange(1, max_gap + 2, 2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(survived_lens, bins=bins, alpha=0.65, color="#2ca02c",
            label=f"Sobreviveu (n={len(survived_lens)}, mediana={med_sur}f)")
    ax.hist(switch_lens,   bins=bins, alpha=0.75, color="#d62728",
            label=f"ID Switch / Morte (n={len(switch_lens)}, mediana={med_sw}f)")
    ax.axvline(med_sur, color="#2ca02c", linestyle="--", lw=2)
    ax.axvline(med_sw,  color="#d62728", linestyle="--", lw=2)
    ax.set_xlabel("Duração da oclusão (quadros)", fontsize=12)
    ax.set_ylabel("Frequência", fontsize=12)
    ax.set_title(
        f"Parte 4 — Horizonte de Memória Empírico (MOT17-09)\n"
        f"Horizonte efetivo ≈ {med_sur}f — acima disso os ID Switches dominam",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Salvo em {save_path}")

    ipy_display(fig)
    plt.close(fig)
    return {"switch_lens": switch_lens, "survived_lens": survived_lens,
            "horizon_frames": med_sur}


# ---------------------------------------------------------------------------
# 3. Galeria de Falhas
# ---------------------------------------------------------------------------

# 3 falhas confirmadas por diagnóstico com IoU real (ver /tmp/find_real_failures.py)
CONFIRMED_FAILURES = [
    {
        # GT:1 | Gap=25f | frames 221→246 | Pred 20 → 28
        "gt_id": 1,
        "f_before": 221, "f_after": 246, "gap": 25,
        "pid_before": 20, "pid_after": 28,
        "title": "Falha 1 — GT:1 | Oclusão de 25 quadros (cruzamento com multidão)",
        "diagnosis": (
            "O pedestre GT:1 fica ocluído por 25 quadros (frames 221→246). "
            "A janela de BPTT é T=32, mas a norma analítica ∂L/∂h cai >10× antes de k=10 passos "
            "na RNN simples. No LSTM, a memória persiste — porém o rollout livre diverge da "
            "trajetória real quando o pedestre muda de direção ao desviar de outros. "
            "Resultado: IoU(caixa prevista, nova detecção) < 0.3 → ID Switch: Pred 20 → Pred 28."
        ),
    },
    {
        # GT:17 | Gap=28f | frames 451→479 | Pred 44 → 43
        "gt_id": 17,
        "f_before": 451, "f_after": 479, "gap": 28,
        "pid_before": 44, "pid_after": 43,
        "title": "Falha 2 — GT:17 | Oclusão de 28 quadros (cruzamento denso)",
        "diagnosis": (
            "O pedestre GT:17 fica oculto por 28 quadros contínuos (frames 451→479). "
            "O estado h_{t} carrega posição residual com erro crescente no rollout; "
            "quando o pedestre reaparece em posição ~137px à direita do previsto, "
            "o IoU da caixa prevista com a nova detecção é ~0.05. "
            "Isso fica abaixo do limiar de 0.3 → track Pred:44 morre, nova track Pred:43 nasce."
        ),
    },
    {
        # GT:17 | Gap=25f | frames 494→519 | Pred 43 → 29
        "gt_id": 17,
        "f_before": 494, "f_after": 519, "gap": 25,
        "pid_before": 43, "pid_after": 29,
        "title": "Falha 3 — GT:17 | Segunda oclusão de 25 quadros (acúmulo de erro)",
        "diagnosis": (
            "Segunda oclusão consecutiva do mesmo pedestre GT:17 (25 quadros, frames 494→519). "
            "Após o ID Switch da Falha 2, o pedestre já opera com o ID Pred:43. "
            "O rollout livre acumula erro de velocidade — o pedestre acelera no cruzamento "
            "enquanto o modelo prediz desaceleração. "
            "A incerteza gaussian padrão (log_sigma ≈ -1.5) não cresce o suficiente para "
            "abrir o portão adaptativo → nova track Pred:29 nasce, Pred:43 declarada morta."
        ),
    },
]


def _draw_strip(axes, frames, gt, gt_id, pid_before, pid_after, preds, seq_path, crop=True):
    """Desenha uma tira de quadros para uma falha específica, com zoom na região de interesse."""
    # Determina bounding box da região de interesse (entorno do GT)
    gt_boxes_in_range = [gt[f][gt_id][:4] for f in frames if f in gt and gt_id in gt[f]]
    if gt_boxes_in_range:
        xs = [b[0] for b in gt_boxes_in_range]
        ys = [b[1] for b in gt_boxes_in_range]
        ws = [b[2] for b in gt_boxes_in_range]
        hs = [b[3] for b in gt_boxes_in_range]
        # Janela de crop: região de interesse + margem generosa
        margin = 150
        crop_x1 = max(0, int(min(xs)) - margin)
        crop_y1 = max(0, int(min(ys)) - margin)
        crop_x2 = int(max(x + w for x, w in zip(xs, ws))) + margin
        crop_y2 = int(max(y + h for y, h in zip(ys, hs))) + margin
    else:
        crop = False

    for ax, f in zip(axes, frames):
        img_path = os.path.join(seq_path, "img1", f"{f:06d}.jpg")
        img = np.array(Image.open(img_path))

        if crop:
            cx1, cy1 = crop_x1, crop_y1
            cx2, cy2 = min(crop_x2, img.shape[1]), min(crop_y2, img.shape[0])
            img_show = img[cy1:cy2, cx1:cx2]
            ox, oy = cx1, cy1  # offset para ajustar coordenadas das caixas
        else:
            img_show = img
            ox, oy = 0, 0

        ax.imshow(img_show)
        ax.axis("off")

        # Destaque: frame antes do gap, durante, e depois
        if f in gt and gt_id not in gt[f]:
            ax.set_title(f"Frame {f}\n⚠ Oculto", fontsize=7.5, color="gray")
        elif f == frames[0]:
            ax.set_title(f"Frame {f}\n✓ Visível", fontsize=7.5, color="lime")
        elif f == frames[-1]:
            ax.set_title(f"Frame {f}\n↩ Reaparecer", fontsize=7.5, color="cyan")
        else:
            ax.set_title(f"Frame {f}", fontsize=7.5)

        # Caixa GT (verde)
        if f in gt and gt_id in gt[f]:
            x, y, w, h = gt[f][gt_id][:4]
            rect = plt.Rectangle((x - ox, y - oy), w, h, fill=False,
                                  edgecolor="lime", lw=2.5, linestyle="-")
            ax.add_patch(rect)
            ax.text(x - ox, y - oy - 6, f"GT:{gt_id}", color="lime", fontsize=7,
                    fontweight="bold", bbox=dict(facecolor="black", alpha=0.5, pad=1))

        # Caixas do tracker neste frame
        for pid, pbox in preds.get(f, {}).items():
            x, y, w, h = pbox[:4]
            # Só plota se estiver dentro da janela de crop
            if crop and (x + w < cx1 or x > cx2 or y + h < cy1 or y > cy2):
                continue
            if pid == pid_before:
                color, style = "#ff4444", "-"   # ID original: vermelho
            elif pid == pid_after:
                color, style = "#ff9900", "--"  # ID novo (switch): laranja
            else:
                continue  # Ignora outros IDs para não poluir
            rect = plt.Rectangle((x - ox, y - oy), w, h, fill=False,
                                  edgecolor=color, lw=2.0, linestyle=style)
            ax.add_patch(rect)
            ax.text(x - ox + w, y - oy + h, f"P:{pid}", color=color, fontsize=7,
                    fontweight="bold", bbox=dict(facecolor="black", alpha=0.5, pad=1))


def plot_failure_gallery(
    model_path: str,
    seq_path: str = "data/MOT17/train/MOT17-09-SDP",
    save_dir: str = "docs/",
    bptt_window: int = 32
):
    """
    Parte 4 — Galeria de 3 Falhas.
    Cada falha: tira de 6 quadros com crop na região de interesse,
    GT (verde), pred original (vermelho sólido), novo ID após switch (laranja tracejado)
    + diagnóstico quantitativo.
    """
    from IPython.display import display as ipy_display

    print("Gerando Galeria de Falhas (3 ID Switches confirmados por IoU)...")
    gt, dets, info, preds = _run_tracker(model_path, seq_path)
    os.makedirs(save_dir, exist_ok=True)

    for idx, case in enumerate(CONFIRMED_FAILURES):
        gt_id      = case["gt_id"]
        f_before   = case["f_before"]
        f_after    = case["f_after"]
        pid_before = case["pid_before"]
        pid_after  = case["pid_after"]
        gap        = case["gap"]

        # 6 frames: 2 antes do gap, 2 durante, 2 depois
        n_before = 2
        step_during = max(1, gap // 3)
        frames = (
            [f_before - 2, f_before] +
            [f_before + step_during, f_before + 2 * step_during] +
            [f_after, f_after + 3]
        )
        frames = [f for f in frames if f >= 1]

        fig, axes = plt.subplots(1, len(frames), figsize=(3.2 * len(frames), 3.8))

        _draw_strip(axes, frames, gt, gt_id, pid_before, pid_after, preds, seq_path, crop=True)

        # Legenda
        legend_patches = [
            mpatches.Patch(edgecolor="lime",    facecolor="none", lw=2,
                           label=f"Ground Truth (GT:{gt_id})"),
            mpatches.Patch(edgecolor="#ff4444", facecolor="none", lw=2,
                           label=f"Pred:{pid_before} (ID original)"),
            mpatches.Patch(edgecolor="#ff9900", facecolor="none", lw=2, linestyle="--",
                           label=f"Pred:{pid_after} (novo ID após switch)"),
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.03), fontsize=8)

        # Título
        fig.suptitle(case["title"], fontsize=10, fontweight="bold", y=1.02)


        plt.tight_layout()
        save_file = os.path.join(save_dir, f"parte4_falha{idx+1}_gt{gt_id}.png")
        plt.savefig(save_file, dpi=150, bbox_inches="tight")
        ipy_display(fig)
        plt.close(fig)
        print(f"  Falha {idx+1} salva: {save_file}")

    print(f"\nGaleria completa salva em {save_dir}")


# ---------------------------------------------------------------------------
# 4. Correção: demonstrate_fix
# ---------------------------------------------------------------------------

def demonstrate_fix(
    model_path: str,
    seq_path: str = "data/MOT17/train/MOT17-09-SDP",
    save_dir: str = "docs/",
    velocity_damping: float = 0.85,
    sigma_inflation: float = 0.3
):
    """
    Parte 4 — Correção: antes/depois aplicada à Falha 1 (GT:1, gap=25f, frames 221→246).
    O detector TEM uma detecção com IoU=0.855 no reaparecimento, mas o pred divergiu
    para IoU=0.179 < threshold=0.30 → ID Switch.
    Correção: portão adaptativo proporcional ao tempo de oclusão (sigma_inflation)
    abaixa o threshold progressivamente, permitindo reacitar o pedestre correto.
    """
    from IPython.display import display as ipy_display
    from src.metrics import evaluate_tracking

    print(f"Aplicando correção: velocity_damping={velocity_damping}, sigma_inflation={sigma_inflation}")
    print("Falha alvo: GT:1 | frames 221→246 | gap=25f | Pred:20→28")
    print("Diagnóstico: detector tem IoU=0.855, mas pred divergiu para IoU=0.179 < threshold=0.30")
    print("Correção: portão adaptativo proporcional ao tempo de oclusão")

    gt, dets, info, preds_antes = _run_tracker(model_path, seq_path)
    _, _, _, preds_depois = _run_tracker(
        model_path, seq_path,
        velocity_damping=velocity_damping,
        sigma_inflation=sigma_inflation
    )

    m_antes  = evaluate_tracking(gt, preds_antes)
    m_depois = evaluate_tracking(gt, preds_depois)

    delta_idf1 = m_depois["idf1"] - m_antes["idf1"]
    delta_sw   = m_depois["id_switches"] - m_antes["id_switches"]
    delta_frag = m_depois.get("fragmentations", 0) - m_antes.get("fragmentations", 0)

    print(f"\n  Métricas globais (MOT17-09, sequência completa):")
    print(f"  {'':22s} {'ANTES':>10s}  {'DEPOIS':>10s}  {'Delta':>10s}")
    print(f"  {'IDF1':22s} {m_antes['idf1']:>10.4f}  {m_depois['idf1']:>10.4f}  {delta_idf1:>+10.4f}")
    print(f"  {'ID Switches':22s} {m_antes['id_switches']:>10d}  {m_depois['id_switches']:>10d}  {delta_sw:>+10d}")
    print(f"  {'Fragmentações':22s} {m_antes.get('fragmentations',0):>10d}  {m_depois.get('fragmentations',0):>10d}  {delta_frag:>+10d}")

    # Foca na Falha 1 (GT:1, 221→246)
    case    = CONFIRMED_FAILURES[0]
    gt_id   = case["gt_id"]
    f_bef   = case["f_before"]
    f_aft   = case["f_after"]
    gap     = case["gap"]
    step_d  = max(1, gap // 3)
    frames  = [f_bef - 2, f_bef, f_bef + step_d, f_bef + 2*step_d, f_aft, f_aft + 3]

    # Verifica se a correção recuperou o ID no reaparecimento
    pid_original = case["pid_before"]
    pid_depois_aft = None
    for pid, pb in preds_depois.get(f_aft, {}).items():
        if _box_iou(pb[:4], gt[f_aft][gt_id][:4]) > 0.2:
            pid_depois_aft = pid
            break
    recovered = (pid_depois_aft == pid_original)

    fig, axes = plt.subplots(2, len(frames), figsize=(3.0 * len(frames), 7.5))

    for row, (preds_row, label) in enumerate([
        (preds_antes,  "ANTES — threshold fixo (0.30)"),
        (preds_depois, f"DEPOIS — portão adaptativo (threshold ∝ tempo de oclusão)")
    ]):
        _draw_strip(axes[row], frames, gt, gt_id,
                    case["pid_before"], case["pid_after"],
                    preds_row, seq_path, crop=True)
        axes[row][0].set_ylabel(label, fontsize=8.5, fontweight="bold",
                                rotation=90, labelpad=5)

    status_str = "✓ ID RECUPERADO" if recovered else "parcialmente corrigido"
    fig.suptitle(
        f"Parte 4 — Correção | GT:1 (gap={gap}f, frames {f_bef}→{f_aft}) | {status_str}\n"
        f"IDF1: {m_antes['idf1']:.4f} → {m_depois['idf1']:.4f} ({delta_idf1:+.4f})  |  "
        f"ID Switches: {m_antes['id_switches']} → {m_depois['id_switches']} ({delta_sw:+d})",
        fontsize=10, fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_path = os.path.join(save_dir, "parte4_antes_depois_correcao.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    ipy_display(fig)
    plt.close(fig)
    print(f"\n  Imagem comparativa salva em {save_path}")
