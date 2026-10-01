"""
src/models.py
Arquiteturas de Redes Recorrentes como Modelo de Movimento (Trilha A do PA2):
- Suporte a células configuráveis: LSTM, GRU, RNN Simples (para ablações da Parte 3)
- Entrada temporal: [cx, cy, w, h, vx, vy, dt]
- Predição residual da caixa: box_{t+1} = box_t + delta_box
- Cabeça de incerteza gaussiana: log(sigma) para portão adaptativo e Gaussian NLL Loss
- Métodos de step unitário e rollout autoregressivo (livre de observações para oclusão)
"""

import torch
import torch.nn as nn
import numpy as np


class MotionPredictor(nn.Module):
    """
    Modelo recorrente para previsão de trajetórias de caixas delimitadoras.
    
    Parâmetros
    ----------
    cell_type   : 'lstm', 'gru' ou 'rnn' (permite comparar as 3 na Parte 3)
    input_dim   : dimensão do vetor de observação (padrão 7: [cx, cy, w, h, vx, vy, dt])
    hidden_dim  : dimensão do estado oculto
    num_layers  : número de camadas recorrentes
    dropout     : taxa de dropout entre camadas recorrentes
    predict_uncertainty : se True, prevê também log(sigma) para portão adaptativo
    """
    def __init__(
        self,
        cell_type: str = "lstm",
        input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        predict_uncertainty: bool = True
    ):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.predict_uncertainty = predict_uncertainty

        # Camada de embedding de entrada
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Núcleo recorrente
        drop_rnn = dropout if num_layers > 1 else 0.0
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=drop_rnn
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=drop_rnn
            )
        elif self.cell_type == "rnn":
            self.rnn = nn.RNN(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity="tanh",
                dropout=drop_rnn
            )
        else:
            raise ValueError(f"cell_type desconhecido: {cell_type}. Escolha 'lstm', 'gru' ou 'rnn'.")

        # Cabeça preditora de deslocamento residual: [d_cx, d_cy, d_w, d_h]
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)
        )

        # Cabeça preditora de log-variância (incerteza aleatórica): [s_cx, s_cy, s_w, s_h]
        if self.predict_uncertainty:
            self.logvar_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 4)
            )

    def init_hidden(self, batch_size: int, device: torch.device):
        """Inicializa o estado oculto da recorrência."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        if self.cell_type == "lstm":
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h0, c0)
        return h0

    def step(self, x_t: torch.Tensor, current_box: torch.Tensor, hidden=None):
        """
        Executa um único passo de inferência temporal (quadro a quadro no tracker).
        
        Parâmetros
        ----------
        x_t         : (B, input_dim) tensor de características no quadro t
        current_box : (B, 4) coordenadas [cx, cy, w, h] da caixa no quadro t
        hidden      : estado oculto prévio da RNN
        
        Retorna
        -------
        pred_box    : (B, 4) caixa prevista para t+1
        log_sigma   : (B, 4) incerteza prevista (ou None)
        next_hidden : estado oculto atualizado
        """
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)  # (B, 1, input_dim)

        emb = self.input_proj(x_t)
        out, next_hidden = self.rnn(emb, hidden)
        feat = out.squeeze(1)  # (B, hidden_dim)

        delta = self.mean_head(feat)
        pred_box = current_box + delta

        log_sigma = None
        if self.predict_uncertainty:
            # Clampa entre -4 e +3 para estabilidade numérica
            log_sigma = torch.clamp(self.logvar_head(feat), min=-4.0, max=3.0)

        return pred_box, log_sigma, next_hidden

    def forward(
        self,
        x_seq: torch.Tensor,
        init_boxes: torch.Tensor,
        hidden=None,
        teacher_forcing_ratio: float = 1.0,
        target_seq: torch.Tensor = None
    ):
        """
        Passagem completa sobre uma sequência de T quadros (usada no treino e ablação).
        Suporta Teacher Forcing e Scheduled Sampling.
        
        Parâmetros
        ----------
        x_seq       : (B, T, input_dim)
        init_boxes  : (B, 4) caixa no instante 0
        hidden      : estado inicial (ou None)
        teacher_forcing_ratio : 1.0 = sempre ground truth, 0.0 = autoregressivo livre
        target_seq  : (B, T, 4) caixas ground truth
        
        Retorna
        -------
        pred_boxes  : (B, T, 4)
        pred_logvars: (B, T, 4) ou None
        final_hidden: último estado oculto
        """
        B, T, _ = x_seq.shape
        device = x_seq.device

        if hidden is None:
            hidden = self.init_hidden(B, device)

        pred_boxes_list = []
        pred_logvars_list = []

        curr_box = init_boxes
        last_box = init_boxes

        for t in range(T):
            # Se scheduled sampling ativo e não for o primeiro passo
            use_teacher = (torch.rand(1).item() < teacher_forcing_ratio) or (t == 0)

            if use_teacher or target_seq is None:
                x_t = x_seq[:, t, :]
                ref_box = curr_box
            else:
                # Constrói x_t autoregressivamente a partir da predição anterior
                v_pred = (curr_box[:, :2] - last_box[:, :2])
                dt = x_seq[:, t, 6:7] if self.input_dim >= 7 else torch.ones((B, 1), device=device)
                x_t = torch.cat([curr_box, v_pred, dt], dim=-1)
                ref_box = curr_box

            pred_box, log_sigma, hidden = self.step(x_t, ref_box, hidden)
            pred_boxes_list.append(pred_box)
            if log_sigma is not None:
                pred_logvars_list.append(log_sigma)

            last_box = curr_box
            if use_teacher and target_seq is not None:
                curr_box = target_seq[:, t, :]
            else:
                curr_box = pred_box

        pred_boxes = torch.stack(pred_boxes_list, dim=1)  # (B, T, 4)
        pred_logvars = torch.stack(pred_logvars_list, dim=1) if self.predict_uncertainty else None

        return pred_boxes, pred_logvars, hidden

    def rollout(self, start_box: torch.Tensor, steps: int, hidden, dt: float = 1.0):
        """
        Projeta uma trajetória livre para frente por 'steps' quadros sem observação.
        Essencial para a oclusão: a track navega livremente mantendo momento.
        """
        curr_box = start_box
        last_box = start_box
        device = start_box.device

        projected = []
        sigmas = []

        for _ in range(steps):
            v = curr_box[:, :2] - last_box[:, :2]
            dt_tensor = torch.full((start_box.shape[0], 1), dt, device=device)
            x_t = torch.cat([curr_box, v, dt_tensor], dim=-1)

            pred_box, log_sigma, hidden = self.step(x_t, curr_box, hidden)
            projected.append(pred_box)
            if log_sigma is not None:
                sigmas.append(torch.exp(log_sigma))

            last_box = curr_box
            curr_box = pred_box

        projected = torch.stack(projected, dim=1)
        sigmas = torch.stack(sigmas, dim=1) if len(sigmas) > 0 else None
        return projected, sigmas, hidden
