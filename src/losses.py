"""
src/losses.py
Funções de perda para treinamento do modelo de movimento da Trilha A:
- SmoothL1BoxLoss: Perda L1 suave sobre as caixas delimitadoras
- GaussianNLLBoxLoss: Negative Log-Likelihood Gaussiana para predição de incerteza
- CombinedMotionLoss: Combinação de regressão geométrica com calibração de incerteza
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothL1BoxLoss(nn.Module):
    """Perda Smooth L1 (Huber) para regressão de caixas delimitadoras."""
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor, mask: torch.Tensor = None):
        """
        pred_boxes   : (B, T, 4) ou (N, 4)
        target_boxes : (B, T, 4) ou (N, 4)
        mask         : (B, T) booleano indicando quadros válidos
        """
        loss = F.smooth_l1_loss(pred_boxes, target_boxes, beta=self.beta, reduction="none")  # (..., 4)
        loss = loss.mean(dim=-1)  # Média sobre as 4 coordenadas

        if mask is not None:
            valid_count = torch.clamp(mask.sum(), min=1.0)
            return (loss * mask).sum() / valid_count
        return loss.mean()


class GaussianNLLBoxLoss(nn.Module):
    """
    Log-Verossimilhança Gaussiana Negativa (Gaussian NLL Loss):
    L = 0.5 * [ exp(-2 * log_sigma) * (target - pred)^2 + 2 * log_sigma ]
    
    Permite à rede aprender incerteza heterocedástica (aleatórica):
    sob oclusão ou movimentos abruptos, a incerteza cresce e atenua o gradiente quadrático.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        log_sigma: torch.Tensor,
        target_boxes: torch.Tensor,
        mask: torch.Tensor = None
    ):
        """
        pred_boxes   : (..., 4)
        log_sigma    : (..., 4) logaritmo do desvio padrão
        target_boxes : (..., 4)
        """
        # Variância sigma^2 = exp(2 * log_sigma)
        var = torch.exp(2.0 * log_sigma)
        sq_diff = (target_boxes - pred_boxes) ** 2

        # NLL por coordenada
        nll = 0.5 * (sq_diff / (var + self.eps) + 2.0 * log_sigma)
        loss = nll.mean(dim=-1)

        if mask is not None:
            valid_count = torch.clamp(mask.sum(), min=1.0)
            return (loss * mask).sum() / valid_count
        return loss.mean()


class CombinedMotionLoss(nn.Module):
    """Perda combinada ponderando Smooth L1 e calibração de incerteza Gaussiana."""
    def __init__(self, alpha_nll: float = 0.5, beta_smooth: float = 1.0):
        super().__init__()
        self.smooth_l1 = SmoothL1BoxLoss(beta=beta_smooth)
        self.gaussian_nll = GaussianNLLBoxLoss()
        self.alpha_nll = alpha_nll

    def forward(
        self,
        pred_boxes: torch.Tensor,
        log_sigma: torch.Tensor,
        target_boxes: torch.Tensor,
        mask: torch.Tensor = None
    ):
        l_reg = self.smooth_l1(pred_boxes, target_boxes, mask=mask)
        if log_sigma is not None:
            l_nll = self.gaussian_nll(pred_boxes, log_sigma, target_boxes, mask=mask)
            total = l_reg + self.alpha_nll * l_nll
            return total, l_reg, l_nll
        return l_reg, l_reg, torch.tensor(0.0, device=pred_boxes.device)
