"""
Loss functions for SEMSPEM bi-encoder training.

Implements a hierarchical, weighted InfoNCE loss over easy / semi-hard / hard
negatives, where harder negative tiers receive a larger exponential weight.
"""

import math
from typing import List

import torch
import torch.nn.functional as F

N_LEVELS = 3


def compute_weight(level: int, n_levels: int = N_LEVELS) -> float:
    """
    exp(1 / (n_levels - level + 1))

    level 1 (Easy):      exp(1/3) ~= 1.40
    level 2 (Semi-hard): exp(1/2) ~= 1.65
    level 3 (Hard):      exp(1/1) ~= 2.72
    """
    return math.exp(1.0 / (n_levels - level + 1))


# Base multiplier per negative tier, scaled by compute_weight()
W_EASY = 1 * compute_weight(level=1)
W_SEMI = 2 * compute_weight(level=2)
W_HARD = 4 * compute_weight(level=3)


def hierarchical_infonce_loss(
    f_anchor:    torch.Tensor,
    f_pos:       torch.Tensor,
    f_easy_list: List[torch.Tensor],
    f_semi_list: List[torch.Tensor],
    f_hard_list: List[torch.Tensor],
    tau: float = 0.1,
    w_e: float = W_EASY,
    w_s: float = W_SEMI,
    w_h: float = W_HARD,
) -> torch.Tensor:
    """InfoNCE loss with three negative tiers, each weighted by its difficulty."""
    f_a = F.normalize(f_anchor, dim=0)
    f_p = F.normalize(f_pos,    dim=0)

    sim_pos = torch.dot(f_a, f_p) / tau

    easy_term = sum(
        w_e * torch.exp(torch.dot(f_a, F.normalize(f_n, dim=0)) / tau)
        for f_n in f_easy_list
    )
    semi_term = sum(
        w_s * torch.exp(torch.dot(f_a, F.normalize(f_n, dim=0)) / tau)
        for f_n in f_semi_list
    )
    hard_term = sum(
        w_h * torch.exp(torch.dot(f_a, F.normalize(f_n, dim=0)) / tau)
        for f_n in f_hard_list
    )

    numerator   = torch.exp(sim_pos)
    denominator = numerator + easy_term + semi_term + hard_term

    return -torch.log(numerator / denominator)
