import torch
import torch.nn.functional as F
from typing import Tuple


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)

def pairwise_scores(
    text_feat: torch.Tensor,
    img0_feat: torch.Tensor,
    img1_feat: torch.Tensor,
    logit_scale_log: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t = l2_normalize(text_feat)
    i0 = l2_normalize(img0_feat)
    i1 = l2_normalize(img1_feat)
    s = logit_scale_log.to(t.dtype).exp()
    sim0 = (t * i0).sum(-1) * s
    sim1 = (t * i1).sum(-1) * s
    return sim0, sim1


def soft_ce_from_pairs(
    s0: torch.Tensor, s1: torch.Tensor, y0: torch.Tensor, y1: torch.Tensor
) -> torch.Tensor:
    logits = torch.stack([s0, s1], dim=-1)
    log_probs = logits.log_softmax(dim=-1)
    targets = torch.stack([y0, y1], dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()
