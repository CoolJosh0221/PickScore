import torch
from typing import Dict


@torch.no_grad()
def pref_metrics(
    s0: torch.Tensor,
    s1: torch.Tensor,
    y0: torch.Tensor,
    y1: torch.Tensor,
    tie_margin: float = 0.1,
) -> Dict[str, float]:
    label_gap = y0 - y1
    pred_gap = s0 - 0.5
    is_tie = label_gap == 0
    is_non_tie = ~is_tie
    nt_correct = (
        (torch.sign(pred_gap[is_non_tie]) == torch.sign(label_gap[is_non_tie]))
        .sum()
        .item()
    )
    nt_total = is_non_tie.sum().item()
    tie_correct = (pred_gap[is_tie].abs() <= tie_margin).sum().item()
    tie_total = is_tie.sum().item()
    pref_acc = nt_correct / max(nt_total, 1)
    tie_acc = tie_correct / max(tie_total, 1)
    overall = (nt_correct + tie_correct) / max(nt_total + tie_total, 1)
    return {
        "pref_acc": pref_acc,
        "tie_acc": tie_acc,
        "overall_acc": overall,
        "non_tie_correct": nt_correct,
        "non_tie_total": nt_total,
        "tie_correct": tie_correct,
        "tie_total": tie_total,
    }


class PrefMetricTracker:
    def __init__(self, tie_margin: float = 0.1) -> None:
        self.tie_margin = tie_margin
        self.reset()

    def reset(self) -> None:
        self.nt_correct = 0
        self.nt_total = 0
        self.t_correct = 0
        self.t_total = 0

    @torch.no_grad()
    def update(
        self, s0: torch.Tensor, s1: torch.Tensor, y0: torch.Tensor, y1: torch.Tensor
    ) -> None:
        m = pref_metrics(s0, s1, y0, y1, self.tie_margin)
        self.nt_correct += m["non_tie_correct"]
        self.nt_total += m["non_tie_total"]
        self.t_correct += m["tie_correct"]
        self.t_total += m["tie_total"]

    def compute(self) -> Dict[str, float]:
        pref = self.nt_correct / max(self.nt_total, 1)
        tie = self.t_correct / max(self.t_total, 1)
        overall = (self.nt_correct + self.t_correct) / max(
            self.nt_total + self.t_total, 1
        )
        return {"pref_acc": pref, "tie_acc": tie, "overall_acc": overall}
