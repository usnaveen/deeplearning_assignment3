"""
Noam learning-rate scheduler from "Attention Is All You Need".
"""

from __future__ import annotations

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_lr_scale(self) -> float:
        step = max(self.last_epoch + 1, 1)
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

    def get_lr(self) -> list[float]:
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


def get_lr_history(
    d_model: int,
    warmup_steps: int,
    total_steps: int,
) -> list[float]:
    dummy_model = torch.nn.Linear(1, 1)
    optimizer = optim.Adam(dummy_model.parameters(), lr=1.0)
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)
    history = []
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return history


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lrs = get_lr_history(d_model=512, warmup_steps=4000, total_steps=20000)
    plt.figure(figsize=(9, 4))
    plt.plot(lrs)
    plt.axvline(4000, color="red", linestyle="--", label="warmup=4000")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Noam LR Schedule")
    plt.legend()
    plt.tight_layout()
    plt.show()
