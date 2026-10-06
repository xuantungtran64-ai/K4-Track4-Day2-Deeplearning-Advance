"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_criterion(kind: str = "ce", **kw):
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=kw.get("weight"))
    elif kind == "ls":
        return LabelSmoothingCE(smoothing=kw.get("smoothing", 0.1))
    elif kind == "focal":
        return FocalLoss(gamma=kw.get("gamma", 2.0), alpha=kw.get("alpha", None))
    elif kind == "ce_weighted":
        return nn.CrossEntropyLoss(weight=kw.get("weight", None))
    else:
        raise ValueError(f"Unknown kind {kind}")


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits, targets):
        return self.criterion(logits, targets)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        log_pt = F.log_softmax(logits, dim=-1)
        pt = torch.exp(log_pt)
        log_pt_target = log_pt.gather(1, targets.view(-1, 1)).squeeze(1)
        pt_target = pt.gather(1, targets.view(-1, 1)).squeeze(1)
        
        loss = - (1 - pt_target) ** self.gamma * log_pt_target
        if self.alpha is not None:
            alpha_target = self.alpha[targets]
            loss = loss * alpha_target
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    counts = torch.tensor(list(counts.values()), dtype=torch.float32) if isinstance(counts, dict) else torch.tensor(counts, dtype=torch.float32)
    if beta == 0.0:
        w = 1.0 / counts
        w = w / w.mean()
        return w
    else:
        w = (1 - beta) / (1 - beta ** counts)
        w = w / w.sum() * len(counts)
        return w


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    if mode == "mixup":
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, (y_a, y_b, lam)
    elif mode == "cutmix":
        W, H = x.size()[2], x.size()[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        mixed_x = x.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[index]
        return mixed_x, (y_a, y_b, lam)
    else:
        raise ValueError(f"Unknown mode {mode}")


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

if __name__ == "__main__":
    print("Testing FocalLoss vs CrossEntropyLoss...")
    ce = nn.CrossEntropyLoss()
    focal = FocalLoss(gamma=0.0)
    logits = torch.randn(4, 9)
    targets = torch.randint(0, 9, (4,))
    l_ce = ce(logits, targets)
    l_focal = focal(logits, targets)
    assert torch.allclose(l_ce, l_focal, atol=1e-5), f"Mismatch: CE={l_ce.item()}, Focal={l_focal.item()}"
    print("-> FocalLoss(gamma=0) matches CE loss!")

    print("Testing mix_batch (CutMix)...")
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 3])
    mixed_x, (y_a, y_b, lam) = mix_batch(x, y, alpha=1.0, mode="cutmix")
    print(f"-> CutMix applied. Lambda = {lam:.4f}")
    
    print("All tests passed in losses.py!")
