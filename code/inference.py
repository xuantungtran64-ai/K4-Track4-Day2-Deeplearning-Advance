"""inference.py - các phương pháp suy luận."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize

def predict_logits(model, loader, device, view=None):
    model.eval()
    all_filenames = []
    all_labels = []
    all_logits = []
    
    with torch.inference_mode():
        for images, labels, filenames in loader:
            images = images.to(device)
            if view:
                images = view(images)
            logits = model(images)
            
            all_filenames.extend(filenames)
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            
    return all_filenames, np.array(all_labels), np.concatenate(all_logits, axis=0)


def view_identity(x):
    return x


def view_hflip(x):
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int):
    B, C, H, W = x.shape
    crops = []
    # top left
    crops.append(x[:, :, :crop, :crop])
    # top right
    crops.append(x[:, :, :crop, -crop:])
    # bottom left
    crops.append(x[:, :, -crop:, :crop])
    # bottom right
    crops.append(x[:, :, -crop:, -crop:])
    # center
    ch, cw = H//2, W//2
    crops.append(x[:, :, ch-crop//2:ch+crop//2, cw-crop//2:cw+crop//2])
    return crops


def views_multiscale(x, sizes):
    return [F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False) for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    if space == "prob":
        probs = [F.softmax(torch.tensor(l), dim=-1).numpy() for l in logits_per_view]
        return np.mean(probs, axis=0)
    else:
        avg_logits = np.mean(logits_per_view, axis=0)
        return F.softmax(torch.tensor(avg_logits), dim=-1).numpy()


def ensemble_probs(list_of_probs):
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    val_logits_t = torch.tensor(val_logits)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long)
    
    def eval_nll(T):
        probs = F.softmax(val_logits_t / T[0], dim=-1)
        return F.nll_loss(torch.log(probs + 1e-12), val_labels_t).item()
        
    res = minimize(eval_nll, [1.0], bounds=[(0.01, 10.0)], method='Nelder-Mead')
    return float(res.x[0])


def apply_temperature(logits, T: float):
    return F.softmax(torch.tensor(logits) / T, dim=-1).numpy()


def fuse_conv_bn(model):
    model.eval()
    for child_name, child in model.named_children():
        if isinstance(child, nn.Sequential):
            for i in range(len(child)-1):
                m1, m2 = child[i], child[i+1]
                if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d):
                    fused = nn.Conv2d(m1.in_channels, m1.out_channels, m1.kernel_size,
                                      m1.stride, m1.padding, m1.dilation, m1.groups, bias=True)
                    
                    w_conv = m1.weight.clone().view(m1.out_channels, -1)
                    w_bn = torch.diag(m2.weight.div(torch.sqrt(m2.eps + m2.running_var)))
                    
                    fused.weight.data = torch.mm(w_bn, w_conv).view(fused.weight.size())
                    
                    if m1.bias is not None:
                        b_conv = m1.bias
                    else:
                        b_conv = torch.zeros(m1.weight.size(0))
                    
                    b_bn = m2.bias - m2.weight.mul(m2.running_mean).div(torch.sqrt(m2.running_var + m2.eps))
                    fused.bias.data = torch.mm(w_bn, b_conv.view(-1, 1)).view(-1) + b_bn
                    
                    child[i] = fused
                    child[i+1] = nn.Identity()
        else:
            fuse_conv_bn(child)
    return model

if __name__ == "__main__":
    import copy
    import timm

    print("Testing fuse_conv_bn...")
    # Khởi tạo model có Conv + BN
    model = timm.create_model("resnet18", pretrained=False, num_classes=9)
    model.eval()
    
    # Clone model để so sánh
    model_fused = copy.deepcopy(model)
    model_fused = fuse_conv_bn(model_fused)
    model_fused.eval()
    
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_orig = model(x)
        out_fused = model_fused(x)
        
    diff = torch.abs(out_orig - out_fused).max().item()
    assert diff < 1e-4, f"Mismatch after fusion! Max diff: {diff}"
    print(f"-> fuse_conv_bn test passed! Max diff = {diff:.6f}")
