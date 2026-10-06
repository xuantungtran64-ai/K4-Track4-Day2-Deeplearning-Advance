"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F)."""
from __future__ import annotations

import random
import numpy as np
import torch
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt

@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    from model import param_groups
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.buffers = {}
        for name, buf in model.named_buffers():
            self.buffers[name] = buf.data.clone()
            
    def update(self, model) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].copy_(self.decay * self.shadow[name] + (1.0 - self.decay) * param.data)
            for name, buf in model.named_buffers():
                self.buffers[name].copy_(buf.data)
                
    def copy_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])
        for name, buf in model.named_buffers():
            buf.data.copy_(self.buffers[name])


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    model.train()
    if cfg.init == "frozen":
        from model import freeze_backbone
        freeze_backbone(model)
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

    total_loss = 0.0
    from losses import mix_batch, mixed_loss
    
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            if cfg.mix:
                images, mixed_targets = mix_batch(images, labels, cfg.mix_alpha, cfg.mix)
                logits = model(images)
                loss = mixed_loss(criterion, logits, mixed_targets)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if ema:
            ema.update(model)
            
        total_loss += loss.item()
        
    return {"train_loss": total_loss / len(loader), "lr": optimizer.param_groups[0]['lr']}


def evaluate(model, loader, criterion, device):
    model.eval()
    all_filenames = []
    all_labels = []
    all_logits = []
    total_loss = 0.0
    
    with torch.inference_mode():
        for images, labels, filenames in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            all_filenames.extend(filenames)
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            
    all_logits = np.concatenate(all_logits, axis=0)
    return all_filenames, np.array(all_labels), all_logits, total_loss / len(loader)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_f1"] for h in history]
    
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, train_loss, label='Train Loss', color='tab:red')
    ax1.plot(epochs, val_loss, label='Val Loss', color='tab:orange')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Macro-F1')
    ax2.plot(epochs, val_f1, label='Val F1', color='tab:blue')
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right')
    
    plt.title(title)
    fig.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def run(cfg: Config) -> dict:
    from dataset import load_split, check_split, build_transforms, make_loader
    from model import build_model, count_params, count_gmacs
    from losses import build_criterion, class_weights
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, '.')
    from eval import save_predictions
    import pandas as pd
    from sklearn.metrics import f1_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    
    r_dir = run_dir(cfg)
    r_dir.mkdir(parents=True, exist_ok=True)
    with open(r_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
        
    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    check_split(train_df, val_df, test_df, cfg.images_dir)
    
    train_t = build_transforms(True, cfg.img_size, cfg.aug)
    val_t = build_transforms(False, cfg.img_size, cfg.aug)
    
    train_loader = make_loader(train_df, cfg.images_dir, train_t, cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    val_loader = make_loader(val_df, cfg.images_dir, val_t, cfg.batch_size, False, None, cfg.num_workers)
    
    model = build_model(cfg.backbone, True, 9, cfg.drop_rate, cfg.init).to(device)
    
    cw = None
    if cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().sort_index().to_dict()
        cw = class_weights(counts, beta=cfg.class_weight_beta or 0.0).to(device)
        
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma, weight=cw).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    
    history = []
    best_f1 = -1
    best_epoch = -1
    best_model_state = None
    
    for epoch in range(cfg.epochs):
        t0 = time.time()
        t_res = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        
        eval_model = model
        if ema:
            eval_model = build_model(cfg.backbone, False, 9, 0, "finetune").to(device)
            eval_model.load_state_dict(model.state_dict())
            ema.copy_to(eval_model)
            
        filenames, y_true, logits, val_loss = evaluate(eval_model, val_loader, criterion, device)
        preds = np.argmax(logits, axis=1)
        
        val_f1 = f1_score(y_true, preds, average='macro')
        
        ep_time = time.time() - t0
        h_dict = {"epoch": epoch, "train_loss": t_res["train_loss"], "val_loss": val_loss, "val_f1": val_f1, "lr": t_res["lr"], "time": ep_time}
        history.append(h_dict)
        print(f"Epoch {epoch} - val_loss: {val_loss:.4f}, val_macro_f1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_model_state = eval_model.state_dict().copy()
            
    # Load best model for saving
    model.load_state_dict(best_model_state)
    filenames, y_true, val_logits, _ = evaluate(model, val_loader, criterion, device)
    
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    val_probs = torch.softmax(torch.tensor(val_logits), dim=1).numpy()
    save_predictions(pred_path(cfg, "val"), filenames, y_true, val_probs)
    
    if cfg.save_test_predictions:
        test_loader = make_loader(test_df, cfg.images_dir, val_t, cfg.batch_size, False, None, cfg.num_workers)
        test_filenames, test_y_true, test_logits, _ = evaluate(model, test_loader, criterion, device)
        test_probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
        save_predictions(pred_path(cfg, "test"), test_filenames, test_y_true, test_probs)
        
    pd.DataFrame(history).to_csv(r_dir / "history.csv", index=False)
    
    curves_dir = Path("curves")
    curves_dir.mkdir(exist_ok=True)
    plot_curves(history, curves_dir / f"{cfg.exp_id}.png", f"{cfg.exp_id} - {cfg.backbone}")
    
    return {
        "best_epoch": best_epoch,
        "macro_f1": best_f1,
        "time_per_epoch": sum(h["time"] for h in history) / len(history),
        "params": count_params(model),
        "gmacs": count_gmacs(model, cfg.img_size)
    }


def parse_overrides(pairs: list[str]) -> dict:
    res = {}
    for pair in pairs:
        if "=" not in pair: continue
        k, v = pair.split("=", 1)
        if v.lower() == "true": v = True
        elif v.lower() == "false": v = False
        elif v.lower() == "none": v = None
        else:
            try: v = int(v)
            except ValueError:
                try: v = float(v)
                except ValueError: pass
        res[k] = v
    return res


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="+", default=[])
    args = parser.parse_args()
    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    res = run(cfg)
    print(res)


if __name__ == "__main__":
    main()
