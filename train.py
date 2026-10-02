"""Train ViT and/or ResNet-50 from scratch on CIFAR-10.

Usage:
    python train.py --model vit
    python train.py --model resnet
    python train.py --model both
Re-running after an interruption resumes from the last saved epoch automatically.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import (MODELS, NUM_CLASSES, CKPT_DIR, OUT_DIR, device, add_data_args,
                    loaders_from_args, load_checkpoint, save_checkpoint, resolve_models, set_seed)

# Per-model training recipes (same augmentation + MixUp/CutMix for both)
TRAIN_CONFIG = {
    'vit': dict(epochs=100, lr=1e-3, weight_decay=0.05, label_smoothing=0.1, patience=15,
                warmup_epochs=7, grad_clip=1.0, mixup_alpha=0.2, cutmix_alpha=1.0),
    'resnet': dict(epochs=100, lr=1e-3, weight_decay=5e-4, label_smoothing=0.1, patience=15,
                   warmup_epochs=5, grad_clip=5.0, mixup_alpha=0.2, cutmix_alpha=1.0),
}


# ----------------------------------------------------------------------------- MixUp / CutMix
def one_hot_smooth(y, num_classes, smoothing=0.0):
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    y_oh = torch.full((y.size(0), num_classes), off, device=y.device)
    y_oh.scatter_(1, y.unsqueeze(1), on)
    return y_oh


def soft_target_ce(logits, target_probs):
    return -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = int(np.clip(cx - cut_w // 2, 0, W)); x2 = int(np.clip(cx + cut_w // 2, 0, W))
    y1 = int(np.clip(cy - cut_h // 2, 0, H)); y2 = int(np.clip(cy + cut_h // 2, 0, H))
    return x1, y1, x2, y2


def mixup_cutmix(x, y, num_classes, smoothing, mixup_alpha, cutmix_alpha, switch_prob=0.5):
    """MixUp or CutMix (chosen per batch). Returns (mixed_x, soft_target)."""
    target = one_hot_smooth(y, num_classes, smoothing)
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return x, target
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < switch_prob)
    perm = torch.randperm(x.size(0), device=x.device)
    if use_cutmix:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        H, W = x.shape[2], x.shape[3]
        x1, y1, x2, y2 = rand_bbox(W, H, lam)
        x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    else:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        x = lam * x + (1.0 - lam) * x[perm]
    return x, lam * target + (1.0 - lam) * target[perm]


# ----------------------------------------------------------------------------- training loop
def train_model(name, model, loaders, epochs, lr, weight_decay=5e-4, label_smoothing=0.1,
                use_amp=True, patience=6, warmup_epochs=0, grad_clip=None,
                mixup_alpha=0.0, cutmix_alpha=0.0, num_classes=NUM_CLASSES):
    model.to(device)
    use_amp = use_amp and device.type == 'cuda'
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    use_mix = mixup_alpha > 0 or cutmix_alpha > 0
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_epoch, best_acc, history = load_checkpoint(name, model, optimizer, scheduler)
    epochs_no_improve = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running_loss, running_correct, n = 0.0, 0, 0
        for x, y in loaders['train']:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if use_mix:
                    x_mixed, target = mixup_cutmix(x, y, num_classes, label_smoothing,
                                                   mixup_alpha, cutmix_alpha)
                    out = model(x_mixed)
                    loss = soft_target_ce(out, target)
                else:
                    out = model(x)
                    loss = criterion(out, y)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * x.size(0)
            # Train acc is vs. the ORIGINAL labels even when the batch was mixed, so it is an
            # approximate/optimistic progress signal under MixUp/CutMix.
            running_correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
        train_loss, train_acc = running_loss / n, running_correct / n
        scheduler.step()

        # validation (always clean, un-mixed)
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in loaders['val']:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        val_loss /= total
        val_acc = correct / total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        save_checkpoint(name, epoch + 1, model, optimizer, scheduler, best_acc, history)
        print(f"[{name}] epoch {epoch+1}/{epochs} | train_loss {train_loss:.4f} | "
              f"train_acc {train_acc:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | "
              f"best {best_acc:.4f} | {time.time()-t0:.0f}s")

        if epochs_no_improve >= patience:
            print(f"Early stopping '{name}' at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    best_path = os.path.join(CKPT_DIR, f"{name}_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    return model, history


def plot_history(history, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history['train_loss'], label='train'); ax[0].plot(history['val_loss'], label='val')
    ax[0].set_title(f'{title} - loss'); ax[0].set_xlabel('epoch'); ax[0].legend()
    ax[1].plot(history['train_acc'], label='train_acc'); ax[1].plot(history['val_acc'], label='val_acc')
    ax[1].set_title(f'{title} - accuracy'); ax[1].set_xlabel('epoch'); ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved curves to {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=['vit', 'resnet', 'both'], required=True)
    p.add_argument('--epochs', type=int, default=None, help='Override the default epoch count.')
    add_data_args(p)
    args = p.parse_args()

    set_seed()
    print("device:", device)
    loaders = loaders_from_args(args)

    for key in resolve_models(args.model):
        cfg = dict(TRAIN_CONFIG[key])
        if args.epochs is not None:
            cfg['epochs'] = args.epochs
        info = MODELS[key]
        model = info['build']().to(device)
        print(f"\n=== Training {info['title']} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params) ===")
        model, history = train_model(info['ckpt_name'], model, loaders, **cfg)

        with open(os.path.join(OUT_DIR, f"{key}_history.json"), 'w') as f:
            json.dump(history, f, indent=2)
        plot_history(history, info['title'], os.path.join(OUT_DIR, f"{key}_curves.png"))


if __name__ == '__main__':
    main()
