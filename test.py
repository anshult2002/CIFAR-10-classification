"""Evaluate trained models on the CIFAR-10 test set (closed-set accuracy).

Usage:
    python test.py --model vit
    python test.py --model resnet
    python test.py --model both
Requires weights saved by train.py.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from common import (MODELS, CLASSES, NUM_CLASSES, OUT_DIR, device, add_data_args,
                    loaders_from_args, load_trained_model, resolve_models, set_seed)


@torch.no_grad()
def evaluate(model, loader):
    """Returns dict with test accuracy, mean loss, per-class accuracy and confusion matrix."""
    model.eval()
    preds, labels, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
        preds.append(logits.argmax(1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)  # rows = true, cols = predicted
    for t, p in zip(labels, preds):
        conf[t, p] += 1
    per_class = {CLASSES[c]: float(conf[c, c] / max(1, conf[c].sum())) for c in range(NUM_CLASSES)}
    return {
        'test_acc': float((preds == labels).mean()),
        'test_loss': loss_sum / len(labels),
        'per_class_acc': per_class,
        'confusion_matrix': conf.tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=['vit', 'resnet', 'both'], required=True)
    add_data_args(p)
    args = p.parse_args()

    set_seed()
    loaders = loaders_from_args(args)
    summary = {}

    for key in resolve_models(args.model):
        info = MODELS[key]
        model = load_trained_model(key)
        res = evaluate(model, loaders['test'])

        hist_path = os.path.join(OUT_DIR, f"{key}_history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                hist = json.load(f)
            res['best_val_acc'] = max(hist['val_acc']) if hist['val_acc'] else None
            res['epochs_trained'] = len(hist['val_acc'])
        res['checkpoint_name'] = info['ckpt_name']

        print(f"\n== {info['title']} ==")
        print(f"  Test accuracy: {res['test_acc']:.4f}   (loss {res['test_loss']:.4f})")
        for c, a in res['per_class_acc'].items():
            print(f"    {c:<11s} {a:.4f}")

        out_path = os.path.join(OUT_DIR, f"{key}_test_results.json")
        with open(out_path, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"  Saved {out_path}")
        summary[key] = res['test_acc']

    if len(summary) > 1:
        print("\nSummary:", "  |  ".join(f"{MODELS[k]['title']}: {v:.4f}" for k, v in summary.items()))


if __name__ == '__main__':
    main()
