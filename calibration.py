"""Out-of-distribution rejection: threshold calibration + OOD-detection metrics.

For each model and each score (MSP, Energy):
  1. Calibrate a rejection threshold tau on the CIFAR-10 *validation* split so that
     `--target-tpr` (default 95%) of in-distribution samples are accepted.
  2. Apply tau to the CIFAR-10 test set (ID) and CIFAR-100 test set (surrogate OOD) and report
     AUROC (ID vs OOD), FPR @ 95% TPR, and open-set accuracy.

Usage:
    python calibration.py --model vit
    python calibration.py --model resnet
    python calibration.py --model both
Requires weights saved by train.py.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from common import (MODELS, OUT_DIR, device, add_data_args, loaders_from_args,
                    load_trained_model, resolve_models, set_seed)


# Both scores: higher = more in-distribution, so one metric code path serves both.
@torch.no_grad()
def msp_scores(model, loader):
    """Max softmax probability."""
    model.eval()
    scores, preds, labels = [], [], []
    for x, y in loader:
        probs = F.softmax(model(x.to(device)), dim=1)
        p_max, pred = probs.max(1)
        scores.append(p_max.cpu()); preds.append(pred.cpu()); labels.append(y)
    return torch.cat(scores).numpy(), torch.cat(preds).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def energy_scores(model, loader):
    """logsumexp(logits) (Liu et al., 2020)."""
    model.eval()
    scores, preds, labels = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        scores.append(torch.logsumexp(logits, dim=1).cpu())
        preds.append(logits.argmax(dim=1).cpu()); labels.append(y)
    return torch.cat(scores).numpy(), torch.cat(preds).numpy(), torch.cat(labels).numpy()


SCORE_FNS = {'MSP': msp_scores, 'Energy': energy_scores}


def calibrate_and_eval(model, name, score_fn, loaders, target_tpr=0.95):
    val_scores, _, _ = score_fn(model, loaders['val'])
    tau = float(np.quantile(val_scores, 1 - target_tpr))  # accept target_tpr of ID val samples

    test_scores, test_preds, test_labels = score_fn(model, loaders['test'])
    ood_scores, _, _ = score_fn(model, loaders['ood'])

    y_true = np.concatenate([np.ones_like(test_scores), np.zeros_like(ood_scores)])  # 1 = ID
    y_score = np.concatenate([test_scores, ood_scores])
    auroc = float(roc_auc_score(y_true, y_score))

    # FPR at the threshold that keeps 95% of ID *test* samples
    fpr95_thresh = np.quantile(test_scores, 0.05)
    fpr95 = float((ood_scores >= fpr95_thresh).mean())

    # Open-set accuracy with the val-calibrated tau: an ID sample counts as correct if it is
    # accepted AND correctly classified; an OOD sample counts as correct if it is rejected.
    accepted_test = test_scores >= tau
    open_set_correct = int(((test_preds == test_labels) & accepted_test).sum())
    rejected_ood = int((ood_scores < tau).sum())
    open_set_acc = (open_set_correct + rejected_ood) / (len(test_labels) + len(ood_scores))

    print(f"  -- {name} --")
    print(f"     tau               : {tau:.4f}")
    print(f"     AUROC (ID vs OOD) : {auroc:.4f}")
    print(f"     FPR @ 95% TPR     : {fpr95:.4f}")
    print(f"     Open-set accuracy : {open_set_acc:.4f}")
    return {'tau': tau, 'auroc': auroc, 'fpr95': fpr95, 'open_set_acc': open_set_acc,
            'id_accept_rate_test': float(accepted_test.mean()),
            'ood_reject_rate': rejected_ood / len(ood_scores)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=['vit', 'resnet', 'both'], required=True)
    p.add_argument('--target-tpr', type=float, default=0.95,
                   help='Fraction of ID validation samples the threshold should accept.')
    add_data_args(p)
    args = p.parse_args()

    set_seed()
    loaders = loaders_from_args(args)
    all_results = {}

    for key in resolve_models(args.model):
        info = MODELS[key]
        model = load_trained_model(key)
        print(f"\n== {info['title']} ==")
        res = {name: calibrate_and_eval(model, name, fn, loaders, args.target_tpr)
               for name, fn in SCORE_FNS.items()}
        res_out = {'checkpoint_name': info['ckpt_name'], 'target_tpr': args.target_tpr,
                   'ood_detection': {k.lower(): v for k, v in res.items()}}
        out_path = os.path.join(OUT_DIR, f"{key}_calibration.json")
        with open(out_path, 'w') as f:
            json.dump(res_out, f, indent=2)
        print(f"  Saved {out_path}")
        all_results[key] = res

    if len(all_results) > 1:
        print("\n== Comparison (AUROC / FPR@95TPR / open-set acc) ==")
        for score in SCORE_FNS:
            for key, res in all_results.items():
                r = res[score]
                print(f"  {score:<6s} {MODELS[key]['title']:<26s} "
                      f"{r['auroc']:.4f} / {r['fpr95']:.4f} / {r['open_set_acc']:.4f}")


if __name__ == '__main__':
    main()
