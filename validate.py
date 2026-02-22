"""
DiffMICv2 Validation Script
============================
Loads a trained Lightning checkpoint and runs validation
on a random 20% subset of the dataset.

Usage:
    python validate.py --ckpt /path/to/last.ckpt
"""

import os, csv, json, argparse, random
import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix as sklearn_confusion_matrix
)

# Import project modules
import sys
sys.path.insert(0, '/kaggle/working/DiffMICv2')
from diffuser_trainer import CoolSystem
from dataloader.loading import ChestXrayDataSet

# ============================================================
# Config
# ============================================================
RESULTS_DIR = '/kaggle/working/DiffMICv2/validation_results'
os.makedirs(RESULTS_DIR, exist_ok=True)
VAL_SPLIT = 0.2   # use 20% of dataset for validation
SEED = 42

CLASS_NAMES = [
    'Atel', 'Card', 'Effu', 'Infi', 'Mass',
    'Nodu', 'Pneu', 'Pnmx', 'Cons', 'Edem',
    'Emph', 'Fibr', 'PlTh', 'Hern', 'NoFi'
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to the trained .ckpt file')
    parser.add_argument('--config', type=str,
                        default='/kaggle/working/DiffMICv2/configs/placental.yml',
                        help='Path to config YAML')
    parser.add_argument('--split', type=float, default=VAL_SPLIT,
                        help='Fraction of data to use for validation (default: 0.2)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)

    # Load model manually from checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    model = CoolSystem(config)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.cuda()
    print("Model loaded successfully!")

    # Build dataset and select random 20%
    dataset = ChestXrayDataSet(
        csv_file=config.data.testdata,
        data_dir=config.data.data_dir,
        train=False  # no augmentation
    )
    total = len(dataset)
    random.seed(SEED)
    indices = list(range(total))
    random.shuffle(indices)
    val_count = int(total * args.split)
    val_indices = indices[:val_count]
    val_subset = Subset(dataset, val_indices)

    val_loader = DataLoader(
        val_subset,
        batch_size=config.testing.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    print(f"\nDataset: {total} total images")
    print(f"Validation subset: {val_count} images ({args.split*100:.0f}%)")

    # ---- Run inference ----
    print("\nRunning validation...")
    all_gt, all_pred = [], []
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            # Run through aux_model
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = model.aux_model(x_batch)

            # Run diffusion sampling
            with torch.cuda.amp.autocast():
                y_pred = model.DiffSampler(
                    model.model, x_batch, y0_aux, y0_aux_global, y0_aux_local,
                    patches, attns, attn_map
                )
            y_pred = y_pred.mean(2)  # average over diffusion samples

            all_pred.append(y_pred.cpu())
            all_gt.append(y_batch.cpu())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {(batch_idx+1)*config.testing.batch_size}/{val_count} samples")

    gt = torch.cat(all_gt).numpy()
    pred = torch.cat(all_pred).numpy()
    gt_class = np.argmax(gt, axis=1)
    pred_class = np.argmax(pred, axis=1)

    # ---- Compute metrics ----
    acc = accuracy_score(gt_class, pred_class)
    prec = precision_score(gt_class, pred_class, average='macro', zero_division=0)
    rec = recall_score(gt_class, pred_class, average='macro', zero_division=0)
    f1 = f1_score(gt_class, pred_class, average='macro', zero_division=0)
    kappa = cohen_kappa_score(gt_class, pred_class, weights='quadratic')
    try:
        auc = roc_auc_score(gt, pred, average='macro', multi_class='ovr')
    except Exception:
        auc = 0.0

    print("\n" + "="*60)
    print("  VALIDATION RESULTS")
    print("="*60)
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Sensitivity: {rec:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  AUC:         {auc:.4f}")
    print(f"  Kappa:       {kappa:.4f}")

    # Save results JSON
    results = {
        'accuracy': float(acc), 'sensitivity': float(rec),
        'precision': float(prec), 'f1': float(f1),
        'auc': float(auc), 'kappa': float(kappa),
        'val_samples': val_count, 'total_samples': total
    }
    json_path = os.path.join(RESULTS_DIR, 'val_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # ---- Confusion Matrix ----
    cm = sklearn_confusion_matrix(gt_class, pred_class, labels=range(len(CLASS_NAMES)))

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted', fontsize=13)
    plt.ylabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=15)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()

    np.savetxt(os.path.join(RESULTS_DIR, 'confusion_matrix.csv'),
               cm, delimiter=',', header=','.join(CLASS_NAMES), fmt='%d')

    # ---- Per-class TP/TN/FP/FN ----
    print("\nPer-class metrics:")
    print(f"{'Class':<6} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6} {'Sens':>8} {'Spec':>8}")
    print("-" * 50)
    rows = []
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"{cls:<6} {tp:>6} {tn:>6} {fp:>6} {fn:>6} {sens:>8.4f} {spec:>8.4f}")
        rows.append({'class': cls, 'TP': int(tp), 'TN': int(tn),
                     'FP': int(fp), 'FN': int(fn),
                     'sensitivity': sens, 'specificity': spec})

    # Save per-class CSV
    csv_path = os.path.join(RESULTS_DIR, 'per_class_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Save TP, TN, FP, FN as separate CSVs
    for metric_name in ['TP', 'TN', 'FP', 'FN']:
        path = os.path.join(RESULTS_DIR, f'{metric_name}.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', metric_name])
            for row in rows:
                writer.writerow([row['class'], row[metric_name]])

    print(f"\nConfusion matrix saved to {cm_path}")
    print(f"Per-class metrics saved to {csv_path}")
    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
