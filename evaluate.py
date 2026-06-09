"""
python evaluate.py --pred_dir ./workspace/workspace/test_example/output/pdac-detection-map --label_dir ./labels --threshold 0.5
"""

import argparse
import os
import os.path as osp
import numpy as np
import SimpleITK as sitk
from glob import glob
import warnings
warnings.filterwarnings("ignore")


def load_nifti(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return arr.astype(np.float32)


def get_case_id(filename):
    base = os.path.basename(filename)
    for ext in [".nii.gz", ".nii", ".mha", ".mhd"]:
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    if base.endswith("_0000"):
        base = base[:-5]
    return base


def match_files(pred_dir, label_dir):
    pred_files = [f for f in glob(osp.join(pred_dir, "*")) if osp.isfile(f)]
    label_files = [f for f in glob(osp.join(label_dir, "*")) if osp.isfile(f)]

    label_map = {get_case_id(f): f for f in label_files}

    pairs = []
    for pred_fp in pred_files:
        case_id = get_case_id(pred_fp)
        if case_id in label_map:
            pairs.append((case_id, pred_fp, label_map[case_id]))
        else:
            pass

    for case_id in label_map:
        pass

    return pairs


def evaluate_case(pred_arr, label_arr, threshold):
    gt = (label_arr == 1)
    pred = (pred_arr >= threshold)

    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    tn = int(np.sum(~pred & ~gt))
    fn = int(np.sum(~pred & gt))

    return tp, fp, tn, fn


def compute_metrics(tp, fp, tn, fn):
    tpr  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    return tpr, fpr, dice


def compute_curve(loaded_cases, thresholds):
    results = []
    for thresh in thresholds:
        total_tp = total_fp = total_tn = total_fn = 0
        for _, pred_arr, label_arr in loaded_cases:
            tp, fp, tn, fn = evaluate_case(pred_arr, label_arr, thresh)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
        tpr, fpr, dice = compute_metrics(total_tp, total_fp, total_tn, total_fn)
        results.append((thresh, tpr, fpr, dice))
    return results


def run(pred_dir, label_dir, threshold):
    pairs = match_files(pred_dir, label_dir)
    if not pairs:
        return

    loaded_cases = [
        (case_id, load_nifti(pred_fp), load_nifti(label_fp))
        for case_id, pred_fp, label_fp in pairs
    ]

    tpr_vals, fpr_vals, dice_vals = [], [], []
    for _, pred, label in loaded_cases:
        tp, fp, tn, fn = evaluate_case(pred, label, threshold)
        tpr, fpr, dice = compute_metrics(tp, fp, tn, fn)
        if not np.isnan(tpr):
            tpr_vals.append(tpr)
        if not np.isnan(fpr):
            fpr_vals.append(fpr)
        if not np.isnan(dice):
            dice_vals.append(dice)

    avg_tpr  = np.mean(tpr_vals)  if tpr_vals  else float("nan")
    avg_fpr  = np.mean(fpr_vals)  if fpr_vals  else float("nan")
    avg_dice = np.mean(dice_vals) if dice_vals else float("nan")
    print(f"TPR: {avg_tpr:.3f}  FPR: {avg_fpr:.3f}  Dice: {avg_dice:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PDAC detection evaluation")
    parser.add_argument("--pred_dir",  type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    run(args.pred_dir, args.label_dir, args.threshold)
