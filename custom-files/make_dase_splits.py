"""

Build the Distribution-Aware Stratified Ensembling (DASE) splits for
Dataset107_PDAC_Detection, as described in PanDx §2.2.

* For every training case, compute lesion size = #voxels where label == 1
  (the PDAC class).
* PDAC-positive cases (size > 0) are split into 4 quartile bins by lesion size.
* PDAC-negative cases (size == 0) form a 5th bin.
* Each bin is shuffled and dealt round-robin into 5 folds. This guarantees
  each fold has (a) the same global pos/neg ratio and (b) the same lesion
  size distribution among its positives.
* Output: nnU-Net's expected `splits_final.json`
  = a list of 5 dicts with "train" / "val" keys (case IDs, no extensions).

"""

import json
import os
import os.path as osp
from glob import glob

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

DATASET_NAME = "Dataset107_PDAC_Detection"
PDAC_LABEL   = 1
N_FOLDS      = 5
SEED         = 42


def compute_lesion_sizes(label_dir, cache_fp):
    """Return {case_id: lesion_voxel_count}. Caches to cache_fp."""
    cache = {}
    if osp.isfile(cache_fp):
        with open(cache_fp) as f:
            cache = json.load(f)

    lbl_fps = sorted(glob(osp.join(label_dir, "*.nii.gz")))
    sizes = {}
    needs_save = False

    for lbl_fp in tqdm(lbl_fps, desc="reading labels"):
        cid = osp.basename(lbl_fp).replace(".nii.gz", "")
        if cid in cache:
            sizes[cid] = cache[cid]
            continue
        arr = sitk.GetArrayFromImage(sitk.ReadImage(lbl_fp))
        sizes[cid] = int((arr == PDAC_LABEL).sum())
        cache[cid] = sizes[cid]
        needs_save = True

    if needs_save:
        with open(cache_fp, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"cached lesion sizes to {cache_fp}")
    return sizes


def assign_round_robin(case_list, n_folds, rng):
    """Shuffle case_list and deal cards into n_folds buckets. Returns list of
    n_folds lists."""
    cases = list(case_list)
    rng.shuffle(cases)
    buckets = [[] for _ in range(n_folds)]
    for i, c in enumerate(cases):
        buckets[i % n_folds].append(c)
    return buckets


def main():
    nnUNet_raw          = os.environ["nnUNet_raw"]
    nnUNet_preprocessed = os.environ["nnUNet_preprocessed"]

    label_dir  = osp.join(nnUNet_raw, DATASET_NAME, "labelsTr")
    preproc    = osp.join(nnUNet_preprocessed, DATASET_NAME)
    cache_fp   = osp.join(preproc, "lesion_sizes_cache.json")
    out_fp     = osp.join(preproc, "splits_final.json")
    os.makedirs(preproc, exist_ok=True)

    # ---- 1. lesion size per case ---------------------------------------------
    sizes = compute_lesion_sizes(label_dir, cache_fp)
    cids  = np.array(sorted(sizes.keys()))
    sz    = np.array([sizes[c] for c in cids])

    positives = cids[sz > 0]
    negatives = cids[sz == 0]
    pos_sizes = sz[sz > 0]
    print(f"\ntotal cases    : {len(cids)}")
    print(f"PDAC positive  : {len(positives)}")
    print(f"PDAC negative  : {len(negatives)}")

    # ---- 2. quartile-bin the positives ---------------------------------------
    q = np.quantile(pos_sizes, [0.25, 0.50, 0.75])
    print(f"\nlesion size quartile cutoffs (voxels): "
          f"Q1={q[0]:.0f}  Q2={q[1]:.0f}  Q3={q[2]:.0f}")

    bin_assign = np.where(pos_sizes <= q[0], 0,
                  np.where(pos_sizes <= q[1], 1,
                   np.where(pos_sizes <= q[2], 2, 3)))

    rng = np.random.RandomState(SEED)

    # ---- 3. deal each bin into 5 folds ---------------------------------------
    fold_val_cases = [[] for _ in range(N_FOLDS)]
    for b in range(4):
        bin_cases = positives[bin_assign == b]
        buckets = assign_round_robin(bin_cases, N_FOLDS, rng)
        for f in range(N_FOLDS):
            fold_val_cases[f].extend(buckets[f])
        print(f"  bin {b}: {len(bin_cases)} cases -> "
              f"per-fold val counts {[len(b) for b in buckets]}")

    # negatives -> round-robin separately
    neg_buckets = assign_round_robin(negatives, N_FOLDS, rng)
    for f in range(N_FOLDS):
        fold_val_cases[f].extend(neg_buckets[f])
    print(f"  negatives: {len(negatives)} cases -> "
          f"per-fold val counts {[len(b) for b in neg_buckets]}")

    # ---- 4. build splits_final.json ------------------------------------------
    all_cases = set(cids.tolist())
    splits = []
    for f in range(N_FOLDS):
        val = sorted(fold_val_cases[f])
        train = sorted(all_cases - set(val))
        splits.append({"train": train, "val": val})

    with open(out_fp, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nwrote {out_fp}")

    # ---- 5. sanity report -----------------------------------------------------
    print("\nper-fold composition:")
    print(f"  {'fold':<6}{'train':>8}{'val':>8}{'val+':>8}{'val-':>8}")
    for f in range(N_FOLDS):
        val = splits[f]["val"]
        val_pos = sum(1 for c in val if sizes[c] > 0)
        val_neg = len(val) - val_pos
        print(f"  {f:<6}{len(splits[f]['train']):>8}{len(val):>8}"
              f"{val_pos:>8}{val_neg:>8}")


if __name__ == "__main__":
    main()
