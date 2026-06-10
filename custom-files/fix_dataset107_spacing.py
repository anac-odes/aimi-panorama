"""

This script repairs Dataset107_PDAC_Detection so `nnUNetv2_plan_and_preprocess
--verify_dataset_integrity` passes. 

It does two operations:

  1. For each image/label pair where the metadata only differs by tiny
     floating-point drift (spacing within 1e-3 relative, origin within 0.5 mm,
     identical direction), copy the image's metadata onto the label. The grids align,
     there is just a small numerical mismatch that raises errors.

  2. For each pair where the mismatch is real (direction flip, large origin
     offset, shape mismatch), move both files into a `_dropped/`
     folder and decrement `dataset.json["numTraining"]`. These cases cannot be
     trusted -- the seg is geometrically inconsistent with the image.

"""

import json
import os
import os.path as osp
import shutil
from glob import glob

import numpy as np
import SimpleITK as sitk


SPACING_REL_TOL = 1e-3   # max relative drift per axis to consider "snappable"
ORIGIN_ABS_TOL  = 0.5    # mm -- larger than this is treated as real misalign
DIRECTION_TOL   = 1e-3   # any direction-cosine difference > this = real flip

DATASET_NAME = "Dataset107_PDAC_Detection"


def metadata_diff(img: sitk.Image, seg: sitk.Image):
    """Return (snappable, hard_reason). snappable=True if drift is within
    tolerance. If snappable=False, hard_reason is a short string."""
    img_sp = np.array(img.GetSpacing())
    seg_sp = np.array(seg.GetSpacing())
    img_or = np.array(img.GetOrigin())
    seg_or = np.array(seg.GetOrigin())
    img_dr = np.array(img.GetDirection())
    seg_dr = np.array(seg.GetDirection())

    if img.GetSize() != seg.GetSize():
        return False, f"shape mismatch ({img.GetSize()} vs {seg.GetSize()})"

    if not np.allclose(img_dr, seg_dr, atol=DIRECTION_TOL):
        return False, f"direction mismatch (img {img_dr.tolist()} vs seg {seg_dr.tolist()})"

    if (np.abs(img_sp - seg_sp) / img_sp).max() > SPACING_REL_TOL:
        return False, f"large spacing drift ({img_sp.tolist()} vs {seg_sp.tolist()})"

    if np.abs(img_or - seg_or).max() > ORIGIN_ABS_TOL:
        return False, f"large origin drift ({img_or.tolist()} vs {seg_or.tolist()})"

    return True, None


def main():
    nnUNet_raw = os.environ["nnUNet_raw"]
    root  = osp.join(nnUNet_raw, DATASET_NAME)
    imgsd = osp.join(root, "imagesTr")
    lblsd = osp.join(root, "labelsTr")
    drop  = osp.join(root, "_dropped")
    os.makedirs(drop, exist_ok=True)
    os.makedirs(osp.join(drop, "imagesTr"), exist_ok=True)
    os.makedirs(osp.join(drop, "labelsTr"), exist_ok=True)

    img_fps = sorted(glob(osp.join(imgsd, "*_0000.nii.gz")))
    print(f"scanning {len(img_fps)} image/label pairs in {root}\n")

    n_ok, n_snapped, n_dropped, n_missing = 0, 0, 0, 0
    snapped_cases, dropped_cases, missing_cases = [], [], []

    for img_fp in img_fps:
        case_id = osp.basename(img_fp).replace("_0000.nii.gz", "")
        lbl_fp = osp.join(lblsd, case_id + ".nii.gz")

        if not osp.isfile(lbl_fp):
            print(f"[missing-label] {case_id} -- moving image aside")
            shutil.move(img_fp, osp.join(drop, "imagesTr", osp.basename(img_fp)))
            n_missing += 1
            missing_cases.append(case_id)
            continue

        img = sitk.ReadImage(img_fp)
        seg = sitk.ReadImage(lbl_fp)

        if (img.GetSpacing() == seg.GetSpacing()
                and img.GetOrigin() == seg.GetOrigin()
                and img.GetDirection() == seg.GetDirection()
                and img.GetSize() == seg.GetSize()):
            n_ok += 1
            continue

        snappable, reason = metadata_diff(img, seg)
        if snappable:
            seg.SetSpacing(img.GetSpacing())
            seg.SetOrigin(img.GetOrigin())
            seg.SetDirection(img.GetDirection())
            sitk.WriteImage(seg, lbl_fp)
            n_snapped += 1
            snapped_cases.append(case_id)
            print(f"[snap]  {case_id}")
        else:
            shutil.move(img_fp, osp.join(drop, "imagesTr", osp.basename(img_fp)))
            shutil.move(lbl_fp, osp.join(drop, "labelsTr", osp.basename(lbl_fp)))
            n_dropped += 1
            dropped_cases.append((case_id, reason))
            print(f"[drop]  {case_id} -- {reason}")

    json_fp = osp.join(root, "dataset.json")
    with open(json_fp) as f:
        meta = json.load(f)
    old = meta.get("numTraining", None)
    new = len(sorted(glob(osp.join(imgsd, "*_0000.nii.gz"))))
    if old != new:
        meta["numTraining"] = new
        with open(json_fp, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nupdated dataset.json: numTraining {old} -> {new}")
    else:
        print(f"\ndataset.json numTraining already correct ({new})")


    print("\n" + "=" * 60)
    print(f"clean (untouched)      : {n_ok}")
    print(f"metadata snapped       : {n_snapped}")
    print(f"dropped (real mismatch): {n_dropped}")
    print(f"dropped (no label)     : {n_missing}")
    print(f"remaining for training : {new}")
    print("=" * 60)
    if dropped_cases:
        print("\nDropped cases (moved to _dropped/):")
        for cid, why in dropped_cases:
            print(f"  - {cid}: {why}")


if __name__ == "__main__":
    main()
