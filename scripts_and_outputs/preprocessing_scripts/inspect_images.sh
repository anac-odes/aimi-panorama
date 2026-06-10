#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
#SBATCH --mem=4G
#SBATCH --output=inspect_images.out
#SBATCH --time=00:10:00

source /vol/csedu-nobackup/course/IMC037_aimi/group03/analeopold/miniconda3/etc/profile.d/conda.sh
conda activate panorama

REPO=/vol/csedu-nobackup/course/IMC037_aimi/group03/analeopold/aimi-panorama

export nnUNet_raw="$REPO/workspace/nnUNet_raw"
export nnUNet_preprocessed="$REPO/workspace/nnUNet_preprocessed"
export nnUNet_results="$REPO/workspace/nnUNet_results"

python <<'PYEOF'
import SimpleITK as sitk
import glob
import os
import os.path as osp

IMG_DIR = os.path.join(os.environ["nnUNet_raw"],
                       "Dataset107_PDAC_Detection", "imagesTr")
LBL_DIR = os.path.join(os.environ["nnUNet_raw"],
                       "Dataset107_PDAC_Detection", "labelsTr")

img_paths = sorted(glob.glob(osp.join(IMG_DIR, "*_0000.nii.gz")))
print(f"imagesTr/  : {IMG_DIR}")
print(f"found {len(img_paths)} images\n")

if len(img_paths) == 0:
    raise SystemExit("No images found. Check the path.")

# sample up to 5 cases evenly across the dataset
n = len(img_paths)
sample_idx = sorted(set(int(round(i * (n - 1) / 4)) for i in range(min(5, n))))
samples = [img_paths[i] for i in sample_idx]

phys_extents = []
for fp in samples:
    img = sitk.ReadImage(fp)
    size = img.GetSize()
    spacing = img.GetSpacing()
    phys = tuple(round(s * sp, 0) for s, sp in zip(size, spacing))
    phys_extents.append(phys)
    fsize_mb = osp.getsize(fp) / (1024 * 1024)

    # try to find the matching label
    base = osp.basename(fp).replace("_0000.nii.gz", "")
    lbl_fp = osp.join(LBL_DIR, base + ".nii.gz")
    if osp.isfile(lbl_fp):
        lbl = sitk.ReadImage(lbl_fp)
        import numpy as np
        lbl_np = sitk.GetArrayFromImage(lbl)
        labels_present = sorted(np.unique(lbl_np).tolist())
        lbl_info = f"labels={labels_present}"
    else:
        lbl_info = "(no matching label found)"

    print(f"{osp.basename(fp)}  [{fsize_mb:.1f} MB]")
    print(f"  voxels   : {size}")
    print(f"  spacing  : {tuple(round(s, 2) for s in spacing)} mm")
    print(f"  physical : {phys} mm")
    print(f"  {lbl_info}\n")

# verdict
def is_crop(p):
    return p[0] <= 350 and p[1] <= 250 and p[2] <= 200

crop_votes = sum(is_crop(p) for p in phys_extents)
print("=" * 60)
print(f"crop-like : {crop_votes}/{len(phys_extents)}")
if crop_votes == len(phys_extents):
    print("VERDICT   : looks like pre-cropped pancreas ROIs.")
    print("            ready for `nnUNetv2_plan_and_preprocess -d 107`.")
elif crop_votes == 0:
    print("VERDICT   : looks like full CTs. You need to crop to the")
    print("            pancreas ROI before preprocessing.")
else:
    print("VERDICT   : mixed. Investigate manually -- some cases look")
    print("            cropped and others do not.")
PYEOF
