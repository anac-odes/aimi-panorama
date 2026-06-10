# Improvement of PanDx
Group 3

## About This Project
This project investigates the effect of inverse alpha and different loss
functions on PDAC lesion detection, building on the winning solution of the
PANORAMA challenge.

Our contributions:
- Data preprocessing pipeline (`fix_dataset107_spacing.py`, `prepare_dataset107.py`)
- Distribution-Aware Stratified Ensembling splits (`make_dase_splits.py`)
- Systematic comparison of 6 loss functions: `nnUNetTrainerCELossLesionSplit`,
  `nnUNetTrainer_WFocalLoss`, `nnUNetTrainerTopK10TrainLossS01`,
  `nnUNetTrainerDiceLoss`, `nnUNetTrainerDiceCELoss_noSmooth`, and
  `nnUNetTrainerWCELoss` 
- Added a script for voxel level evaluation which can be run by the following command:
  `python evaluate.py --pred_dir ./workspace/workspace/test_example/output/pdac-detection-map --label_dir ./labels --threshold 0.5`
  Where the locations of the predicted directory and the label directory are replaced by the correct locations


The data preprocessing and DASE scripts can be found in `custom-files` folder.

Winning model: [PanDx](https://github.com/han-liu/PanDx) by Han Liu et al.
Challenge: https://panorama.grand-challenge.org/

## Installation
### Requirements
```bash
cuda-11.1, cudnn/9.0.0-cuda-12
```

### Create a virtual environment
```bash
conda create -n pdac python=3.12 -y
conda activate pdac
```

### Install dependencies
```bash
git clone https://github.com/han-liu/PDAC_Detection.git
cd PDAC_Detection
pip install -r requirements.txt

cd packages/nnunetv2
pip install -e .

cd ../report-guided-annotation
pip install -e .
```

## Dataset Preparation

### Step 1: Fix metadata inconsistencies
```bash
python fix_dataset107_spacing.py
# Repairs spacing/origin mismatches in Dataset107 so the nnU-Net
# integrity check passes. Safe to re-run (idempotent).
```

### Step 2: Create DASE splits
```bash
python make_dase_splits.py
# Builds 5-fold cross-validation splits stratified by lesion size,
# following the Distribution-Aware Stratified Ensembling strategy.
```

### Step 3: Prepare dataset
```bash
python prepare_dataset107.py
# Crops CTs to the pancreas ROI using the Dataset103 localizer.
```

### Step 4: Verify dataset and run planning/preprocessing
```bash
nnUNetv2_plan_and_preprocess -d 107 --verify_dataset_integrity
# Verifies dataset integrity, then generates the experiment plan and
# preprocessed data required for training.
```

## Loss functions

Each loss is implemented as a custom nnU-Net trainer. The trainer classes are
defined in:
`packages/nnunetv2/nnunetv2/training/nnUNetTrainer/variants/loss/`

The underlying loss implementations they call are in:
`packages/nnunetv2/nnunetv2/training/loss/`




## Training

### Set up environment variables for nnU-Net
```bash
export nnUNet_raw="./workspace/nnUNet_raw"
export nnUNet_preprocessed="./workspace/nnUNet_preprocessed"
export nnUNet_results="./workspace/nnUNet_results"
```

### Train
To train with a specific loss function:
```bash
nnUNetv2_train 107 3d_fullres 0 \
    -tr nnUNetTrainer_WFocalLoss \
    -p nnUNetPlans_v3 \
    --npz
```
Replace `nnUNetTrainer_WFocalLoss` with any trainer from the
[Loss functions] table:
`nnUNetTrainerCELossLesionSplit`,
`nnUNetTrainerTopK10TrainLossS01`,
`nnUNetTrainerDiceLoss`,
`nnUNetTrainerDiceCELoss_noSmooth`,
`nnUNetTrainerWCELoss`.

> **Note:** `nnUNetPlans_v3` is the pre-computed plan from the original PanDx
> winning solution and must be copied from their released model bundle before
> training. See the [PanDx repository](https://github.com/han-liu/PanDx) for
> instructions on downloading the pretrained bundle.



## Inference

### Set up environment variables for nnU-Net
```bash
export nnUNet_raw="./workspace/nnUNet_raw"
export nnUNet_preprocessed="./workspace/nnUNet_preprocessed"
export nnUNet_results="./workspace/nnUNet_results"
```


### For a quick test using the example testing images, run:
```bash
python main.py -i ./workspace/test_example/input -o ./workspace/test_example/output
```
This uses the default trainer (`nnUNetTrainerCELossLesionSplit`).

### What are the outputs?
- A PDAC detection map (ranging from 0–1) where each predicted lesion is
  assigned a confidence score.
- A patient-level likelihood score (computed as the **maximum** value of the
  detection map).

The PDAC detection maps are saved under `${OUTPUT_DIR}/pdac-detection-map`:
```
├── ${OUTPUT_DIR}/
    ├── pdac-likelihood.json
    └── pdac-detection-map/
        ├── filename1.nii.gz
        ├── filename2.nii.gz
        └── ...
```

The `pdac-likelihood.json` contains the likelihood scores for each patient:
```json
{
    "filename1": 0.9965946078300476,
    "filename2": 0.9977765679359436
}
```
