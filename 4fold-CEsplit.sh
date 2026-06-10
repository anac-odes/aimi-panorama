#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
#SBATCH --qos=csedu-large
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
##SBATCH --cpus-per-task=12
#SBATCH --output=celesion.out
#SBATCH --time=48:00:00
#SBATCH --mail-user=analeopold
#SBATCH --mail-type=BEGIN,END,FAIL

source /vol/csedu-nobackup/course/IMC037_aimi/group03/analeopold/miniconda3/etc/profile.d/conda.sh
conda activate panorama

REPO=/vol/csedu-nobackup/course/IMC037_aimi/group03/analeopold/aimi-panorama

export nnUNet_raw="$REPO/workspace/nnUNet_raw"
export nnUNet_preprocessed="$REPO/workspace/nnUNet_preprocessed"
export nnUNet_results="$REPO/workspace/nnUNet_results"

nnUNetv2_train 107 3d_fullres 4 \
    -tr nnUNetTrainerCELossLesionSplit \
    -p nnUNetPlans_v3 \
    --npz \
    --c