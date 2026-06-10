#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
##SBATCH --qos=csedu-large
##SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=12
#SBATCH --output=processv3.out
#SBATCH --time=12:00:00


source /vol/csedu-nobackup/course/IMC037_aimi/group03/analeopold/miniconda3/etc/profile.d/conda.sh
conda activate panorama

REPO=/vol/csedu-nobackup/course/IMC037_aimi/group03/analeopold/aimi-panorama

export nnUNet_raw="$REPO/workspace/nnUNet_raw"
export nnUNet_preprocessed="$REPO/workspace/nnUNet_preprocessed"
export nnUNet_results="$REPO/workspace/nnUNet_results"


nnUNetv2_preprocess -d 107 -plans_name nnUNetPlans_v3 -c 3d_fullres -np 12