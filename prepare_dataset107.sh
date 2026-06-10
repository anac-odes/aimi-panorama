#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
##SBATCH --qos=csedu-large
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=prepare_dataset107.out
#SBATCH --time=12:00:00

source /vol/csedu-nobackup/course/IMC037_aimi/group03/analeopold/miniconda3/etc/profile.d/conda.sh
conda activate panorama

REPO=/vol/csedu-nobackup/course/IMC037_aimi/group03/analeopold/aimi-panorama

export nnUNet_raw="$REPO/workspace/nnUNet_raw"
export nnUNet_preprocessed="$REPO/workspace/nnUNet_preprocessed"
export nnUNet_results="$REPO/workspace/nnUNet_results"

python $REPO/prepare_dataset107.py
