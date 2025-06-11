#!/bin/bash -l

# The following are parameters for sbatch, adjust accordingly:

#SBATCH --qos=normal                        # Either 'normal' (max 1 day) or 'long' (max 7 days)
#SBATCH --job-name=trainSD        # Job name, can be anything
#SBATCH --cpus-per-task=16                   # CPU cores needed for job
#SBATCH --mem=64G                           # Total RAM (up to 500G)
#SBATCH --gres=gpu:4                        # Number of GPUs to use (if no GPU required, use 0, up to 8)
#SBATCH --output=logs_cluster/job.%j.out    # Stdout & Stderr in a single file (%j=jobId), make sure to manually create ./logs
#SBATCH --time=0-23:59:59                   # Time limit for job, in days-hrs:min:sec
#SBATCH --ntasks-per-node=1                 # Keep this at 1
# # SBATCH --array=0-12                     # Job array index values%Simultaneous running tasks


# activate conda environment with libraries needed for job
conda activate defects

export PYTHONPATH="${PYTHONPATH}:/trinity/home/skadimisetty/data/sathvik/Steel-Defect-Detection"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/trinity/home/skadimisetty/data/sathvik/miniconda3/envs/defects/lib
export CXX=g++

# run script
python src/train.py \
    --data-dir /trinity/home/skadimisetty/data/sathvik/Steel-Defect-Detection/src/processed_data/ \
    --epochs 25 \
    --model-path /trinity/home/skadimisetty/data/sathvik/Steel-Defect-Detection/src/unet_model.pth \
    --multi-gpu

# seff job_id