#!/bin/bash -l

# The following are parameters for sbatch, adjust accordingly:

#SBATCH --qos=normal                        # Either 'normal' (max 1 day) or 'long' (max 7 days)
#SBATCH --job-name=decodeSteelDefects        # Job name, can be anything
#SBATCH --cpus-per-task=4                   # CPU cores needed for job
#SBATCH --mem=128G                           # Total RAM (up to 500G)
#SBATCH --gres=gpu:1                        # Number of GPUs to use (if no GPU required, use 0, up to 8)
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
python src/data_preprocessing.py \
    --image-dir /trinity/home/skadimisetty/data/sathvik/Steel-Defect-Detection/src/train_images/ \
    --mask-dir /trinity/home/skadimisetty/data/sathvik/Steel-Defect-Detection/src/mask_output/ \
    --output-dir /trinity/home/skadimisetty/data/sathvik/Steel-Defect-Detection/src/processed_data/ \
    --img-size 512 \
    --val-split 0.2 \
    --augment

# seff job_id