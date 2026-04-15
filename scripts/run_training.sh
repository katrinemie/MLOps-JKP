#!/bin/bash
#SBATCH --job-name=catdog_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j_err.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --time=01:00:00

CONTAINER="/ceph/container/pytorch/pytorch_26.02.sif"
WORKDIR="/ceph/home/student.aau.dk/gd65nz/MLOPS_DAKI4/MLOps-JKP"

mkdir -p "$WORKDIR/logs" "$WORKDIR/models"

singularity exec --nv \
    --env PYTHONNOUSERSITE=1 \
    "$CONTAINER" bash -c "
    cd $WORKDIR/src
    pip install --quiet carbontracker mlflow pyyaml scikit-learn tqdm 2>/dev/null
    python train.py --config ../configs/config.yaml
"
