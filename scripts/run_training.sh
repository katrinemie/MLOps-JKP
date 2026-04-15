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
VENV="/ceph/home/student.aau.dk/gd65nz/envs/mlops"

mkdir -p "$WORKDIR/logs" "$WORKDIR/models"

# Opret venv hvis det ikke findes
if [ ! -d "$VENV" ]; then
    singularity exec --nv "$CONTAINER" bash -c "
        python -m venv --system-site-packages $VENV
        source $VENV/bin/activate
        pip install --quiet carbontracker mlflow[s3] pyyaml scikit-learn tqdm
    "
fi

singularity exec --nv \
    --env PYTHONNOUSERSITE=1 \
    "$CONTAINER" bash -c "
    source $VENV/bin/activate
    cd $WORKDIR/src
    python train.py --config ../configs/config.yaml
"
