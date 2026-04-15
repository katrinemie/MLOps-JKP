#!/bin/bash
#SBATCH --job-name=module7
#SBATCH --output=logs/module7_%j.log
#SBATCH --error=logs/module7_%j_err.log
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

CONTAINER="/ceph/container/pytorch/pytorch_26.02.sif"
WORKDIR="/ceph/home/student.aau.dk/gd65nz/MLOPS_DAKI4/MLOps-JKP"

mkdir -p "$WORKDIR/logs" "$WORKDIR/results"

singularity exec --nv \
    --env PYTHONNOUSERSITE=1 \
    "$CONTAINER" bash -c "
    cd $WORKDIR/src

    echo '=============================='
    echo 'Exercise 1: Continual Learning'
    echo '=============================='
    python continual_learning.py

    echo ''
    echo '========================='
    echo 'Exercise 2: Unlearning'
    echo '========================='
    python unlearning.py
"
