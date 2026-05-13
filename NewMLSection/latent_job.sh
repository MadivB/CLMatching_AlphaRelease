#!/bin/bash
#SBATCH -A dune
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH -t 720
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH -J latent_train
#SBATCH -o /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/M5p1ReleaseVersion/NewMLSection/runs/latent_run/train_%j.log
#SBATCH -e /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/M5p1ReleaseVersion/NewMLSection/runs/latent_run/train_%j.log

echo "========================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

cd /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/M5p1ReleaseVersion/NewMLSection

torchrun \
    --nproc_per_node=4 \
    latentMLTrain.py \
    --input-dir ./zarr_outputs_enhanced \
    --out-dir ./runs/latent_run \
    --batch-size 4 \
    --epochs 200 \
    --num-workers 16 \
    --latent-dim 32 \
    --no-amp \
    --resume

echo "========================================"
echo "Job finished: $(date)"
echo "========================================"
