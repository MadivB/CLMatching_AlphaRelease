#!/bin/bash
#SBATCH -A dune
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH -t 1440
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH -J ndfull_train
#SBATCH -o /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/M5p1ReleaseVersion/NewMLSection/runs/ndfull_run_distributed/train_%j.log
#SBATCH -e /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/M5p1ReleaseVersion/NewMLSection/runs/ndfull_run_distributed/train_%j.log

echo "========================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

cd /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/M5p1ReleaseVersion/NewMLSection

torchrun \
    --nproc_per_node=4 \
    train_ndfull.py \
    --input-dir ./zarr_outputs_enhanced \
    --out-dir ./runs/ndfull_run_distributed \
    --batch-size 4 \
    --epochs 120 \
    --num-workers 4 \
    --no-amp \
    --resume

echo "========================================"
echo "Job finished: $(date)"
echo "========================================"
