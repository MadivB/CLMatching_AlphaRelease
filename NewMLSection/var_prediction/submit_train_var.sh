#!/bin/bash
#SBATCH -A dune
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH -t 16:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -J train_var_v2
#SBATCH -o /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/M5p1ReleaseVersion/NewMLSection/var_prediction/logs/train_var_v2_%j.log

cd /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/M5p1ReleaseVersion/NewMLSection/var_prediction

# Tell PyTorch to use all 4 GPUs on the Perlmutter node
mkdir -p ./logs
torchrun --nproc_per_node=4 train_var_ndfl.py --zarr-path ./var_multipeak/multi_train_perceiver_aligned_0001000_v2.zarr --out-dir ./runs/var_run_perceiver_aligned_0001000_v2 --epochs 500 --batch-size 32 --calib-lambda 5.0 --split-seed 1234
