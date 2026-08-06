#!/bin/bash
#SBATCH -A dune
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:25:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -J smoke_p1nb
#SBATCH -o /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/smoke_phase1_%j.log
set -uo pipefail
PY=/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python
$PY -u /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing/smoke_phase1_nb.py
