#!/bin/bash
#SBATCH -A dune
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 240
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -J compile_samples
#SBATCH -o /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/M5p1ReleaseVersion/NewMLSection/compile_extra_samples.log

cd /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/M5p1ReleaseVersion/NewMLSection
python -u compile_samples_to_zarr.py \
    --in-dir /pscratch/sd/d/dunepro/yuxuan/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.lowintensity.sanddrift/FLOW/0001000 \
    --out-dir ./zarr_outputs_enhanced
