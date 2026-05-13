import h5py
import numpy as np
import yaml
from lut import LUT

f = '/pscratch/sd/d/dunepro/yuxuan/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.lowintensity.sanddrift/FLOW/0000000/MiniProdN5p1_NDComplex_FHC.flow.full.lowintensity.sanddrift.0000001.FLOW.hdf5'
h5 = h5py.File(f,'r')
sipm_rel_pos = LUT.from_array(h5["geometry_info/sipm_rel_pos"].attrs["meta"],h5["geometry_info/sipm_rel_pos/data"])
new_lookup_table = {}
for adc in range(140):
    for channel in range(64):
        key = (adc, channel)
        try:
            tpc_side_y = sipm_rel_pos[key]
            TPC, side, y = tpc_side_y[0]
            new_lookup_table[(TPC, side, y)] = [adc, channel]
        except:
            pass

print("Number of entries in LUT:", len(new_lookup_table))
sample_tpc = 0
ys_side0 = [y for (tpc, side, y) in new_lookup_table if tpc == sample_tpc and side == 0]
ys_side1 = [y for (tpc, side, y) in new_lookup_table if tpc == sample_tpc and side == 1]
print("TPC 0, side 0 y range:", min(ys_side0), max(ys_side0), "count:", len(ys_side0))
print("TPC 0, side 1 y range:", min(ys_side1), max(ys_side1), "count:", len(ys_side1))

