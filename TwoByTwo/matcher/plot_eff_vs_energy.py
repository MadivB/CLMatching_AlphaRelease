import sys; sys.path.insert(0,'/pscratch/sd/y/yuxuan/2x2QLMatching/QLMatching2x2')
import numpy as np, h5py
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import pipeline_2x2 as pipe, light_model_2x2 as lm, truth_2x2 as truth
m=lm.load_light_model('sim',device='cuda')
h5=h5py.File('/global/cfs/cdirs/dune/users/yuxuan/2x2CLMatching/MiniRun6.4_1E19_RHC.flow.0000541.FLOW.hdf5','r')
tt=truth.TruthTables(h5)
ref=np.asarray(h5['charge/events/ref/light/events/ref'][()],np.int64)
evs=np.unique(ref[:,0])[:55]
E=[]; C=[]
for ev in evs:
    res=pipe.run_pipeline_for_event(h5,int(ev),light_model=m)   # unit variance default
    if res is None: continue
    ec=truth.evaluate_clusters(tt, ev_id=int(ev), hit_refs=res['hit_refs'], labels=res['labels'],
        hit_t0=res['hit_timestamps'], Eset=res['event'].Eset, hitTPCid=res['event'].hitTPCid, tolerance_ticks=10)
    for r in ec['rows']:
        if r['has_truth']:
            E.append(r['energy_mev']); C.append(1.0 if r['correct'] else 0.0)
E=np.array(E); C=np.array(C)
np.savez(sys.argv[1]+'.npz', energy=E, correct=C)

# 1-MeV bins 0-1 ... 9-10, then 10+
edges=list(range(0,11)); labels=[f"{i}-{i+1}" for i in range(10)]+[">10"]
effs=[]; ns=[]
for i in range(10):
    msk=(E>=i)&(E<i+1); ns.append(int(msk.sum()))
    effs.append(100*C[msk].mean() if msk.any() else 0.0)
msk=E>=10; ns.append(int(msk.sum())); effs.append(100*C[msk].mean() if msk.any() else 0.0)

fig,ax=plt.subplots(figsize=(11,5.5))
xpos=np.arange(len(labels))
bars=ax.bar(xpos, effs, color='#4C78A8', edgecolor='k', width=0.8)
# color the low-E (<5 MeV) bars differently to highlight the priority
for i,b in enumerate(bars):
    if i<5: b.set_color('#E45756')
for i,(e,n) in enumerate(zip(effs,ns)):
    ax.text(i, e+1.0, f"{e:.0f}%\nN={n}", ha='center', va='bottom', fontsize=9)
ax.axhline(np.average(effs[:5],weights=[ns[i] for i in range(5)]) if sum(ns[:5]) else 0,
           ls=':',color='#E45756',alpha=0.6)
ax.set_xticks(xpos); ax.set_xticklabels(labels)
ax.set_xlabel("cluster energy [MeV]"); ax.set_ylabel("matching efficiency [%]")
ax.set_ylim(0,105); ax.set_title("2x2 QL-matching efficiency vs cluster energy (per-cluster, |reco-truth|<=160ns, unit variance)")
ax.grid(axis='y',alpha=0.3,ls=':')
ax.text(0.99,0.04,f"{len(evs)} sim events, {len(E)} clusters\nred = low-energy (<5 MeV) priority",
        transform=ax.transAxes,ha='right',va='bottom',fontsize=9,
        bbox=dict(boxstyle='round',facecolor='white',alpha=0.85))
fig.tight_layout(); fig.savefig(sys.argv[1]+'.png',dpi=140,bbox_inches='tight')
print("bins:"); 
for l,e,n in zip(labels,effs,ns): print(f"  {l:>5} MeV: eff={e:5.1f}%  N={n}")
print("saved",sys.argv[1]+'.png')
