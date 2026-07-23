"""Year-1 matched-window metrics, FULL 199x52 frame (no slicing)."""
import sys, numpy as np, netCDF4 as nc
from scipy.ndimage import uniform_filter
LEGO = sys.argv[1]
R1 = '/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_1Y'
mm = nc.Dataset('/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ/mesh_mask.nc')
def llz(a): return np.moveaxis(np.asarray(a).squeeze(),0,-1)
tm = llz(mm['tmask'][0])>0.5; wet = tm[:,:,0]
um = llz(mm['umask'][0])>0.5
e3 = np.asarray(mm['e3t_1d'][:]).squeeze()
e2u = np.asarray(mm['e2u'][0]).squeeze()[:,25]
gT = nc.Dataset(f'{R1}/DINO_1y_00010101_00011230_grid_T.nc')
gU = nc.Dataset(f'{R1}/DINO_1y_00010101_00011230_grid_U.nc')
e3s = np.asarray(gT['vovvle3t'][0]).squeeze()[0]
nSST = np.asarray(gT['votemper'][0]).squeeze()[0]/np.maximum(e3s,1e-6)
nSSH = np.asarray(gT['sossheig'][0]).squeeze()
nU = llz(gU['vozocrtx'][0])
nACC = np.median(np.einsum('jik,k,j->i', np.where(um,nU,0.0), e3, e2u)[2:-2])/1e6
def hp(a):
    sm = uniform_filter(np.nan_to_num(a), size=5, mode='nearest')
    return np.where(wet, a-sm, np.nan)
nhp = np.nanstd(hp(nSSH))
d = np.load(LEGO)
lU = d['u']; lu = lU[:,1:53,:].copy(); lu[:,47,:]=lU[:,48,:]*0+lU[:,48,:]  # faces 1..52 <-> NEMO u-cols
lACC = np.median(np.einsum('jik,k,j->i', np.where(um,lu,0.0), e3, e2u)[2:-2])/1e6
r = np.nanstd(hp(d['eta']))/nhp
sst = d['T'][:,:,0]
c = np.corrcoef(sst[wet], nSST[wet])[0,1]
print(f"FULL-FRAME year-1: ACC lego {lACC:.1f} vs NEMO {nACC:.1f} Sv | SSH small-scale ratio {r:.2f} | SST corr {c:.3f} bias {np.mean(sst[wet]-nSST[wet]):+.2f}")
