#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=avmself
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:25:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/avmself_%j.out
#
# SELF-CONSISTENCY (codex Q3): is our dumped avm even consistent with our OWN
# closure inputs?  K_M = c_k * l_k * sqrt(en), l_k should be the bounded
# buoyancy length ~ sqrt(2*en)/N.  From the snapshot's own en (tke), N2 (via
# the model's bn2), and dumped K_M_diag, back out the IMPLIED length
# l_impl = K_M/(c_k*sqrt(en)) and compare to the buoyancy length
# l_buoy = sqrt(2*en)/N.
#   l_impl ~ l_buoy  -> our closure is self-consistent; the avm value IS what
#                       our own en+N2 produce, and the NEMO gap is NEMO doing
#                       something different (needs NEMO pointwise dump).
#   l_impl >> l_buoy -> our K assembly uses a LONGER length than the buoyancy
#                       length (a bug in OUR mixing-length/K path), independent
#                       of NEMO -- the real lever, fixable offline-verifiable.
# Reads existing snapshot, runs no model.  c_k = rn_ediff = 0.1.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
$PY - <<'PY'
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2
z = np.load("/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/"
            "nemolev_trp_kmean_d30/snapshot_day0028.npz")
T = np.asarray(z["T"], float); S = np.asarray(z["S"], float)
lat = np.asarray(z["lat_T"], float); lon = np.asarray(z["lon_T"], float) % 360
wet = np.asarray(z["land_mask"], float) > 0.5
tke = np.asarray(z["tke"], float)            # (ny,nx,nlev-1) interfaces
KM = np.asarray(z["K_M_diag"], float)        # (ny,nx,nlev-1)
gdept = np.abs(np.asarray(z["z_center_ref"], float))
zint = np.abs(np.asarray(z["z_interface_ref"], float))
# z_interface_ref (74) ARE the interior interface depths where N2/K live.
n2 = np.asarray(compute_buoyancy_frequency_nemo_bn2(
    jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept),
    jnp.asarray(zint), eos_form="teos10"))   # (ny,nx,nlev-1)
box = (np.abs(lat) <= 2) & (lon >= 200) & (lon <= 260) & wet
c_k = 0.1
print("depth   en_med    N2_med   avm_med  l_impl  l_buoy  ratio(impl/buoy)")
for target in [11, 30, 60, 90, 120]:
    k = int(np.argmin(np.abs(zint - target)))
    en = tke[..., k][box]; N2 = n2[..., k][box]; km = KM[..., k][box]
    ok = np.isfinite(en) & np.isfinite(N2) & (N2 > 0) & (en > 0) & (km > 0)
    en, N2, km = en[ok], N2[ok], km[ok]
    l_impl = km / (c_k * np.sqrt(en))
    l_buoy = np.sqrt(2 * en) / np.sqrt(N2)
    r = l_impl / l_buoy
    print(f"{zint[k]:5.1f}  {np.median(en):.2e} {np.median(N2):.2e} "
          f"{np.median(km):.2e} {np.median(l_impl):.3e} {np.median(l_buoy):.3e} "
          f"{np.median(r):.2f}")
print("\nratio~1 => avm is what our own en+N2 give (closure self-consistent);")
print("ratio>>1 => our K uses a longer length than the buoyancy length = OUR bug.")
PY
echo DONE
