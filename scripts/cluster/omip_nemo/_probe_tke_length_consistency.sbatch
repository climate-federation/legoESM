#!/bin/bash
#SBATCH --job-name=tkelen
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/tkelen_%j.out
#
# IS OUR SMALL EQUATORIAL DIFFUSIVITY A CAUSE OR A CONSEQUENCE?
#
# The equatorial Pacific mixing deficit (tracer diffusivity ours/NEMO = 0.024
# at 0-20 m, 0.0003 at 20-60 m; our turbulent layer ends at 25 m where the
# oracle's reaches 73 m) has survived every named lever. Two mechanisms were
# proposed and BOTH REFUTED this session before any compute was spent:
#
#   * the TKE-diffusion floor (NEMO zdftke.F90:407-409, hardcoded 2e-5, no
#     namelist entry, and inert in our production branch) -- refuted on
#     SCALING: the TKE decay time is ~40 s, so sqrt(K*tau) with K = 2e-5 is
#     ~3 cm, not the ~50 m required. Reaching 73 m by diffusion would need
#     K ~ 18 m2/s.
#   * the buoyancy-sink discretisation (our default 'implicit_linearized' vs
#     the available NEMO-faithful 'nemo_explicit') -- refuted by BOTH
#     reviewers independently, and the sign of my proposed effect was
#     BACKWARDS: before flooring, implicit gives e/(1+x) and explicit gives
#     e(1-x), so the implicit form we run PRESERVES MORE energy, not less.
#     GLM added the general argument: at a discrete fixed point with identical
#     sources the two forms are algebraically identical, e*(1+dt*r) = e* +
#     dt*S, so they share equilibria and differ only in trajectory. A
#     quasi-steady deficit in AMPLITUDE cannot be set by the sink's form.
#
# So the question is no longer which term is mis-discretised, but whether our
# diffusivity is small BECAUSE the closure is faithfully reporting a small
# turbulent energy, or small on its own account. GLM's discriminator, which is
# a single pass over fields we already have:
#
#     l_implied = K_M / (c_k * sqrt(e))      the length the closure ACTUALLY used
#     l_b       = sqrt(2e) / N               the stability-limited length
#     l_wall    = kappa * z                  the wall length
#
# GLM wrote K_H in that first formula; it is K_M. Our closure sets
# K_M = c_k * l * sqrt(e) (tke.py:1959) and obtains K_H from it by the Prandtl
# ratio, so inverting K_H would fold the Prandtl factor into the length.
#
# PRE-REGISTERED, so the answer cannot be argued afterwards. Over 20-73 m,
# the band between where OUR turbulence stops and where the ORACLE'S reaches:
#   * l_implied within 2x of l_b -> the length is stability-limited and K_M is
#     a SLAVE to e. The closure is internally consistent, the diffusivity
#     deficit is a CONSEQUENCE, and the cause lives in e's SOURCES. Mixing
#     length and the sink forms are then both dead ends, consistent with the
#     eliminations already on record.
#   * l_implied << l_b/2 -> the length itself is anomalously short for the
#     energy present, and the length branch deserves re-opening despite being
#     on the eliminated list.
#   * l_implied >> 2*l_b -> K_M is too large for its own energy, i.e. the
#     closure is internally INCONSISTENT and the bug is in the assembly.
# GLM computed l_implied = 0.022 m against l_b = 0.036 m from my quoted
# numbers, i.e. the FIRST outcome, but those were remembered single values;
# this measures the profile.
#
# CONVENTIONS PINNED FROM THE RUN'S RESOLVED CONFIG, not from class defaults:
# n2_mode = nemo_bn2, n2_eos_form = teos10, c_k = 0.1, tke_background = 1e-10.
# The eos_form is passed EXPLICITLY because this helper defaults to 'seos'
# while ORCA1 runs TEOS-10, and that exact omission was a previously recorded
# defect on this thread.
#
# CAVEAT STATED UP FRONT: N2 is evaluated on the REFERENCE depths
# (z_center_ref / z_interface_ref), not the live z-star/partial-cell
# thicknesses. That is a small distortion for a ratio of lengths and is the
# reason this probe reports a ratio rather than an absolute diffusivity.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

$PY - <<'PY'
import numpy as np
import jax.numpy as jnp
from legoesm import constants
from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2

C_K = 0.1                 # resolved TKEConfig.c_k
E_BG = 1.0e-10            # resolved TKEConfig.tke_background
KARMAN = 0.4

S_PATH = ("/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/"
          "trp_orca1faith_d30/snapshot_final.npz")
d = np.load(S_PATH, allow_pickle=True)
T = np.asarray(d["T"], dtype=np.float64)
S = np.asarray(d["S"], dtype=np.float64)
e = np.asarray(d["tke"], dtype=np.float64)
KM = np.asarray(d["K_M_diag"], dtype=np.float64)
lat = np.asarray(d["lat_T"], dtype=np.float64)
lon = np.asarray(d["lon_T"], dtype=np.float64)
zc = np.abs(np.asarray(d["z_center_ref"], dtype=np.float64))
zi = np.abs(np.asarray(d["z_interface_ref"], dtype=np.float64))
print("T", T.shape, " tke", e.shape, " K_M", KM.shape,
      " z_center", zc.shape, " z_interface", zi.shape)

# N2 on INTERIOR interfaces: (..., nlev-1). MEASURED, not assumed: T carries
# 75 levels and z_interface_ref carries 74 entries, i.e. it is ALREADY the
# interior w-points, so it is passed whole rather than sliced.
# e3w_source: the 'mesh_reference' default demands the raw-mesh e3w this
# snapshot does not store, so the documented 'depth_difference' construction
# is selected explicitly. That is a metric approximation and it is the second
# reason this probe reports a RATIO of two lengths, in which the metric
# largely cancels, rather than an absolute diffusivity.
assert zi.size == zc.size - 1, (zi.size, zc.size)
N2 = np.asarray(compute_buoyancy_frequency_nemo_bn2(
    jnp.asarray(T), jnp.asarray(S), jnp.asarray(zc), jnp.asarray(zi),
    eos_form="teos10", e3w_source="depth_difference"), dtype=np.float64)
print("N2", N2.shape, " finite:", bool(np.all(np.isfinite(N2))))

# Put e and K_M on the SAME interior interfaces as N2 before any ratio is
# formed. Naming both staggerings out loud is the rule that keeps being paid
# for on this thread.
def to_interior(a):
    if a.shape[-1] == N2.shape[-1]:
        return a
    if a.shape[-1] == N2.shape[-1] + 1:
        return 0.5 * (a[..., :-1] + a[..., 1:])
    if a.shape[-1] == N2.shape[-1] + 2:
        return a[..., 1:-1]
    raise SystemExit(f"cannot align {a.shape} onto N2 {N2.shape}")

e_i, KM_i = to_interior(e), to_interior(KM)
zf = zi                                   # the interior w-point depths [m]
assert zf.size == N2.shape[-1], (zf.size, N2.shape)

lon360 = np.mod(lon, 360.0)
box = (lat >= -2.0) & (lat <= 2.0) & (lon360 >= 190.0) & (lon360 <= 260.0)
print("equatorial box cells:", int(box.sum()))
if int(box.sum()) == 0:
    raise SystemExit("box selected no cells -- check lat/lon conventions")

ec, Kc, Nc = e_i[box], KM_i[box], N2[box]

# Only where there IS turbulence to speak of: e above its background, positive
# stratification, positive diffusivity. A ratio built on floored cells would
# measure the floor, not the closure.
live = (ec > 10.0 * E_BG) & (Nc > 0.0) & (Kc > 0.0) & np.isfinite(Nc)

l_impl = np.where(live, Kc / (C_K * np.sqrt(np.maximum(ec, E_BG))), np.nan)
l_b = np.where(live, np.sqrt(2.0 * np.maximum(ec, E_BG)) / np.sqrt(
    np.maximum(Nc, 1e-20)), np.nan)
l_wall = KARMAN * zf[None, :]

print("\nMIXING LENGTH THE CLOSURE ACTUALLY USED vs THE STABILITY LIMIT")
print("  depth[m]     n    l_implied[m]   l_b[m]    l_wall[m]  l_impl/l_b")
for lo, hi in ((0, 20), (20, 40), (40, 73), (73, 120), (120, 300)):
    m = (zf >= lo) & (zf < hi)
    if not np.any(m):
        continue
    li, lb = l_impl[:, m], l_b[:, m]
    lw = np.broadcast_to(l_wall, l_impl.shape)[:, m]
    ok = np.isfinite(li) & np.isfinite(lb)
    if ok.sum() == 0:
        print("  %4d-%-4d    0    (no live cells -- turbulence at background)"
              % (lo, hi))
        continue
    r = np.median(li[ok] / lb[ok])
    print("  %4d-%-4d %6d   %.4e   %.4e  %.4e   %.3f"
          % (lo, hi, int(ok.sum()), np.median(li[ok]), np.median(lb[ok]),
             np.median(lw[ok]), r))

m = (zf >= 20.0) & (zf < 73.0)
li, lb = l_impl[:, m], l_b[:, m]
ok = np.isfinite(li) & np.isfinite(lb)
frac_live = float(ok.sum()) / float(li.size) if li.size else 0.0
ratio = float(np.median(li[ok] / lb[ok])) if ok.sum() else float("nan")
print("\nPRE-REGISTERED READOUT (20-73 m, the band the oracle fills and we do not)")
print("  live cells: %.1f%% of %d" % (100.0 * frac_live, li.size))
print("  median l_implied / l_b = %.3f" % ratio)
print("  WITHIN 0.5-2.0 -> length is stability-limited, K_M is a SLAVE to e,")
print("     the diffusivity deficit is a CONSEQUENCE and the cause is in e's")
print("     SOURCES. BELOW 0.5 -> the length is anomalously short. ABOVE 2.0")
print("     -> the closure is internally inconsistent (assembly bug).")
if frac_live < 0.05:
    print("  NOTE: almost nothing is live here -- our turbulence is AT")
    print("     background through this band, which is itself the finding and")
    print("     makes the ratio above unrepresentative.")
PY
echo DONE
