# Design: `freshwater_closure = "real_freshwater"` (ocean)

Status: DESIGN, not implemented. Written 2026-08-05 while the A/B (9310972)
and exact-salt (9311823) runs are still in flight.

## The defect this fixes

Measured, days 30->90, Arctic >=66N column salt, one reviewed estimator on both
sides, identical 5,164-cell support, instantaneous endpoints:

| quantity | value |
|---|---|
| ours (B') | **+16.829** psu.m |
| NEMO (N') | **+6.720** psu.m |
| gap | **+10.109** psu.m |

Confirmed in code (codex 9320563): the latlon C-grid tracer equation already
uses a moving z-star thickness,

    h_old = thickness(eta_mid);  h_new = thickness(eta_new)
    hS_new = h_old*S_mid - dt*flux_div;   S_new = hS_new / h_new

so with no transport the stretching alone gives `S_new = h_old*S/h_new`, which
**preserves `h*S` while the column expands or contracts**. Freshwater dilution
is therefore ALREADY handled conservatively. On top of that the model applies a
virtual salt flux `-S_ref*F_fw/(rho_0*dz_0)`, which adds a *separate*
salt-content change. NEMO, running variable volume (`ocean.output`:
`dom_qco_init : Variable volume activated`), has no such term.

Both grids are affected: `ocean_model_latlon_cgrid.py:83/3158/4212-4213` and
`ocean_model_mpas.py:744` + the VSF in `ocean_pe_mpas.py`. Both validators
allow only `{none, virtual_salt_flux}`; `real_freshwater` is named ONLY in the
MPAS error string as unavailable (`ocean_model_mpas.py:185-190`).

Corroboration (all measured this session):
- excess is ice-colocated, latitude-controlled: at fixed 70-74N, open water
  **-1.48** vs ice-covered **+22.49** psu.m.
- global budget: Arctic **+2.244e14**, non-Arctic **-2.007e14** psu.m3 =>
  **89% redistribution** plus ~11% net drift (+0.0653 psu.m global mean).
- out-of-sample: d30-90 is NH winter AND SH summer. Arctic ice grows -> we ADD
  salt (+10.1); Southern Ocean ice melts -> we REMOVE salt (**-4.367**).
  Opposite hemispheres, opposite seasons, opposite signs.
- surface signature: global mean SSS rises **+0.10 psu in 60 days**, consistent
  with the +0.065 psu.m column drift deposited into a ~1 m top cell.

## Proposed semantics (REVISED after codex design review, job 9321685)

**My first draft proposed adding `sfx = (S_o - S_i)*m_ice`. That was WRONG in
three independent ways and is REMOVED.** Recorded because the errors are
instructive:
1. **Sign inverted.** For freezing (m>0) the true direct salt flux is NEGATIVE,
   `F_salt = -1e-3 * S_i * m`, because low-salinity ice STORES salt; melting
   returns it (positive). `(S_o - S_i)*m` is the result AFTER the thickness
   change, not a source to add on top. Confirmed in our own
   `ice/brine.py:247-265` and NEMO `icethd_dh.F90:349-354, 406-413`.
2. **1000x unit error.** `salt_flux` is kg(salt)/m2/s and converts as
   `dS/dt = salt_flux*1e3/(rho*h)` (`freshwater.py:649-679`). PSU*kg/m2/s would
   be silently 1000x wrong.
3. **Double count.** A genuine salt channel ALREADY exists and is already
   applied with the actual top-cell thickness in both cores
   (`ocean_pe_latlon_cgrid.py:3674-3686`, `ocean_pe_mpas.py:1055-1088`); the
   coupler already maps `ice_resp.salt_flux` separately from `ice_fw`
   (`coupler/ocean_forcing.py:77-98`).

### The actual design

`freshwater_closure="real_freshwater"`:
- **eta/volume channel: UNCHANGED** (already correct, already NEMO-like).
- **skip ONLY the freshwater-derived virtual-salt block.**
- **retain the existing `surface_forcing.salt_flux` pathway untouched.**
- **normalization applies to the eta budget only; NEVER to the genuine
  salt_flux.**

### Answers from the review (cite before relying on these)

1. `ice_fw` is the WATER leg of an already two-leg exchange, is **positive on
   MELT** (not freeze), and in the v2 path also carries snow, pond drainage,
   rain runoff, lead freezing, flooding and ablation
   (`ice/sea_ice.py:2205-2262`). **Do not derive any salt flux from it.**
2. `S_i` is conditional: prognostic with `brine.enabled=True`
   (`ice/state.py:33-56`); in the default slab path the ice salt flux is
   explicitly ZERO (`ice/sea_ice.py:627-630`), i.e. operationally `S_i=0`.
   **LANDMINE:** an inactive `S_ice` carrier defaults to ~4 PSU
   (`ice/state.py:69-95`) — reading it would silently invent salt flux while
   brine is disabled.
3. NEMO `nn_sssr=2` (this deck) adds restoring to **emp**, not `sfx`
   (`sbcssr.F90:127-140`) => restoring must be **volume-only**. BUT our
   restoring flux is derived FOR the virtual-salt formulation using `S_target`
   and a configured `z1` (`ocean/forcing/sss_restoring.py:326-357`); reused
   volume-only its realized strength is WRONG unless recomputed with the
   current surface salinity and the ACTUAL top-cell thickness. Also decide
   whether to reproduce NEMO's associated heat correction.
4. Normalization: eta only. **Cross-grid inconsistency found:** MPAS normalizes
   eta (`ocean_model_mpas.py:741-766`) while lat-lon sends RAW freshwater to eta
   (`ocean_model_latlon_cgrid.py:3205-3210`). Pick ONE policy explicitly:
   normalize full `P-E+R+ice+restoring` (fixed liquid-ocean volume) OR exclude
   `ice_fw` (NEMO-like ice+ocean volume, `sbcfwb.F90:233-238`).
5. Conservation invariant = salt MASS, not mean salinity:
   `M_s = rho0*1e-3*sum_ik A_i h_ik S_ik`, `dM_s = dt*sum_i A_i F_salt,i`.
   For a closed P/E/R-only test `F_salt=0`, so `sum A h S` is invariant to
   roundoff. **NON-VACUITY WARNING: with normalized scalar-S_ref forcing this
   test can PASS VACUOUSLY.** It must use a single forced wet column,
   `normalize_freshwater=False`, zero real salt flux, and no
   transport/mixing/fixers. It then fails on today's code, which changes salt
   mass by ~`-1e-3*dt*S_ref*sum(A*F)`.

### Scale note

NEMO's Arctic ice salt STORAGE moved only -0.325 psu.m equivalent over the
window vs our +8.03 surface term (~25x). If the small genuine ice exchange is
wanted in the default slab run, that is a SEA-ICE decision (enable the
brine-aware budget / add an ice salt inventory), **not** an ocean-closure
correction.

## Required before merge (repo rules)

- direct unit test for every touched module; dispatch must `raise` on unknown
  closure (both grids) and be locked in `test_dispatch_hardening.py`.
- a synthetic-violation test proving the conservation gate CAN fail.
- codex adversarial review; sign-convention walk on every term.
- a freshwater/VSF diagnostic in the run output — the present runs log no
  freshwater field at all (`diag_timeseries.csv` has 13 columns, only
  `mean_sss`), which is why the ice-VSF vs normalization split could not be
  closed offline this session.
