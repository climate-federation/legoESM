# DINO setup audit — legoESM vs NEMO `usrdef_*` (code-read, term-by-term)

> **2026-09-08 — ROWS 1, 9 AND 10 ARE SUPERSEDED.** Everything below was
> written on the halo-strip-era **48×195** frame. NEMO's DINO domain is
> **52×199** (`usrdef_nam.F90:141-155`; the haloless `mesh_mask.nc` is 52×199,
> and `ocean.output`'s `jpiglo=56`/`jpjglo=203` are those plus `2*nn_hls`), and
> `run_dino.py --nemo-faithful-grid` now builds it by TRANSCRIBING NEMO's own
> construction (`packages/ocean/legoesm/ocean/fidelity/nemo_dino_mesh.py`)
> instead of approximating it. What that changes here:
>
> | row | was | now |
> |---|---|---|
> | 1 Horizontal grid | "reproduced to 3e-6°" | **bit-exact** — 0 cells unequal on `glamt/u/v/f`, `gphit/u/v/f`, `e1*`, `e2*` |
> | 9 Bowl bathymetry extent | corr 0.92, "precise cause not isolated" | **RESOLVED** — the cause WAS `zgr_get_boundaries`. On the true frame it returns `pminlam=0`, `pmaxlam=51`, `pminphi=-69.678802541192084`, `pmaxphi=70.023256525040722` (printed by `RUN_TRAJ/ocean.output:382-385`), and `distPhi = cos(rad·pmaxphi)·distLam`. With those, `tmask` matches with **0 cells unequal** |
> | 10 Seam wall placement | MISMATCH, 9201 wet vs "NEMO's 9360" | **RESOLVED for this path** — it uses NEMO's own mask: land ring on columns 0 and 51 outside the ACC band (`usrdef_zgr.F90:150-163`) plus the closed first/last row (`domzgr.F90:308-314`). **9920** surface wet, 342134 3-D wet, `umask`/`vmask` exact. (The 9360 figure was itself measured on the wrong frame.) legoESM's own analytic seam wall, described below, still applies to the NON-faithful DINO recipes. |
>
> The `glamu ∈ [2.00, 49.00]` / `gphiv ∈ [−68.97, 69.33]` extents quoted in row
> 9's discussion are likewise halo-strip-era; the true ones are `glamu ∈ [0, 51]`
> and `gphiv ∈ [−69.6788, 70.0233]`.
>
> Gate: `scripts/validate/ocean_fidelity/dino_1226/nemo_dino_mesh_gate.py` —
> every one of the 45 variables in `mesh_mask.nc` is VERIFIED or WAIVED with a
> reason, and it exits non-zero on any inequality (`--plant` proves it can).
>
> **The one remaining inequality**, and it is the oracle's: `ff_t`/`ff_f` differ
> by ≤3 ulp (2.7e-20 absolute). `-O3` vectorised `usr_def_hgr.F90:152-153`'s two
> whole-array Coriolis assignments into glibc's 2-wide vector sine — 14
> `_ZGVbN2v_sin@plt` calls in `usr_def_hgr`, against scalar `asin`/`cos`/`tanh`
> for the loop body — which is accurate to 4 ulp rather than correctly rounded.
> legoESM never reads `ff_t` (the bridge builds Coriolis from `gphit` and reads
> `ff_t` only to check itself), so this is bounded and waived rather than
> chased. To close it on the oracle's side, run
> `scripts/experiment/dino/nemo_scalar_math_rebuild.sh` (it pre-registers what
> confirms and what refutes the diagnosis) and re-run the gate against the
> regenerated mesh.

A line-by-line verification of legoESM's DINO *experiment setup* (domain,
bathymetry, vertical coordinate, masks) against NEMO's `usrdef_*.F90` source on
our NEMO 5.0.2 build (`cfgs/DINO/MY_SRC`). This complements the *operator*
certification in `dino_tendency_certificate.md`: that verified the numerics
(tendencies ≥0.99); this verifies the setup the operators run on.

Motivation: the operator certificate ran on the fidelity **bridge** (legoESM
state built from NEMO's own `mesh_mask`), which uses NEMO's geometry by
construction — so setup-layer differences were structurally invisible to it.
Two were later surfaced by visual inspection of solution plots. This audit reads
NEMO's source directly rather than inferring from processed arrays.

## Scorecard

| # | Component | NEMO source | Verdict | Detail |
|---|-----------|-------------|---------|--------|
| 1 | Horizontal grid (Mercator) | `usrdef_hgr.F90` | **MATCH** | `sin φ = tanh(Δλ·k)`, reproduced to 3e-6° |
| 2 | Vertical coordinate | `usrdef_zgr` + `e3t_1d` | **MATCH** | 36 levels; `e3t_1d` sums to 4506.4 m both; bridge uses NEMO's `e3t_1d` |
| 3 | Initial T/S | `usrdef_istate.F90` CASE(4) | **MATCH** | +0.08 °C (see tendency certificate) |
| 4 | Surface forcing | `usrdef_sbc.F90` CASE(4) | **MATCH** | all 6 fields byte-exact (see certificate) |
| 5 | Bowl formula `exp_bathy` | `usrdef_zgr.F90:233-288` | **MATCH** | identical: exp taper + **quintic** `smooth_step` (6t⁵−15t⁴+10t³) blend, `H_deep=4000`/`H_shallow=2000`, `dist_lam=3`, `taper=cha_width/2=10` |
| 6 | Drake sill `gauss_ring` | `usrdef_zgr.F90:367-374` | **MATCH** | identical ring formula + params: depth 2500, width 4, lat −55, radius 10, anchored at west wall |
| 7 | Mid-Atlantic ridge | `usrdef_zgr.F90:321` | **MATCH** | both OFF (`ln_mid_ridge=.false.`) |
| 8 | Channel band | `rn_cha_min/max` | **MATCH** | both [−65, −45], width 20° |
| 9 | Bowl *numeric* result | `tmask`/`e3t_0` | **MINOR** | corr 0.92, −56 m mean / 183 m rms — formula identical; residual = boundary-*extent* convention (below) |
| 10 | Seam wall placement | `usrdef_zgr.F90:152-170` | **MISMATCH** | NEMO walls the periodic *seam*; legoESM masks an interior column (below) |

Verdict: the DINO setup **matches NEMO term-by-term** — every formula and
parameter is the same — with one structural difference (seam wall) and one ~2%
bathymetry-extent residual, both characterized below.

## #10 Seam wall placement (MISMATCH)

NEMO (`usrdef_zgr.F90:152-170`, `IF (ln_Iperio)`): sets the first and last
columns to land (`zbathy=0` → `k_bot=0`) **outside** the channel band, keeping
them wet inside it. In the `tmask` this lands at the periodic-seam / halo columns
(i=0, i=51 in the 52-wide haloed array), wet only at lat [−64, −45] (the ACC
channel). **All 48 interior columns stay fully wet.** The basin is closed by
walling the periodic wrap, not by removing an ocean column.

legoESM (`init_latlon_cgrid.py::partial_periodic_seam_wall_latlon`, called in
`dino.py::dino_lat_lon_initial_state_arrays`): marks **interior column 0** as
land outside the channel band. This removes one ocean column (surface wet cells
9201 vs NEMO's 9360 — the 159-cell difference is column 0 minus its channel
band) and shifts the western wall one cell east.

**Feasibility of matching NEMO** (`scratchpad/seam_feasibility.py`): a variant
that keeps all cells wet and instead zeros the wrap-seam u-face (`u_mask` is
shape `(n_lat, n_lon+1)` with explicit seam faces 0/`n_lon`) outside the channel
**reproduces NEMO's domain exactly — 9360 wet cells, all 48 interior columns
wet, re-entrant only in the channel**. But a naive `u_mask` override **blows up
in 20 days**: legoESM's C-grid handles longitude periodicity with `jnp.roll`
pervasively and derives all face masks from `land_mask`, so a single seam-face
override does not propagate to the momentum / tracer / continuity / barotropic
operators — the western boundary stays effectively open and the wind-driven flow
runs away.

**Conclusion**: legoESM's land-column seam wall is a deliberate, *stable*
architectural choice (a land cell zeros every adjacent face consistently).
Matching NEMO's seam wall exactly needs genuine **partial-periodic C-grid
support** — threading a seam mask through every `jnp.roll`-based operator — which
is a scoped infrastructure project, high-risk for a 1-column (~2% of a 48-column
basin) domain difference. Recorded as a characterized follow-up; the target is
proven reachable (exact 9360-cell match), only the stable implementation remains.

## #9 Bowl bathymetry extent (MINOR)

The bowl **formula** is identical (row 5). The numeric bathymetry (legoESM in
the correct lon frame vs NEMO `e3t_0·tmask`) is corr 0.92, mean bias −56 m, rms
183 m — legoESM is a few percent shallower in the gyre. The residual is the
boundary-*extent* convention fed to `exp_bathy`:

- NEMO `zgr_get_boundaries` uses `MAXVAL/MINVAL(glamu)` and `MAXVAL/MINVAL(gphiv)`
  — the **U/V face** coordinates (`glamu ∈ [2.00, 49.00]`, `gphiv ∈ [−68.97,
  69.33]`) — plus a periodicity correction, and `distPhi = cos(gphiv_max)·distLam`.
- legoESM `dino_bathymetry` uses `cfg.lon_west/east` and `±lat_max_deg=70`, with
  `distPhi = cos(70°)·dist_lam`.

These differ by ~half a cell in extent and ~3% in `distPhi`. A standalone
reproduction of the residual was **not clean** (a hand-rebuilt bowl scored corr
0.72 — worse than the shipped code's 0.92, i.e. the reconstruction was missing
detail), so the precise cause is not isolated and **no speculative code change
was made** (verify-against-code discipline: do not patch what you cannot
precisely diagnose). corr 0.92 is already a good match; deferred until the exact
`zgr_get_boundaries` convention can be reproduced cell-for-cell.

## Corrections to earlier inferences (the discipline lesson)

Three claims made by inferring from processed arrays were later **refuted by
reading NEMO's source** — recorded so the failure mode is not repeated:

1. "NEMO uses cosh, legoESM uses exp" — **wrong**. `nn_botcase=1` is *named*
   "cosh" in the namelist comment, but CASE(1) calls `exp_bathy` (`usrdef_zgr.F90:271-286`).
2. "NEMO has no seam wall, all columns wet" — **wrong**. Artifact of stripping
   the halo, exactly where NEMO's wall lives (`usrdef_zgr.F90:152-170`).
3. "z-coordinate mismatch, 4506 vs 4000 m" — **wrong**. 4506 m is the `e3t_1d`
   sum; 4000 m is `gdepw` (max depth) — different quantities, both consistent.

Separately, the "legoESM basin is 2280 m shallow" seen in the first comparison
plots was a **harness bug** (the comparison built `dino_lat_lon_state` on the
bridge grid — NEMO lon frame [1.5, 48.5] — with the default config lon frame
[−50, 0], collapsing the bowl), not a legoESM defect. The production bowl in the
correct frame is 3758 m (corr 0.92 vs NEMO). Use `nemo_faithful_dino_config`
(co-sets the lon frame) for matched-grid comparisons.

## Provenance

NEMO source: `~/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/MY_SRC/usrdef_*.F90`.
Harnesses: `~/oracle-builds/nemo5/gap_audit/` and session scratchpad
(`dino_bathy_fixed.py`, `dino_mask_check.py`, `seam_feasibility.py`). Grid/IC/
forcing matches: `dino_tendency_certificate.md`.
