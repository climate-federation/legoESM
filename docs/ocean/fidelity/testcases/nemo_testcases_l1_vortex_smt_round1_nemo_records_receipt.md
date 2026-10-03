# Round 211 / VORTEX_SMT round 1 — the NEMO side: a seamount, partial steps, four records

**Decision 88 (user, 2026-10-03), operator note CC.** This round builds NEMO's
VORTEX with topography and acquires its records. No legoESM card, no legoESM
physics or configuration change, no ladder: those are round 2 and later.

---

## 0. THE ONE DEVIATION FROM THE ROUND BRIEF, AND WHY IT IS NOT A CHOICE

The round brief said to copy the certified instrumented builds with their
**cpp keys unchanged** and to set `ld_zps = .TRUE.`. **Those two instructions
cannot both hold, and NEMO is the one that says so.** The certified VORTEX
cards compile `key_qco key_vco_1d key_RK3`, and
`src/OCE/DOM/domzgr.F90:259` is

```fortran
IF( l_zps )   CALL ctl_stop( 'STOP','domzgr: key_vco_1d and l_zps=T are incompatible. Fix usrdef_zgr !' )
```

Under `key_vco_1d` there is no three-dimensional `e3t` array at all — `dom_oce.F90:359-362`
allocates the 1-D vertical arrays only — so a partial cell has nowhere to
live. The key that NEMO itself labels **"z-partial cells"** is `key_vco_1d3d`
(`domzgr.F90:242-243`), and it is the key both shipped partial-step
user configurations (`tests/OVERFLOW`, `tests/IWAVE`) guard their zps branch
with (`lk_vco_1d3d`).

So this round changes **one** cpp key, `key_vco_1d` → `key_vco_1d3d`, and
nothing else: `key_qco` and `key_RK3` are carried, `key_xios` and `key_agrif`
are dropped exactly as every other VORTEX variant drops them. It is reported
here as the first line of the receipt rather than buried, because it is the
single place where the built configuration departs from "the certified build
with a new bathymetry". Decision 88's own words are "partial steps ON
(`ln_zps`)", and partial steps are reachable by exactly one route.

The acquisition re-reads that refusal out of the compiled tree before it
builds anything (`run.sh`, the `SMT_ZGR` preflight block), so if a future NEMO
drops the restriction the script says so instead of quoting this paragraph.

---

## 1. The bathymetry, and where every constant comes from

```
h(x,y) = 5000 m  -  1000 m * exp( -((x-x0)^2 + (y-y0)^2) / (150 km)^2 )
(x0, y0) = (-300 km, 0 km)
```

`H0 = 5000 m`, `A = 1000 m`, `L = 150 km` and the 300 km westward shift are
the **user's**, stated in Decision 88. The only thing this round had to read
out of NEMO is where the vortex centre is and which way west points:

| quantity | value | citation |
|---|---|---|
| vortex centre | `(glamt, gphit) = (0, 0)` km | `tests/VORTEX/MY_SRC/usrdef_istate.F90:79-80` sets `zx = glamt*1.e3`, `zy = gphit*1.e3`, and `:86` puts the anomaly at `exp(-(zx²+zy²)/zlambda²)`; the velocity does the same at `:98-99`, `:105` |
| `glamt` units | kilometres | `tests/VORTEX/MY_SRC/usrdef_hgr.F90:78` "Position coordinates (in kilometers)" |
| west | `-glamt`, for `nn_rot = 0` | `usrdef_hgr.F90:108`: `plamt = zroffsetx + rn_dx*1.e-3*(zti-0.5)`, i.e. `glamt` increases with the i index |
| the deck's `nn_rot` | `0` | `tests/VORTEX/EXPREF/namelist_cfg:24` |

Everything else is the shipped 30 km deck, untouched: `rn_dx = rn_dy = 30000`,
`rn_dz = 500`, `rn_Dt = 2880`, `rn_ppgphi0 = 38.5`, `rn_ppumax = 1.0`,
`nn_e = 48`, both lateral operators OFF, the shipped simplified equation of
state (decision 69). The deck patches change four lines and no physics:
`cn_exp`, `nn_itend`, `nn_stock`, `ln_meshmask`.

---

## 2. The partial-step construction, statement by statement

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/vortex_smt_usrdef_zgr.F90`.
Its module header carries the same list; this is the short form.

| what | transcribed from | note |
|---|---|---|
| 1-D reference coordinate `zgr_z` | `tests/VORTEX/MY_SRC/usrdef_zgr.F90:89-161` | verbatim; ten uniform 500 m levels unchanged |
| land mask `k_top` | `tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:113-115` | `z2d = 1`, `lbc_lnk`, `k_top = NINT(z2d)`; the same mask VORTEX's own `:187-193` builds |
| minimum partial cell | `OVERFLOW:204` | `ze3min = 0.1_wp * rn_dz` = 50 m |
| bottom level | `OVERFLOW:209-212` | `k_bot = jpkm1`; `DO jk = jpkm1,1,-1 ; WHERE( zht < pdepw_1d(jk)+ze3min ) k_bot = jk-1` |
| land applied to `k_bot` | `OVERFLOW:143 / :160 / :197` | `k_bot = k_bot * k_top` |
| partial `e3t` | `OVERFLOW:221-225` | `e3t(ik) = MIN(zht, pdepw_1d(ik+1)) - pdepw_1d(ik)`; `e3t(ik+1) = e3t(ik)` |
| `e3u`, `e3v` | `tools/DOMAINcfg/src/domzgr.F90::zgr_zps:1162-1170` | `MIN` of the two neighbouring `e3t` |
| `lbc_lnk` on `e3u`, `e3v` before `e3f` | `zgr_zps:1176-1177` | |
| `e3f` | `zgr_zps:1191-1197` | `MIN` of the two neighbouring `e3v` |
| `lbc_lnk` on `e3f` | `zgr_zps:1198` | |

**The exclusion that mattered, named per the reuse rule.** OVERFLOW's zps
branch finishes with `pe3u = pe3v = pe3f = pe3t` and documents why that is
legal *there*: `OVERFLOW:228-231`, "HERE OVERFLOW configuration: e3 increases
with i-index and identical with j-index". A seamount varies in **both**
directions, so that shortcut is wrong here and was **not** ported; NEMO's
general two-dimensional rule from the DOMAINcfg tool was transcribed instead.

**Three more exclusions, deliberate:** `zgr_zps`'s duplication of row `jj=1`
onto `jj=2` (`:1203-1207`, which NEMO itself flags `!!gm bug ?` and which is
ORCA-specific), its ice-shelf branches (no cavities here), and its
`WHERE(e3 == 0) e3 = e3_1d` repair (`:1179-1184`, `:1200-1202`) — nothing can
be zero here because every array is initialised to the 1-D reference first.

**What stays 1-D.** Under `key_vco_1d3d`, `dom_zgr` calls `usr_def_zgr` with
exactly four optional arrays, `e3t_3d e3u_3d e3v_3d e3f_3d`
(`domzgr.F90:265-269`), and applies `lbc_lnk` to all four itself afterwards
(`:273-274`). `e3w`, `gdept` and `gdepw` remain the 1-D reference. This is
NEMO's own zps shape, not a simplification taken here.

---

## 3. Can ten 500 m levels carry a 1000 m seamount? — PREDICTED, THEN MEASURED

Pre-registered before any run (`phase3/vortex_smt/round1/predictions.md`):

