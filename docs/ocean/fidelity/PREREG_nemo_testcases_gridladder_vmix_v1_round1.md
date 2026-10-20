# PREREG — vertical-mixing lane, rung V1, round 1 (vmix_r1)

Frozen before any V1 record exists. Lane B of Decision 128: certified G5 plus
ORCA2's vertical mixing, one switch per rung. Base: branch
`fidelity/nemo-testcases-orca2-gridladder-vmix` at `1405ca632` (G5 certified,
G5 round 4).

## 1. Survey — RESOLVED vertical-mixing deck, G5 vs ORCA2

Resolved = `cfgs/SHARED/namelist_ref` overlaid by the case's `namelist_cfg`.
G5 = `scripts/validate/ocean_fidelity/testcases/nemo_testcase_g5_orca2ice/namelist_cfg`,
ORCA2 = `cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg`. Only differing rows are listed.

| key | G5 | ORCA2 | source |
|---|---|---|---|
| `ln_zdfcst` | T | F | G5 :301; ORCA2 states none, ref :1180 |
| `ln_zdftke` | F | T | G5 :303; ORCA2 :389 |
| `ln_zdfevd` (`nn_evdm`=0, `rn_evd`=100 both) | F | T | G5 :306; ORCA2 :390-392 |
| `ln_zdfddm` (`rn_avts`=1e-4, `rn_hsbfr`=1.6 both) | F | T | G5 :309; ORCA2 :393-395 |
| `ln_zdfiwm` | F | T | G5 :310; ORCA2 :396 |
| `nn_havtb` | 0 | 1 | G5 :315; ORCA2 :401 |
| `namzdf_tke.nn_mxlice` | 0 (ref, inert) | 2 | ref :1237; ORCA2 :415 |
| `namzdf_iwm.sn_*` (6 files) | `'NOT USED'` (inert) | `zdfiwm_forcing_orca2` | ref :1348-1353; ORCA2 :430-435 |

Equal in both (no row): `rn_avm0`=1.2e-4, `rn_avt0`=1.2e-5, `nn_avb`=0,
`ln_zad_Aimp`=F, ric/gls/osm/npc/mfc/swm off, every other `namzdf_tke` key
(ORCA2's `nn_mxl`=3, `nn_etau`=1, `ln_mxhsw`=F restate the reference), and
`namzdf_iwm.ln_mevar`=F, `ln_tsdiff`=T. Runoff-mouth mixing
(`ln_rnf_mouth`, `rn_avt_rnf`) is already G5's (ORCA2's).

Code-level couplings the deck table cannot show (compiled
`ORCA2_OMIP_L4/BLD/ppsrc/nemo`; the G5 binaries' `zdftke.f90` and
`zdfphy.f90` are byte-identical to it):

- **IWM rewrites TKE's floors.** `zdf_tke_init` sets `rn_emin=1e-10`,
  `rmxl_min=1e-3` when `ln_zdfiwm` is true, else `rmxl_min =
  1e-6/(rn_ediff*SQRT(rn_emin))` = 1e-2 m with `rn_emin`=1e-6
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:835-841`); `rn_mxl0` is then
  overwritten by `rmxl_min` (`:853-856`).
- **IWM rewrites the backgrounds.** `zdf_iwm_init` sets `avmb=rnu`,
  `avtb=1e-10`, `avtb_2d=1` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:441-443`),
  after `zdf_phy_init` seeded `avt_k`/`avm_k` from the namelist backgrounds
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:226-229`). So in ORCA2,
  `nn_havtb=1` shapes only the initial `avt_k`; the per-step floors at
  `zdftke.f90:709-710` are uniform.
- **Drag sets TKE's bottom value.** `en(mbkt+1)` from bottom friction runs
  only when `ln_drg_OFF` is false (`zdftke.f90:285-293`). G5 (and V1) keep
  `ln_drg_OFF=T`; lane A's drag rung moves it — a cross-lane interaction at
  merge.
- **TKE needs no companion to run.** `zdf_phy_init` accepts exactly one
  closure (`zdfphy.f90:263-270`); EVD, DDM and IWM are independent calls
  after it (`:359-372`). No cpp key is involved.

Proposed one-switch order: **V1 TKE → V2 EVD → V3 DDM → V4 IWM (+ its
forcing file; moves TKE floors and backgrounds) → V5 `nn_havtb=1`** (after
IWM its only effect is the initial `avt_k` seed, which is all it does in
ORCA2). V3 and V4's `ln_tsdiff` need separate salt diffusivities, which the
shared implicit tracer solve does not carry yet (the full ORCA2 card's
declared gap).

## 2. V1 definition

V1 = G5's deck + `ln_zdftke=T`, `ln_zdfcst=F` + ORCA2's four `&namzdf_tke`
lines (`nn_mxl=3`, `nn_etau=1`, `ln_mxhsw=F`, `nn_mxlice=2`), everything else
G5's (`ln_zdfiwm=F`, `ln_drg_OFF=T`, `nn_havtb=0`). Deck SHA-256
`5b5dd21ee452d98797a822320e2cdfcf8fa21b35c4c94b5d2a9ff7fcc814ba2c`; SI3
overlay unchanged (`9d5ae3fa...`).

**Not settled by ORCA2's deck (DECISION_NEEDED, built as option 1):** with
IWM off, TKE resolves its own floors (minimum TKE 1e-6 m²/s², minimum mixing
length 1 cm) instead of ORCA2's IWM-forced 1e-10 / 1 mm. Option 1 (pick):
keep one switch; the floors move at the IWM rung. Option 2: bundle IWM into
V1.

## 3. Card (reuse)

Searched `packages/ocean` for `zdftke`/`tke`/`mxlice`/`etau`: the NEMO TKE
already lives in `physics/vertical_mixing/tke.py`, and the ORCA2 identity
(`whole_step_identity="orca2_vector_een_c2"`) already carries every ORCA2 TKE
selector inside G5's (inert) `tke` block. V1 selects it (`scheme="tke"`,
`vmix_background_mode="nemo_max_floor"`) and states the deck values; the
only config moves vs G5 are `scheme`, the background mode, `etau_mode` and
`etau_htau_mode` (nn_etau=1 / nn_htau=1, the DINO card's existing arm).
Initial `en`/`avm_k`/`avt_k`/`dissl` follow `tke_rst` from rest
(`zdftke.f90:913-916`) and `zdfphy.f90:226-229`. Found missing:
`nn_mxlice=2` (no arm in legoESM; G5's record has no `hm_i`) → card label
`tke_nn_mxlice2_under_ice_mixing_length`, and the acquisition records `hm_i`.

## 4. Predictions (frozen)

- **P1 (boundary, PLAUSIBLE).** NEMO's writer-free V1 reference stops in
  `stp_ctl` at kt 11 like G5 (G5's stop is a deep `|V|`=10.25 m/s at level
  27, where TKE sits on its background floor). Falsifier: a different stop
  step or 100 completed steps. The acquisition records to
  `min(10, completed)` either way.
- **P2 (first non-bit statement).** `nn_mxlice=2`'s surface mixing length
  under ice, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:619-623`
  (SUBROUTINE `tke_avn`, :531), at kt 1. Observable: kt-1 step-final T/S
  differ only in wet columns with `fr_i>0`, U/V only on faces touching them,
  ssh exact at kt 1; every ice-free column bit-identical through kt 1.
- **Falsifiers of P2.** (a) any kt-1 mismatch in a column with `fr_i==0` and
  no ice-touching face — an ice-free statement owns the walk; (b) ice columns
  exact at kt 1; (c) mismatches in ice columns that persist when NEMO's
  recorded `hm_i` is inserted in an offline replay of `tke_avn` — then the
  ice attenuation (`nn_eice=1`, `:259-264`) or another ice-side statement owns
  it, not `nn_mxlice`.

## 5. Acquisition

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_v1_orca2tke/run.sh`:
writer-free reference = G5's pinned writer-free binary, reused unbuilt
(TKE is a run-time switch); record build = G5's record MY_SRC set plus one
additive write-only `hm_i` writer called right after the reused surface
writer. TKE's state (`en`, `avt_k`, `avm_k`, `dissl`) is in NEMO's restart
(`zdftke.f90:922-926`). Order: two-step smoke → 100-step writer-free boundary
→ twins A/B/C to `min(10, completed)` rounded to `nn_fsbc=2`, restarts at
every step 2..N, never step 1. Passivity (writer-free vs twin restart
identity) is next round's admission, as in G5 round 2.
