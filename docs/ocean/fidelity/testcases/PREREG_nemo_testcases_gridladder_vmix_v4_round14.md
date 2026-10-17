# PREREG — vertical-mixing lane V4 (preregistered round 14, vmix_r14)

Frozen before any V4 code or measurement. No V4 record exists yet; nothing is scored this round.
Decisions 128-130 in force (D129: the IWM-forced TKE floors arrive with this rung). Codex review: paused (quota).
Evidence: `gridladder_vmix_rounds/round14/`.

**V4 = certified V3 + ORCA2's internal-wave mixing, exactly as ORCA2 sets it** — nothing else moves:
- `ORCA2_OMIP_L4/EXP00/namelist_cfg:396` `ln_zdfiwm = .true.`
- `ORCA2_OMIP_L4/EXP00/namelist_cfg:421-436` `&namzdf_iwm`: `ln_mevar = .false.` (:423), `ln_tsdiff = .true.` (:424),
  `cn_dir = './'` (:426), and the six fields of `zdfiwm_forcing_orca2` (:430-435): `power_bot`, `power_cri`,
  `power_nsq`, `power_sho`, `scale_bot`, `scale_cri` (frequency -12, no time interpolation, climatological, yearly).

NEMO (all in `ORCA2_OMIP_L4/BLD/ppsrc/nemo`): `zdf_iwm` = `zdfiwm.f90:140` (SUBROUTINE), called at `zdfphy.f90:372`
in `zdf_phy` after `zdf_ddm` (:363); `zdf_iwm_init` = `zdfiwm.f90:373`, called at `zdfphy.f90:283` in `zdf_phy_init`.

## Survey facts (read off the code / files, not measured)

- **Input file**: `zdfiwm_forcing_orca2.nc` is NOT in the ladder's input set (`round21/oracle_g4_r21/p6_a`). It is in the
  official `ORCA2_ICE_v5.0.0` input tarball already on disk (`nemo-testcases-l4/inputs/`, fetched with its md5/sha256 in
  `fetch.log`), sha256 `f4279899be77119f…`; every ladder input in p6_a is byte-identical to that same tarball (checked:
  domain, T, S, chlorophyll, runoff, SSS, t_10); every ORCA2_OMIP_L4 run in this campaign reads this exact file; the full
  ORCA2 card already reads it. Nothing is fetched and nothing is vendored: the acquisition copies one more file from the
  same tarball, pinned by sha256. One record (`time`=1), float64, 148 x 180 = ORCA2's global grid.
- **Compiled in**: `zdfiwm.f90` is byte-identical (sha256 `13ba29ab…`) in ORCA2_OMIP_L4, V1's record build and G5's
  writer-free build; `zdfiwm.o` `.text` and `.rodata` are identical in all three (only `.rodata.str1.8` differs: it
  holds the build's own source path). `nemo.exe` of both reused builds links `zdf_iwm`/`zdf_iwm_init`.
- **No new model state**: the six maps are static 2-D inputs read once (`zdfiwm.f90:476`); the model already takes them
  as closure constants (`iwm_forcing`), and the recipe already has the slot. `zdf_iwm` reads step-entry `rn2`,
  `r3t(Kmm=Nbb)`, `ht_0`, the 1-D `gdept_1d`/`e3w_1d`, `wmask` — all on the card.
- **What it changes besides adding Kz** (NEMO's own coupling, D129):
  - `zdf_tke_init` (`zdftke.f90:835-838`): `rn_emin = 1e-10`, `rmxl_min = 1e-3` (and so `rn_mxl0 = rmxl_min = 1e-3`,
    `:853-856`); the from-rest `en` seed is `rn_emin*wmask` (`:915`).
  - `zdf_iwm_init` (`zdfiwm.f90:441-443`), run after `zdf_phy_init` set the backgrounds (`zdfphy.f90:206-223`):
    `avmb = rnu = 1.4e-6`, `avtb = 1e-10`, `avtb_2d = 1`. The `avt_k`/`avm_k` seeds (`zdfphy.f90:227-228`) are set
    BEFORE that reset and keep `rn_avt0`/`rn_avm0`.
- **Finding for the lane's plan (no action)**: because `zdf_iwm_init` resets `avtb_2d = 1` after `nn_havtb` shaped it,
  ORCA2's `nn_havtb = 1` (cfg :401) reaches only the `avt_k` seed (`zdfphy.f90:227`), not the per-step floor
  (`zdftke.f90:710`). The planned V5 (`nn_havtb`) is therefore nearly inert on ORCA2's deck.
- **Our repo** (searched `zdfiwm|zdf_iwm|ln_zdfiwm|iwm|internal.wave|de.lavergne|power_cri|zdfiwm_forcing` in packages,
  src, scripts, tests): one IWM module (`internal_wave_mixing.py`, a generic port of NEMO 5.0.1 for ORCA1: clamps on
  the exponent and the algebraic denominator, tree-reduced column sums, 3-D cell depths, refuses `tsdiff`); the shared
  K-profile builder adds it after DDM; the full ORCA2 card uses it with `tsdiff=False`; the shared forcing loader
  (`iwm_forcing.load_iwm_forcing`) and the ORCA2 card's `_orca2_iwm_forcing` read this exact file. Extend all of them;
  no second module.

## The card change (V4-selected; every other card unchanged)

- A NEMO-literal `zdf_iwm` form in the EXISTING IWM module (`IWMConfig.form`, default the current generic form):
  `zdfiwm.f90:233-317` in source order — `ht_0*(1+r3t)` and `gdept_1d*(1+r3t)`, `e3w_1d*(1+r3t)`, the sequential
  `zfact3`/`zfact4` sums over jk = 2..jpkm1, raw `rn2` in `zReb`, `ln_mevar = F`, `ln_tsdiff = T` ratio with LOG10 and
  TANH, clamp `[1.4e-7, 1e-2]*wmask`, then `avs += zav_wave*zav_ratio`, `avt += zav_wave`, `avm += zav_wave`;
  EXP/TANH/LOG10 through the scalar-libm precision policy; every operation source-rounded.
- Composed in the shared K-profile builder after the literal DDM (NEMO order), adding onto V3's salt diffusivity.
- Maps: the existing loader, read from the deck root (`zdfiwm_init`: powers times `smask0`, `hcri = 1/scale_cri`).
- TKE: `rn_emin = 1e-10`, `rmxl_min = 1e-3` (the forced arm, not the derived one), the `rn_mxl0` overwrite kept;
  floors `avmb = 1.4e-6`, `avtb*avtb_2d = 1e-10`; `en` seed `1e-10*wmask`; `avt_k`/`avm_k` seeds unchanged.
- Card `ORCA2V4-iwm`; deck = V3's deck plus ORCA2's `ln_zdfiwm` line and `&namzdf_iwm` group (only diff).

## Predictions

- **P1 (record).** The V4 run resolves `ln_zdfiwm = T`, `ln_mevar = F`, `ln_tsdiff = T`, prints `zdf_iwm_init :
  internal wave-driven mixing`, `force   rn_emin = 1.e-10 and rmxl_min = 1.e-3` and the four `Dissipation ... TW`
  lines, keeps every V3 switch, completes >= 10 steps, three twins agree. Falsifier: the acquisition refuses.
- **P2 (IWM is not inert).** (a) NEMO's V4 kt-1 post-`zdf_phy` avt differs from V3's record frame in >= 1 wet W cell
  (expected: almost every wet interior cell, since `zav_wave >= 1.4e-7` everywhere wet); (b) NEMO's V4 T differs from
  V3's record at kt 2 in >= 1 wet cell. Falsifier: either count is 0 (then report, do not certify).
- **P3 (first owner if V4 misses, PLAUSIBLE).** One of: the column depth operand (`ht_0` vs the card's bathymetry
  sum), the sequential `zfact3`/`zfact4` sums (`zdfiwm.f90:249-250`), LOG10/TANH in the ratio (`:303-305`), or the
  stored `hcri = 1/scale_cri` (`:485`). Falsifier: the walk names another owner, or V4 is bit-for-bit kt 1..10 under
  both labels (then V4 certifies as built).
- **P4.** Flag-off one-step jaxprs of the full ORCA2 and DINO cards identical to base `0b9443cef279`; V2's and V3's
  one-step jaxprs identical too; V1-V3 card fields and initial states unchanged (their card tests).
