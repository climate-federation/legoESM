# PREREG — vertical-mixing lane V5 (preregistered round 15, vmix_r15)

Frozen before any V5 code or measurement. No V5 record exists yet; nothing is scored this round.
Decisions 128-130 and 136 in force. Codex review: paused (quota). Evidence: `gridladder_vmix_rounds/round15/`.

**V5 = certified V4 + ORCA2's horizontally varying background diffusivity, exactly as ORCA2 sets it** — nothing else:
- `ORCA2_OMIP_L4/EXP00/namelist_cfg:401` `nn_havtb = 1` (V4's deck: 0, the `namelist_ref` value).

NEMO (all in `ORCA2_OMIP_L4/BLD/ppsrc/nemo`), `zdf_phy_init` (`zdfphy.f90:126`):
- `zdfphy.f90:215` `avtb_2d = 1`; `zdfphy.f90:217-224` with `nn_havtb == 1`, three WHERE statements on `gphit`:
  `1 - 0.09*(gphit+15)` on [-15, -5), `0.1` on [-5, 5), `0.1 + 0.09*(gphit-5)` on [5, 15).
- `zdfphy.f90:226-228` seeds `avt_k = avtb_2d*avtb(jk)*wmask` (avtb = rn_avt0 = 1.2e-5, nn_avb = 0).
- `zdfphy.f90:283` then calls `zdf_iwm_init`, which resets `avtb_2d = 1` (`zdfiwm.f90:443`) and `avtb = 1e-10`.

## Survey facts (read off the code / files, not measured)

- **Where the shape reaches the run.** Only the `avt_k` seed: the per-step floor `MAX(zav, avtb_2d*avtb)`
  (`zdftke.f90:710`, `:717`) reads `avtb_2d` after IWM's reset (= 1). The seed is read once, at kt 1, by the TKE
  buoyancy term `- p_avt*rn2` (`zdftke.f90:424`, `tke_tke`), because this deck is a cold start (`ln_rstart = .false.`)
  and `tke_avn` overwrites `avt_k` at the end of kt 1 (`zdftke.f90:710`). No other reader of `avtb_2d`
  (searched `avtb_2d` in the build's ppsrc: `zdfphy.f90`, `zdftke.f90`, `zdfiwm.f90`, `zdf_oce.f90` only).
- **Input files**: none new. `gphit` comes from the domain file already in the ladder's input set
  (`ln_read_cfg = .true.`, `ORCA2_OMIP_L4/EXP00/namelist_cfg:33-34`); the card already reads it. No band T point sits
  exactly on -15, -5, 5 or 15 degrees; 4,302 wet columns lie in [-15, 15).
- **Compiled in**: `zdfphy.f90` and `zdf_oce.f90` byte-identical in ORCA2_OMIP_L4, V1's record build and G5's
  writer-free build; `zdfphy.o` `.text`/`.rodata` identical in all three; no FMA instruction in `zdfphy.o`; all three
  built with `-fdefault-real-8` (so `0.09`, `15.`, `0.1` are doubles). Nothing to build.
- **No new model state**: `avt_k` is already the card's carried `tke_avt` field, read by the card's TKE as the
  pre-closure diffusivity. V5 changes its initial values only.
- **Our repo** (searched `havtb|avtb_2d|equatorial` in packages, src, scripts, tests): the full ORCA2 card already
  transcribes the three WHERE statements inline for its seed; the ladder's V seed (`_orca2v1_tke_initial_state`)
  uses `rn_avt0*wmask`. Factor the full card's statements into one helper used by both; no second copy.

## The card change (V5-selected; every other card unchanged)

- Card `ORCA2V5-havtb` = V4's card; its `tke_avt` seed = `(avtb_2d*rn_avt0)*wmask` with `avtb_2d` from the shared
  helper on the domain file's `gphit`. Model config identical to V4's.
- Deck = V4's deck with `nn_havtb = 1` (only diff); ice deck identical. The reader requires 1 on V5, 0 elsewhere.

## Predictions

- **P1 (record).** The V5 run prints `nn_havtb  = 1`, keeps every V4 line (IWM, DDM, EVD, TKE), completes >= 10
  steps, three twins agree. Falsifier: the acquisition refuses.
- **P2 (the shape acts, and only where NEMO's code says).** NEMO's V5 kt-1 post-`zdf_phy` `en` differs from V4's
  admitted frame in >= 1 wet W cell, and every differing cell (any of avm, avt, avs, en) lies in a column with
  -15 <= gphit < 15. Falsifier: no differing cell (then the seed is inert: report, do not certify), or a differing
  cell outside the band (then something else moved).
- **P3 (score, PLAUSIBLE).** V5 bit-for-bit kt 1..10 under both labels with no further change (the only new
  statement is three double-precision expressions in numpy on the same `gphit`). First suspect if not: the
  association of `avtb_2d*avtb(jk)*wmask`. Falsifier: any unequal cell.
- **P4.** One-step jaxprs of the full ORCA2, DINO, V1-V4 cards identical to this round's D136 tip; V5's one-step
  jaxpr identical to V4's; V1-V4 initial states unchanged; V5's initial state differs from V4's only in `tke_avt`,
  only in band columns.
