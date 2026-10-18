# Round 313 preregistration — V4 internal-wave mixing and V3 double diffusion on ORCA2 under D122

Frozen before any measurement (addendum 108; D122, D130).

Survey (config only, no numbers from runs): of the six ORCA2 cards, only
rung 10 runs internal-wave mixing, on the generic form with `tsdiff`
False; the deck sets `ln_tsdiff = .true.`
(`ORCA2_OMIP_L4/EXP00/namelist_cfg:424`). Double diffusion is off on all
six; the deck sets `ln_zdfddm = .true.` (`:393`). Rung 0 and OMT-1..4 run
constant mixing; they stay untouched.

Code imported, not re-implemented: gridladder-vmix `1fc44b9b98`'s
`internal_wave_mixing.py`, `k_profiles.py`, `tke.py` (byte-identical) and
its four model-file hunks. V3's double diffusion (with its divisor fix) and
EVD step-entry trigger are already here (round 309).

Arms on rung 10, both labels, vs the round-312 shipped card (`control`):
- `IWM_LIT`: internal-wave form generic -> literal, tsdiff stays off.
- `IWM`: literal form + `tsdiff` on (the deck's setting) — the D122 row
  "literal internal-wave mixing".
- `DDM`: literal double diffusion on the shipped (generic-IWM) card.
- `BOTH`: `IWM` + `DDM`.
Rung 0, OMT-1..4: `control` and `D122` only.

Predictions:
- R313-P1: import-tree controls equal round 312's shipped D122 rows, 7/7
  (new code is reached only under the literal form).
- R313-P2: `DDM` refuses at trace: "zdf_iwm's avs update is missing".
- R313-P3: `IWM_LIT`, `IWM`, `BOTH` run and move rows on both labels.
- R313-P4: `IWM` and `BOTH` pass D122 (c)+(d) on both labels (0 exact
  rows lost; first over-bar row unchanged). Direction: no prediction.
- R313-P5: after selection, rung 0 / OMT-1..4 move 0 rows; their configs
  unchanged; GYRE, DINO and the four VORTEX step jaxprs identical.

Falsifiers: any import-tree control differing from round 312 (P1); `DDM`
running (P2); an exact row lost or first over-bar row moved away (P4);
any constant-mixing card or GYRE/DINO/VORTEX moving (P5).

A feature failing D122 stays off and its numbers are reported. One passing
but leaning strongly away is asked about, not landed silently.
