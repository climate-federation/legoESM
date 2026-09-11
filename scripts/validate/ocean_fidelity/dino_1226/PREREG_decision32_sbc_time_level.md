# PREREG — decision 32: the surface restoring flux's TIME LEVEL

Written BEFORE the edit exists. The measurement that follows is scored against
the numbers in §3 and nothing else.

## §1 — The statement, as NEMO compiles it

`cfgs/DINO/BLD/ppsrc/nemo/usrdef_sbc.f90`, the `np_DINO` branch of
`usr_def_sbc`, which `sbc` reaches as `CALL sbc( kstp, Nbb, Nnn )`
(`stpmlf.f90:173` — so the routine's `Kbb` IS the step's before level):

```fortran
:388   sfx(ji,jj) = ( rn_srp * ( ts(ji,jj,1,jp_sal, Kbb) - zsstar(ji,jj) ) ) * tmask(ji,jj,1)
:436   qns(ji,jj) = (  rn_trp * ( ts(ji,jj,1,jp_tem, Kbb) - ztstar(ji,jj) ) &
:437        &      - emp(ji,jj) * ts(ji,jj,1,jp_tem, Kbb) * rcp           &
:438        &      - zqsr_dayMean(ji,jj)                         ) * tmask(ji,jj,1)
```

Every tracer read in the restoring flux is `Kbb`. legoESM's
`apply_dino_lat_lon_surface_forcing` reads `state.T`/`state.S` — the NOW level.

**THE EULER-START CASE.** `Nbb = 1 ; Nnn = 2` are distinct SLOTS from
`nemogcm.f90:368` onward, so `Kbb` is never aliased to `Kmm`; what makes step 1
insensitive is that `usr_def_istate` fills both slots with the SAME field, so
`ts(Kbb) == ts(Kmm)` at `kt = nit000` from rest. legoESM's equivalent is
`state.T_before is None` on the first step (the forward-Euler start that then
populates it), and the transcription falls back to `state.T` there. **Step 1
must therefore be BIT-IDENTICAL, and that is a falsifier, not a hope.**

## §2 — No new carried state, no knob

`LatLonCGridOceanState.T_before`/`S_before` (`state.py:607-608`) already exist
and are already populated by the leap-frog integrator — this card carries them
today. A card with a different outer integrator has them `None` and is
BYTE-UNCHANGED by construction, which is why this ships without a config
field: there is nothing to select.

## §3 — PREREGISTERED NUMBERS (Rule 1b: the bar is in the gate, not in judgment)

Measured before the edit, `kt2_leapfrog_gate.py`, wet-masked rms of the state
at the end of kt=2 against NEMO's kt=2 restart:

| field | before | NEMO's own kt=2 step | fraction |
|---|---|---|---|
| T | 6.1224e-06 K | 1.3384e-02 | 0.0005 |
| S | 1.1380e-05 psu | 2.2399e-03 | 0.0051 |
| u | 1.4101e-04 m/s | 2.3409e-03 | 0.0602 |
| eta | 3.4306e-04 m | 3.4988e-02 | 0.0098 |
| v | 1.4292e-05 m/s | 2.5491e-03 | 0.0056 |

`kt1_surface_gate.py --kt2-dir` sizes THIS statement, off NEMO's own restarts
with no model run, at `5.8839e-06 K` pooled = **0.961 of the T row** and
`0.008` of the S row.

**PREDICTION.** The sizing is a MAGNITUDE, so it bounds the outcome from two
sides and both are written down:

* perfectly aligned → `|6.1224 - 5.8839|e-06 = 2.39e-07 K` (0.039x)
* orthogonal → `sqrt(6.1224^2 - 5.8839^2)e-06 = 1.70e-06 K` (0.277x)

**PREREGISTERED WINDOW for the kt=2 T row: `[2.0e-07, 1.9e-06] K.`**

**FALSIFIER.** If the kt=2 T row does not fall below `3.06e-06 K` (half its
current value), the 0.961 sizing is REFUTED and this edit does not own the
row — report that, do not re-explain it.

**SALT.** The S row is preregistered to move by less than 1% (sizing 0.008),
i.e. to stay inside `[1.127e-05, 1.149e-05]`. A large salt move would mean the
edit did something other than what it claims.

**STEP 1.** Bit-identical. `nemo_dino_step1_gate.py`'s five rows and the
one-step snapshot's sha256 must both be unchanged.

## §4 — What this does NOT do

NEMO's TWO-STEP FLUX AVERAGE (`trasbc.f90:145-150`,
`zfact*(sbc_tsc_b + sbc_tsc)`) needs a carried `sbc_tsc_b` that legoESM does
not have. It is sized at 0.025 of the same row and stays in the ASKED table.
The solar term's own two-step average (`traqsr.f90:229-231`) is the same
class, sized at 4.36e-08 K by `kt1_qsr_gate.py --kt2-dir`, and likewise stays.

## §5 — Rule 12 disposition

The DINO applicator is shared. Every caller is fingerprinted before the edit
and the ones whose state carries no before level are byte-unchanged by
construction (§2). The certified twin is re-measured.

---

# POST HOC — and the first thing in it is a RETRACTION OF §3

**§3's "orthogonal" bound is MIS-DERIVED, so the "MISSED window" and the
"TRIPPED falsifier" it produced are WITHDRAWN.** An independent claim review
found it and the arithmetic was re-checked here rather than taken on trust:

If the correction **b** were orthogonal to the standing residual **a**, the
result would be `|a - b| = sqrt(a^2 + b^2) = 8.49e-06 K` — an INCREASE.
`sqrt(a^2 - b^2)` is a different case entirely (**b** a component *of* **a**).
From magnitudes alone the admissible range is

    [ | |a| - |b| | , |a| + |b| ] = [ 2.386e-07 , 1.2006e-05 ] K

and the measured `3.1059e-06 K` sits **inside** it. So the preregistered
window `[2.0e-07, 1.9e-06]` excluded ~85% of the legitimate range, and the
falsifier derived from it cannot refute anything. **The 0.961 sizing is NOT
refuted.**

| | value |
|---|---|
| before, kt=2 T state residual | `6.1225e-06 K` |
| the statement's size, off NEMO's own restarts | `5.8839e-06 K` |
| after | `3.1059e-06 K` (0.5073x) |
| admissible from magnitudes alone | `[2.386e-07, 1.2006e-05]` — measured is INSIDE |
| implied alignment `cos(theta)` | **0.8669** (theta = 29.9 deg) |

The alignment cosine is COMPUTED FROM the measurement, not predicted by it:
`cos(theta) = (a^2 + b^2 - m^2) / (2ab)`. It says the landed statement removes
a component that is ~87% aligned with the residual it targets — a sizing right
in magnitude and about 30 degrees off in direction. No mechanism is claimed.

**ALSO RETRACTED: the 4/9 story.** Commit `7799bd39f976` explained the 0.507x
as `0.961 x ~0.5`, citing the leap-frog/Asselin retention of 4/9 documented at
`dino.py:275-292`. Two things are wrong with it and the review found both:
that 4/9 is the retention of the **`applied_now`** placement, while this card
sets `surface_tendency_placement="leapfrog_rhs"` (`dino.py:1704`) whose whole
purpose is NOT being cancelled; and the commit multiplied by ~0.5, not by
4/9 = 0.444, which would have given 0.427 against a measured 0.493. **There is
no retained mechanism claim here.** The alignment cosine above replaces it and
requires none.

**WHAT STILL STANDS, unchanged.** The salt row did not move
(`1.1380e-05 psu`, preregistered `[1.127e-05, 1.149e-05]`) — HIT. Step 1 is
bit-identical under this edit, discriminated by S/u/v/eta being byte-identical
while only T moved (by 1 ulp, which is the qsr commit's).
