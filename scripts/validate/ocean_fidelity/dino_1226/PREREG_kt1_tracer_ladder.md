# PREREGISTRATION — NEMO's kt=1 TRACER program on the DINO card

Written BEFORE any measurement of this round, and before any code change.
Round owner: the tracer path. Step 1 currently stands at T 4.996e-6 K rms
(0.039% of NEMO's own first step), S 2.3e-7, eta 6.82e-7 m, u 1.08e-8,
v 6.46e-8 m/s. The geometry round moved eta/u/v by 300-700x and left T and S
untouched (1.007x / 1.010x), so T is now the largest step-1 fraction and the
tracer path owns it.

Everything below cites the COMPILED source DINO runs
(`cfgs/DINO/BLD/ppsrc/nemo/*.f90`), not the `src/` template.

---

## PART A — the alignment table: NEMO's kt=1 tracer program vs legoESM's

NEMO's `stp_MLF` at `kstp = nit000 = 1` with `ln_rstart = .false.`, so
`l_1st_euler` is forced `.TRUE.` (`RUN_FROMREST_KT1/ocean.output:312`) and
`rDt = rn_Dt = 2700 s` (one dt, not two).

| # | NEMO statement | time levels it reads | legoESM's statement | expected |
|---|---|---|---|---|
| T0 | `CALL zdf_phy( kstp, Nbb, Nnn, Nrhs )` `stpmlf.f90:193` — TKE closure, then `avt`/`avm` | Nbb (rn2b), Nnn | `_tke_step_entry_n2_bundle` + the TKE block inside `_step_impl`; carried on `state.tke`, `state.tke_avt`, `state.tke_avm`, `state.tke_dissl` (`ocean/state.py:555-564`) | MEASURE |
| T1 | `ts(ji,jj,jk,jn,Nrhs) = 0.` `stpmlf.f90:453-457` | — | legoESM has no `Krhs` accumulator; each tendency is built fresh in `_step_impl` | structural MATCH |
| T2 | `CALL tra_sbc( kstp, Nnn, ts, Nrhs )` `:459` | `Kmm = Nnn` | `apply_dino_lat_lon_surface_forcing(...)` returned as `external_tracer_rate` | MEASURE |
| T3 | `CALL tra_qsr( kstp, Nnn, ts, Nrhs )` `:466` (`ln_traqsr = T`) | `Kmm = Nnn` | the solar-penetration term of the same helper | MEASURE |
| T4 | `CALL tra_adv( kstp, Nbb, Nnn, Naa, ts, Nrhs )` `:484` (FCT) | Kbb tracer base, **Nnn transports** | FCT advection with `_fct_tracer_before=(T_before, S_before)` | see PART B, prediction 1 |
| T5 | `CALL tra_ldf( kstp, Nbb, Nnn, ts, Nrhs )` `:504` (`traldf_iso_lap`) | Kbb tracer | the isoneutral GM/Redi tendency, reading Nbb via `_ldf_state` | MEASURE |
| T6 | `CALL tra_zdf( kstp, Nbb, Nnn, Nrhs, ts, Naa )` `:507` | Kbb base, `e3t(Kaa)` / `e3t(Kmm)` thicknesses | `_apply_implicit_vertical_mixing` at `rdt` | MEASURE |
| T7 | `CALL tra_atf_qco( kstp, Nbb, Nnn, Naa, ts )` `:568` — the Asselin filter, **SKIPPED**: `IF( l_1st_euler )` takes the Euler branch (`traatf_qco.f90:152-159`), which writes only a zero `jptra_atf` trend | — | `_nemo_mlf_step`'s `_euler_start` skips the filter | MATCH |

---

## PART B — what the instrument is, and why the obvious one cannot be used

**The per-stage `stp_dump_*` files in `RUN_FROMREST_KT1` are NOT usable and
this round does not use them.** `RUN_FROMREST_KT1` is a **16-rank** run
(`layout.dat`: `jpnij 16`, `jpi 30`, `jpj 29`) and every `stp_dump_*.bin` /
`tke_dump_*.bin` / `sbc_dump_*.bin` carries NO rank suffix, so all 16 ranks
open the SAME filename. Measured, not argued:
`stp_dump_14_trasbc_kt00000001_tem.bin` is 243600 B = 30 x 29 x 35 x 8 —
ONE rank's haloed tile, not the 56 x 203 global field; `tke_dump_en.bin` is
187200 B = 26 x 25 x 36 x 8 — one rank's interior tile. Which rank survives
is a filename race, and the file may even be a MIXTURE of ranks' buffers.
The previous round recorded this defect for the barotropic streams only; it
is in fact EVERY un-suffixed `*_dump_*.bin` in this record, the whole tracer
ladder and the whole TKE chain included. Registered as a finding.

**The instrument this round uses instead: NEMO's own per-operator tracer
trend diagnostics, carried in the RESTART.** `ln_tra_trd = T`
(`ocean.output:1133`), so `trd_tra_mng` calls `trd_tra_iom`
(`trdtra.F90:282`), which calls `trddump_tra` (`trdtra.F90:348`), which
stores each operator's masked trend into `ttrd_store`/`strd_store`
(`trddump.F90:262-265`); `trddump.F90:336-362` writes all 15 of them into
the restart. They are proper per-rank netCDF variables, so
`rebuild_nemo_restart.py` stitches them to the full 56 x 203 domain with no
race at all.

**Rule 5 says a trend diagnostic is integrator bookkeeping until closure is
proved, so closure is the FIRST thing the probe does.** NEMO's own total is
`ttrd_tot` = `( T(Kaa)*(1+r3t(Kaa))/(1+r3t(Kmm)) - T(Kmm) ) / rn_Dt`
(`traatf_qco.f90:133-139`), and at kt=1 `ttrd_atf` is written as EXACTLY
zero (`traatf_qco.f90:152-159`). The probe requires

    ttrd_tot == ttrd_nsr + ttrd_qsr + ttrd_xad + ttrd_yad + ttrd_zad
                + ttrd_ldf + ttrd_zdf

to roundoff before any row is quoted, and REFUSES to print the table if it
does not. `ttrd_zdfp` and `ttrd_evd` are excluded from the sum by
construction: they are re-diagnoses computed inside `trdtra.F90:152-190`
from `avt`/`avt_evd` and `ts(Krhs)`, i.e. Rule 5's "non-additive
re-diagnostic of something already inside another bucket".

---

## PART C — the preregistered predictions, each with its falsifier

**Prediction 1 (structural, the sharpest one). The three advection trends
are EXACTLY zero at kt=1.** DINO starts from rest — `usr_def_istate` sets
u = v = 0 — and `tra_adv` reads the transports at `Kmm = Nnn`, the pre-step
velocity. So `ttrd_xad = ttrd_yad = ttrd_zad = ttrd_totad = 0` identically,
and ADVECTION CANNOT OWN ANY OF THE 4.996e-6 K.
*Falsifier:* any nonzero cell in those four arrays. If they are nonzero, my
reading of the time level is wrong and every row below is suspect.

**Prediction 2 (the named owner). `ttrd_zdf` — the implicit vertical mixing
solve, through its operand `avt` — carries the largest share of the T
residual.** From rest the only live tracer operators are the surface flux
(T2+T3), the isoneutral lateral diffusion (T5) and the vertical solve (T6);
of those, T5's coefficient and slopes are already measured near the bar
(`traldf_iso_lap tendency` 0.999987 / 0.999466 in `fidelity_bar_gate.py`)
while T6's operand comes from the TKE closure, whose FIRST-STEP seeding is
the least-certified thing in the chain — NEMO's own log says
"start from rest: set en to the background value" (`ocean.output:853`), a
statement legoESM's TKE state has never been scored against.
*Falsifier:* if `avt_k`, `avm_k`, `en` and `dissl` are all AT BAR (0 cells
unequal) after one step, the closure is exonerated; and if legoESM's
vertical-solve tendency then matches `ttrd_zdf` given NEMO's own `avt_k`,
prediction 2 is DEAD and the owner is the surface flux (T2/T3) or the
isoneutral term (T5).

**Prediction 3 (the alternative, named so it cannot be retrofitted).** If
prediction 2 dies, the owner is the SURFACE FLUX, and the specific suspect
is the heat capacity: `DINOConfig.c_p = 3991.86` (`dino.py:96`) truncates
NEMO's `rcp = 3991.86795711963` (`eosbn2.F90:1899`) at 6 significant figures,
relative error 1.9933e-06. The `nemo_faithful` path overrides it
(`dino.py:1437-1445`) — whether the OVERRIDE actually reaches every consumer
on this card is a statement to check by printing, not by reading
(Rule 10).
*Falsifier:* print the resolved constant the card runs with. If it is
NEMO's full-precision value everywhere, prediction 3 is dead before it is
measured.

**What this round will NOT claim.** A per-operator trend comparison scores
legoESM's tendency against NEMO's for that operator. It cannot by itself say
which one owns the FINAL state residual, because the operators compose. The
ranking claim is therefore explicitly limited to "which operator's own
tendency is furthest from NEMO's", and any statement about ownership of the
final T residual needs the composition, which is what the step-1 gate
already measures end to end.

---

## PART D — the bar

Zero cells unequal, per operator, per tracer. Anything else is DEBT and is
written DEBT. No row is cleared by an explanation of its residual.

---

# RESULT — measured, with the retractions first (Rule 11)

Run: `kt1_tracer_ladder.py`, fp64, `LEGOESM_NEMO_E3T=both`, the shipped card
(`nemo_faithful_dino_config(base=dino_config_for_recipe("nemo_dino_kamm_mlf"))`,
the same builder `run_dino.py` uses), one step from NEMO's own initial state
against `RUN_FROMREST_KT1`.

## RETRACTED

1. **PREDICTION 1 IS DEAD.** I predicted the advective tracer trends would be
   exactly zero at kt=1 because DINO starts from rest. They are not:
   `ttrd_totad` reaches 3.575e-07 K/s on all 342134 wet cells. The reading
   that killed it: `ln_ldfeiv = T` (`ocean.output:912`), so `traadv.f90:250`
   adds the eddy-induced (GM bolus) transport, which the initial
   stratification's slopes make nonzero from rest; and `stpmlf.f90:384` calls
   `wzv` a SECOND time after the momentum step, so the vertical velocity
   `tra_adv` consumes at `:484` is built from the post-barotropic sea level,
   not from the zero pre-step state. Row T4's "Nnn transports" was right for
   the horizontal component (`ttrd_xad` max 2.87e-23, i.e. zero) and wrong
   for the vertical.

2. **PREDICTION 2 IS DEAD.** I named the TKE closure's first-step seeding as
   the likely owner. It is EXONERATED: every one of `avt_k`, `avm_k`, `en`
   and `dissl` reproduces NEMO to ~1e-17 of NEMO's own rms — 0.1 ulp of
   fp64, thirteen orders of magnitude below the tendency residual. That is
   the preregistered falsifier, and it fired.

3. **PREDICTION 3 WAS DEAD BEFORE IT WAS MEASURED**, exactly as its own
   falsifier said it should be: the card resolves `c_p = 3991.86795711963`,
   NEMO's full-precision `rcp`, not `DINOConfig`'s truncated default. Printed,
   not read (Rule 10).

4. **A REVIEWER'S FINDING, REFUTED BY MEASUREMENT.** The claim review argued
   that `ttrd_zdf` is bit-identically `ttrd_tot` at kt=1 (because
   `trazdf.f90:94-95` saves `pts(...,Kaa)`, which it held to be zero on the
   first step), and that the whole instrument therefore collapses. Measured:
   `max|ttrd_zdf - ttrd_tot| = 5.4809e-06`, differing on all 342134 cells,
   9.26% of `max|ttrd_tot|`; and `ttrd_tot - ttrd_zdf` equals
   `ttrd_nsr + ttrd_qsr + ttrd_ldf + ttrd_totad` to 2.65e-18. The budget is
   real. Recorded here because a reviewer's finding is a hypothesis, not an
   instruction.

## CONFIRMED

**The instrument.** The budget closes: `ttrd_tot == nsr + qsr + totad + ldf +
zdf` to rel 4.47e-14, and `strd_tot == nsr + totad + ldf + zdf` to 4.85e-13.
The `xad/yad/zad` triple does NOT close (rel 6.07e-03 / 5.52e-02) — it is the
advective-form decomposition, not the additive one, and quoting it would have
produced a false finding. `ttrd_tot` reconstructed independently from
`tn`/`tb`/`sshn`/`sshb` agrees with the stored array to rel 1.42e-15.

**NEMO's own first step is almost entirely vertical mixing** (max|trend|,
K/s): zdf 5.92e-5, nsr 7.52e-6, qsr 4.04e-6, ldf 1.77e-6, totad 3.58e-7.

**The closure, at the end of kt=1** (0 cells unequal is the bar):

| field | cells unequal | max\|diff\| | frac of NEMO rms |
|---|---|---|---|
| `avt_k` | 23060 / 332214 | 1.30e-18 | 2.21e-17 |
| `avm_k` | 23924 / 332214 | 1.39e-17 | 6.37e-17 |
| `en`    | 32704 / 332214 | 2.17e-19 | 5.74e-17 |
| `dissl` | 15834 / 332214 | 8.67e-19 | 4.44e-17 |

DEBT by the exact bar, and the residual is fp64 last-bit. The TKE closure is
not the owner of anything at this scale.

**The total tracer tendency** (NEMO's own definition on both sides):
T rms 1.837e-09 K/s against NEMO's 4.764e-06 (frac 3.855e-04);
S rms 8.579e-11 against 8.205e-07 (frac 1.046e-04). Cross-check: 1.837e-09
K/s x 2700 s = 4.96e-06 K, which is the step-1 gate's own T residual — the
two instruments agree.

**THE NAMED OWNER, measured operator-to-operator through the model's own
path:** legoESM's isoneutral lateral diffusion tendency is
**0.996571 x NEMO's on temperature and 0.996696 x on salinity** — 0.34% and
0.33% low, captured during the same step by spying the module-level
`nemo_iso_lap_tracer_tendency_latlon_cgrid` and scored against `ttrd_ldf` /
`strd_ldf`. Its own residual rms is 8.67e-11 psu/s against a TOTAL salinity
residual of 8.58e-11 — i.e. **`tra_ldf` accounts for essentially ALL of the
salinity step-1 residual**. On temperature it is 5.01e-10 of 1.837e-09, about
a quarter by rms.

**Temperature's remaining explainer is the SOLAR term, and it is still only a
correlation.** Regressing the T residual on NEMO's own operators: qsr
explains 50.6% of its variance (corr -0.711, best-fit -2.56e-03), nsr 14.6%,
ldf 7.0%, zdf 0.2%, totad 0.02%; the operators are near-orthogonal (largest
pairwise |corr| 0.31, nsr vs qsr). The worst level by a factor of 30 is the
SURFACE, k=0 (residual/NEMO 4.21e-03 there against 1.24e-04 at the next
worst). Labelled PLAUSIBLE: a correlation is not a mechanism, and no
operator-to-operator capture exists yet for `tra_qsr`.

## NEXT OWNER, and the measurement that discriminates it

`tra_ldf`. It is confirmed 0.34% low, it owns salinity outright, and the
ratio is NOT the same constant on the two tracers (0.996571 vs 0.996696), so
it is structural rather than a single mis-set coefficient — the diffusivity
itself is already at the bar (`ldftra ahtu/ahtv` rows, ratio 1 - 3e-8). The
discriminating measurement is a stage-by-stage walk of `traldf_iso.f90`'s own
assembly against legoESM's, in NEMO's order: the `zmsku`/`zmskv` triad
normalisations (`:234-236`, `:267-269`), `zA13`/`zA23` (`:239-240`),
`zA31`/`zA32` (`:277-278`), and the `ah_wslp2 - akz` stabilising-correction
term (`:285-286`) which is identically zero only when `ln_traldf_msc = F` and
DINO has it `T` (`ocean.output:890`, `rn_slpmax = 1e-2`). Each of those
carries an independent operand that can be captured on the legoESM side the
same way stage 4 captures the total.

Second: an operator-to-operator capture for `tra_qsr`, to turn temperature's
51%-of-variance correlation into a number.
