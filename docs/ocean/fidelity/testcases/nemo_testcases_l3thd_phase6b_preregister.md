# Lane 3b SI3 thermodynamics Phase 6b corrected DH-owner preregistration

Tracker: `climate-federation/legoESM#1699`

Status: **PREREGISTERED after H1 in the Phase-6 preregistration was refuted by
the registered operands, and before changing legoESM's snow-sublimation
sequence or running the corrected arm/year gate.**

## Refutation that fixes the hypothesis boundary

The copy-only operand run shows that kt5734 first differs before `snw_ent`.
After NEMO's sequential sublimation, `h_s_1d` is exactly zero but
`zh_s(3)=2.117582368135751e-22 m`; the other three segments are zero.  The
legacy legoESM mass-space removal leaves all four segments exactly zero.
Therefore Phase-6 H1's proposed first differing *remap-threshold branch* is
**REFUTED**.  The NEMO scalar `snw_ent` replay nevertheless reproduces all
three dumped `777833.3635432672 J m-3` values at zero ulp, so the remap is a
required second operation, not the first owner.

## Corrected hypothesis — capped thickness-space sublimation plus remap

NEMO computes one capped thickness increment
`MAX(-evap/rhos*rDt_ice,-h_s_1d)` and a remaining mass diagnostic, then visits
segments 0 through `nlay_s` with `MAX(-zh_s,zdeltah)` and updates the residual
as `MIN(zdeltah-zdum,0)` (`icethd_dh.F90:179-202`).  It deliberately does not
clear `ze_s` when `zh_s` reaches zero.  After later melt/flood branches it
passes the residual segment state to the unconditional cumulative `snw_ent`
remap, whose denominator is `MAX(zhnew,epsi20)` (`:494-507,535-613`).  Neither
operation has a NEMO namelist switch.

The legacy legoESM path instead holds a remaining mass, divides it by snow
density afresh for each segment, subtracts `remove*rho_snow`, and then calls a
generic overlap remap which hard-zeros new layers at `epsi10`
(`bitz_lipscomb.py:711-719,641-656` before this dispatch).

Hypothesis **H3-CAPPED-SUBLIMATION-REMAP**: the kt5734 owner is this exact
two-operation source identity.  Confirm only if:

1. a scalar replay of the NEMO sublimation assignments reproduces the dumped
   POST_SUBLIMATION `h_s_1d`, all four `zh_s`, `zdeltah`, and `zevap_rema` to at
   most two ulp;
2. a private sublimation-order-only arm reproduces the residual operand but is
   final-output inert while the legacy remap remains enabled;
3. a private remap-only arm is final-output inert on the legacy all-zero
   segments;
4. enabling both operations changes no earlier registered DH stage and reduces
   kt5734 POST_DH `e_s` error by at least 100-fold; and
5. a plant replacing the combined enabled output with the legacy output exits
   nonzero.

This is an explicitly preregistered interaction, not a claim that either arm
alone owns the final row.  Production may expose neither arm publicly: the
ORCA1-resolved SI3 identity must always execute both source operations.

Scaling uses the dumped residual segment multiplied by `1`, `1/2`, and `1/4`.
Record NEMO cumulative-remap output, legacy output, and error at each scale
before naming ownership.  The expected discriminator is linear preservation
under NEMO's `epsi20` denominator and exact zero under the legacy `epsi10`
guard; another scaling behavior refutes H3.

## Independent kt4242 row

The snow-remap operations cannot update `h_i` after flooding in the same DH
call.  The combined H3 arm must therefore leave kt4242 `POST_DH.h_i`
bit-identical to its pre-arm result.  A moved row refutes isolation.  If it is
unchanged, retain kt4242 as the first unresolved genuine injection unless a
separately registered written-order replay proves and repairs it.

## Choice register

- ASKED — first-divergence the melt-season DH snow-enthalpy row with WRITE-only
  operands, scaling, private hooks, a planted control, and an in-identity fix.
- ASKED — rerun exact-entry and continuous full-year results and report the
  requested growth and phenomenology rows.
- ASKED — classify residual noise honestly; kt4242 prevents a blanket noise
  classification unless it is independently resolved.
- UNASKED — a public snow-remap selector, a second ice identity, changes to the
  forcing/timestep/bar, coupled bulk-flux certification, GPU/MPI, or mutation
  or deletion of any shipped source/configuration or earlier run root.
