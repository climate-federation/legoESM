# Lane 3b SI3 thermodynamics Phase 6 DH-owner preregistration

Tracker: `climate-federation/legoESM#1699`

Status: **PREREGISTERED before adding the DH operand stream, evaluating either
private arm, or changing the production snow remap.**  The existing registered
boundary frames motivate the tests but do not resolve the operands within
`ice_thd_dh`.

## Fixed protocol

Keep the reviewed Phase-5 oracle root, ERA5 forcing, one-hour step, 8,760-step
window, CPU backend, fp64 policy, and `1e-15` boundary bar.  The additive oracle
run may only use a copy of the prior NEMO source/configuration and WRITE-only
instrumentation.  It must not alter the shipped NEMO tree or any earlier run
root.  Retain full registered boundary state for kt4241--4243 and kt5733--5735.

The operand stream is restricted to kt4242 and kt5734 and records, in NEMO
assignment order, `h_s_1d`, `zh_s(0:nlay_s)`, `ze_s(0:nlay_s)`, `zdeltah`,
`zevap_rema`, `zq_top`, `dh_snowice`, the `snw_ent` new-layer thickness,
cumulative thickness/enthalpy arrays, and remapped `e_s_1d`.  It also records
the ice/snow thickness operands surrounding flooding at kt4242.  Each record is
current-step, current-category, after the named DH branch and before the next
named branch; the committed reader must fail closed on magic, version, step,
stage, payload size, and order.

## Rule 0 hypotheses

NEMO initializes four snow segments at `icethd_dh.F90:139-145,166-177`, then
applies sublimation/deposition sequentially without clearing a segment's
enthalpy when its thickness reaches zero (`:179-202`).  Surface heat, when
available, melts snow before ice and likewise retains the enthalpy carrier
(`:204-225`).  The remaining heat then enters the ice-surface loop
(`:231-315`).  Complete ice loss removes snow at `:426-439`; otherwise flooding
converts snow from the base at `:441-485`.  `snw_ent` remaps the resulting
segments at `:494-507,535-613`: it uses `SUM(ph_old)/nlay_s`, sets the last
cumulative content exactly, and divides each nonnegative content increment by
`MAX(zhnew,epsi20)`.  There is no namelist switch around this remap.

NEMO `ice_var_zapsmall` does **not** test `v_s` against `epsi20`.  It clears snow
enthalpy only when `MIN(a_i,v_i,h_i)<epsi10` (`icevar.F90:601-706`).  The
`v_s>epsi20` predicate instead selects snow-temperature reconstruction versus
`rt0` in `ice_var_glo2eqv` (`icevar.F90:404-416`).

The current legoESM DH path executes the corresponding precipitation,
sublimation, melt, flooding, and remap sequence in
`packages/ice/legoesm/ice/bitz_lipscomb.py:659-926`.  Its shared
`_piecewise_remap` currently hard-zeros all new cells when the new layer
thickness is not above `epsi10` (`:641-656`), unlike `snw_ent`.

### H1 — kt5734 small-snow remap owner

The registered POST_ZDF state has small positive snow and the POST_DH state has
reported zero snow thickness; NEMO retains `777833.3635432672 J m-3` in each
snow-enthalpy layer while legoESM returns zero.  Hypothesis: sequential
sublimation leaves representable segment/cumulative operands that NEMO's
`epsi20`-denominator remap preserves, while legoESM's `epsi10` hard-zero loses
them.

Confirm only if the operand stream plus a scalar NEMO-order replay reproduces
the dumped POST_DH `e_s` to at most two ulp, the first different branch is the
remap threshold (`zhnew > epsi10` in legoESM versus unconditional NEMO remap),
and a private `_nemo_snow_remap=True/False` arm changes no upstream operand and
improves kt5734 POST_DH `e_s` by at least 100-fold.  Refute if the replay fails,
an earlier branch condition differs, or the arm does not move the row.

Scaling is measured before ownership: replay the fixed dumped segment state at
`1`, `1/2`, and `1/4` residual thickness scales and report NEMO/legacy remap
output and error.  The control substitutes the legacy arm at the registered
kt5734 row and must exit nonzero.

### H2 — kt4242 thickness injection is independent unless measured otherwise

The first Phase-5 injection above `1e-12` is kt4242 `POST_DH.h_i`, not `e_s`.
Because snow remapping occurs after flooding and cannot update `h_i` in the
same call, preregister it as a separate DH operand/order hypothesis.  Confirm
an association owner only if the dumped NEMO operands and a written-order
scalar replay close the `h_i` gap to at most two ulp, and a one-variable private
hook moves kt4242 while the snow-remap hook does not.  If neither discriminator
does so, retain kt4242 as unresolved debt; do not merge it into H1 by narrative.

## Required after-test reporting

Run the full exact-entry and continuous year before/after.  Retain exact-entry
over-bar count, first genuine injection step, largest genuine row, normalized
growth rows at steps 1, 10, 100, 1000, 3000, 5000, and 8760 for `t_su`, `e_i`,
`h_i`, and `h_s`, plus all six phenomenology rows.  Classify the residual as
threshold-amplified `<=2e-15` noise only if every same-step exact-entry
injection at every continuous jump satisfies that bound; otherwise name the
counterexample and leave DEBT.

## Choice register

- ASKED — quote the active NEMO snow-melt/sublimation/flooding/remap/correction
  branches and the legoESM executing lines.
- ASKED — retain boundary states at kt4241--4243 and kt5733--5735 and add a
  WRITE-only, copy-only DH operand discriminator where required.
- ASKED — measure scaling before ownership, use private one-variable arms, and
  fix only inside the selected SI3 identity when NEMO has no switch.
- ASKED — rerun the CPU/fp64 year gate and report exact-entry, growth, and six
  phenomenology results before/after.
- ASKED — explicit-path recovery-git commits and bundle; no push and no
  shipped-NEMO changes.
- UNASKED — new public selectors, alternative ice identities, coupled `sbcblk`
  certification, altered forcing/timestep/bar, GPU/MPI, or deletion/modification
  of any shipped source, configuration, or earlier run root.
