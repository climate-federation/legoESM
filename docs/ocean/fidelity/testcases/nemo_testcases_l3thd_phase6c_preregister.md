# Lane 3b SI3 Phase 6c exact-entry thickness bridge correction

Tracker: `climate-federation/legoESM#1699`

Status: **PREREGISTERED after the first Phase-6 year run exposed the bridge
defect and before changing or rerunning the exact-entry gate.**

## Finding

The exact-entry gate reconstructed `h_i` and `h_s` from the global ENTRY
`v_i/a_i` and `v_s/a_i`.  NEMO instead carries separate equivalent-thickness
arrays into `ice_thd_1d2d`; they can differ from a reconstructed quotient by
one binary64 ulp.  At kt5734 that one-ulp difference reverses which carrier
retains the final `2.117582368135751e-22 m` subtraction residual.  Applying the
now-confirmed source operation to the reconstructed operand therefore creates
rows that are not the one-step injection from NEMO's exact 1-D entry.

The registered POST_ZDF frame already contains NEMO's exact `h_i_1d` and
`h_s_1d`.  The active ZDF call reads but does not assign either thickness; the
driver calls it and dumps the frame immediately afterward
(`icethd.F90:147-161`; `icethd_zdf_bl99.F90:159-230,433-590`).  Those values
are consequently valid **post-glo2eqv/pre-ZDF 1-D thickness operands** as well
as POST_ZDF thicknesses.  The two-step Phase-6 DH stream independently records
the same `h_s_1d` at DH initialization.

## Hypothesis and gate

Replace only the exact-entry sweep/arm bridge's reconstructed thicknesses with
the registered unchanged 1-D thickness operands.  Do not inject oracle values
into the continuous trajectory.  Confirm **HARNESS CORRECTION** only if:

1. the kt4242 and kt5734 bridged values equal the DH INITIALIZED operands
   bit-for-bit;
2. the combined DH arm still improves kt5734 POST_DH `e_s` by at least 100-fold
   and its NEMO scalar replays remain within two ulp;
3. the new zero-oracle artifacts at kt5551/kt5997 disappear rather than being
   suppressed by a reporting filter; and
4. the continuous trajectory and phenomenology are unchanged by this
   exact-entry-only correction.

A plant restoring quotient reconstruction at kt5734 must make the exact-entry
bridge check exit nonzero.  If any continuous value changes, or the registered
thickness differs across ZDF, refute the correction and retain the first run as
the result.

## Choice register

- ASKED — consume the oracle's exact exchange/state operands and register time
  levels; correct a non-exact bridge discovered by the operand discriminator.
- ASKED — rerun and report exact-entry, continuous growth, and phenomenology
  honestly.
- UNASKED — mid-trajectory oracle injection, altered physics/forcing/bar,
  public selectors, GPU/MPI, or changes to shipped NEMO and earlier run roots.
