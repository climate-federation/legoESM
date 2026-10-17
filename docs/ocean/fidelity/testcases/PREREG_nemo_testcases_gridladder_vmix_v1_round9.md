# PREREG — vertical-mixing lane V1 round 9 (vmix_r9)

Frozen before the landing and the kt 1..10 ladders. Record: `gridladder_vmix_rounds/round1/oracle_v1_r1`
(admitted round 3). Decisions 129 = 1 and 130 = 1 in force. Codex review: paused (quota).

Already measured (evidence `gridladder_vmix_rounds/round9/`, done BEFORE this file):
- kt-2 surface u: 5,048 unequal cells, normalised 1.70e-15 at (102,43) (the ladder prints the
  absolute 2.22e-15; scale max|u| = 1.31 m/s).
- Substituting NEMO's kt-2 `avm_k` for the closure's viscosity (`probe_avm_seam.py`, identity
  control bit-identical) makes kt-2 u and v EXACT at every level; NEMO's `avt_k` makes T and S
  exact. So `dyn_zdf` (`dynzdf.f90:67`, surface stress `:338`, ice top drag) and `tra_zdf` are
  bit-for-bit; the u debt is inherited from TKE's viscosity (30,324 unequal cells).
- kt 2 is the first step with nonzero sea level at the step entry (from rest), and every unequal
  kt-2 `en` cell has nonzero shear (round 8). The step-entry shear's private `r3u`/`r3v` divides by
  `e1e2u`; NEMO multiplies by `r1_e1e2u = 1/e1e2u` (`dom_qco_r3c`, `domqco.f90:188-234`;
  `domhgr.f90:170`), as the certified shared stage-face helper already does.

Landing (V1-selected only): the step-entry shear's `r3u`/`r3v` multiply by the reciprocal face area.

- **P1.** kt-2 `en`, `avm_k`, `avt_k`, `dissl`: unequal cells -> 0 (from 47,358 / 30,324 / 28,274 /
  24,058). Falsifier: any unequal cell.
- **P2.** kt-2 u, v, T, S exact (follows from P1 and the substitution). Falsifier: any unequal cell.
- **P3.** kt 1 stays bit-identical on every field under both labels. Falsifier: any unequal cell.
- **P4.** Flag-off `make_jaxpr` of one step of the full ORCA2 and DINO cards identical to base.
- **P5.** V1 certifies (every field at the bar kt 1..10, both labels). PLAUSIBLE only; falsifier:
  any DEBT row.
