# NEMO testcase lane 2 GYRE — stage-3 zub/zvb reassociation preregistration

Date: 2026-09-03  
Parent register: stage-2 HPG/vorticity/advection and corrected Kaa are AT-BAR.

The next ordered rows are stage-3 `zFu/zFv/zFw`.  The fixed faithful baseline
measures normalized `8.736385768894321e-16`, `1.0966867098887257e-15`, and
`2.4980249777374853e-13`, respectively.  The first DEBT is therefore `zFv`,
not the later vertical recurrence.

NEMO retains the barotropic velocity `vv_b(Kmm)` as a separate prognostic and
forms `zvb = vn_adv*r1_hv(Kmm) - vv_b(Kmm)`, then
`zFv=e1v*e3v(Kmm)*(vv(Kmm)+zvb*vmask)`
(`src/OCE/stprk3_stg.F90:263-278`).  legoESM currently reconstructs the same
quantity by reducing `sum(e3v(Kmm)*v(Kmm))` inside the transport builder.  The
two are algebraically equivalent after the stage mean replacement but do not
preserve NEMO's stored-operand association.

## Fixed arm and outcomes

Add one private arm which hands the already-computed stage barotropic velocity
to the existing stage-transport kernel and evaluates the cited NEMO statements
in their source order.  The legacy arm restores the reduction-based expression;
there is one production implementation.

- CONFIRM causal ownership only if the source-associated arm moves at least
  99% of the faithful zFv residual and makes zFv AT-BAR; report scaling before
  the owner label.
- If zFv clears but zFw remains DEBT, advance to the `wzv(np_transport)`
  operand recurrence without claiming zub/zvb owns the later row.
- If zFv does not clear, label the arm NOT-SOLE-OWNER and retain zFv as the
  first boundary.

All comparisons are CPU/fp64, pointwise, and use the immutable normalized
`1e-15` bar.  A planted wet-face perturbation must fail by at least `0.9`.
