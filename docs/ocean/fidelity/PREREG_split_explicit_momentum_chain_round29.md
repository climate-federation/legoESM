# Preregistration: post-split div_hor and dom_qco_r3c, round 29

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Frozen before measuring rows 2 or 3.
The clean round-28 production replay is bound at SHA-256
`1b4e83ec06416711cfc2b866c878cccb2a23988e8953c9b440a9e94c22bc2be9`:
row 1.3 SSH/U/V and all five row-1.4 final outputs are AT BAR. The literal
coefficient receipt is bound at SHA-256
`0e8757a2c10cd99cbb8394f639e15beb84c05b244f4eeb349718785763e9106e`.

## Row 2: second div_hor

At active `stpmlf.F90:349-376`, NEMO calls `div_hor(kstp,Nbb,Nnn)` after
`dyn_spg_ts`; the executed MLF kernel is `divhor.F90:172-181`. Existing
`seq_dump_hdiv_nnn_kt00005761.bin` is the full-halo, zero-initialized dump
written immediately after that call. No new writer or run is needed.

The primary candidate is the current production `divergence_cgrid` applied to
the bridged Nnn restart U/V at the matched day-180 state, with native bridge
metrics and masks. Score the active 3-D T-mask population using the canonical
campaign tuple (correlation, mean-absolute ratio, normalized RMS and
maximum/NEMO-RMS) and `fidelity_bar_gate.classify(...,
name="ssh_nxt / div_hor")`. This row's already-reviewed mechanism ceiling is
a legal terminal state: AT BAR or CEILING releases row 3; DEBT, NEAR-CLASS,
or UNMEASURED stops the walk. Report, but do not use as a gate, the
source-ordered literal transcription of `divhor.F90:172-181`; it localizes any
production residual to transport/metric association rather than inventing a
new physical term.

Required controls: identity must classify AT BAR before the row name is
applied; a four-nextafter plant at the maximum-magnitude wet point must cross
the strict unconditioned 1e-15 pointwise bar; the zero-offset alignment must
beat all eight neighbouring shifts; tracked tree, CPU/fp64, lane, shapes,
populations, and every input/source hash must be recorded.

## Row 3: second dom_qco_r3c (conditional)

Only a released row 2 permits row 3. At `stpmlf.F90:378-394`, NEMO calls
`dom_qco_r3c(ssh(Naa),r3t(Naa),r3u(Naa),r3v(Naa),r3f)`; active formulas are
`domqco.F90:153-185`. Existing `seq_dump_r3{t,u,v}_aaa_kt00005761.bin` and
`seq_dump_r3f_kt00005761.bin` are the outputs.

Use the row-1.4 production `pssh_final` captured by the bound replay, not a
restart or oracle-substituted SSH. Evaluate the literal formulas in source
association: T multiply; U/V `0.5*(area*ssh + neighbour_area*neighbour_ssh)`
then `*r1_h0*r1_area`; F the two parenthesized pairs then the same ordered
post factors. Score T and each face/F field at the strict POINTWISE 1e-15
class bar and through the registered `dom_qco_r3c r3t` / `r3u/r3v` gate rows;
`r3f` defaults to the strict bar. All four must be AT BAR to release row 4.
The same identity, four-nextafter, zero-shift, wall/seam, and actual-binding
perturbation controls apply.

Rows 4--6, the free-surface filter, momentum RHS, and tracer tail remain
ordered-blocked until this scorer releases them. Existing dumps are sufficient
for this round; no SLOT block, GPU, NEMO process, or MPI process is authorized.
