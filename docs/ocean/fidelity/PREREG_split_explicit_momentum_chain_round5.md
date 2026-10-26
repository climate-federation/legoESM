# PREREGISTRATION — split-explicit momentum chain round 5

Date: 2026-08-29.  Frozen before calculating the pressure-bearing corners of
the V slow-forcing composition.

## Admission and question

Round 4's clean receipt at `6d435fe1f53` and artifact SHA-256
`4539d9a7ec4e2f20a41226a5707268ebcd268f97f03f0e2d59901906775fc154`
fully disposes U as the lateral-friction × vorticity × pre-loop-Coriolis
composition.  The same V trio is UNRESOLVED (`removal=0.091570022652`,
`Q=0.825245023744`).  Hydrostatic pressure gradient is the only
noncandidate V term above the frozen `0.10*RMS(r0)` bound
(`gain=0.910571701895`).

The next source-ordered discriminator is therefore the complete four-axis
V matrix, not a pressure-only fix.  Axes retain the round-4 bit order and add
pressure last:

`(L lateral friction, V total EEN vorticity, C pre-loop Coriolis removal,
P hydrostatic pressure gradient)`.

## Active NEMO rows and time levels

| Axis | Active NEMO source/time level |
|---|---|
| `L` | Kbb/BEFORE lateral increment, `cfgs/DINO/MY_SRC/dynldf.F90:73-115`, called at `stpmlf.F90:319-322` |
| `V` | Kmm/NOW total EEN increment, `cfgs/DINO/MY_SRC/dynvor.F90:143-194`, called at `stpmlf.F90:315-318` |
| `C` | Kmm/NOW 2-D Coriolis removal, `cfgs/DINO/MY_SRC/dynspg_ts.F90:358-370` |
| `P` | Kmm/NOW SCO hydrostatic pressure-gradient increment, `cfgs/DINO/MY_SRC/dynhpg.F90:348-413`, called at `stpmlf.F90:324-328` |

Every axis replaces the faithful legoESM term with its existing NEMO oracle
term on the registered 9,868 wet V faces.  No term is reevaluated at a
different state.  `C` retains the round-2 exact-ledger inference and all its
controls.

## Matrix, locked corners, and algebra

For each of the 16 subsets `S`, use the round-4 residual definition
`r_S=r0-sum(e_i for i in S)`, `Q_S=mean(r_S^2)/mean(r0^2)`, and
`B_S=1-Q_S`.  All eight `P=0` corners must reproduce the accepted round-4 V
artifact within `1e-12` in normalized RMS removal, and the pressure singleton
must reproduce the committed round-2 V attribution within `1e-12`, before
any other pressure-bearing corner is admitted.

Compute the unique Möbius coefficient for every nonempty subset:

`m(S) = B(S) - sum(m(T) for nonempty proper subsets T of S)`.

Compute the order-independent Shapley allocation
`phi_i = sum(m(S)/|S| for S containing i)`.  Require both
`sum_S m(S)=B(full)` and `sum_i phi_i=B(full)` within `1e-12`.
The field-level Möbius contrast for every interaction order 2--4 must remain
below `1e-12*RMS(r0)`; energy interactions do not imply nonlinear physics.

Retain all round-4 bars without change:

- a correction CONFIRMS iff correlation is at least `0.99`, gain is in
  `[0.90,1.10]`, and RMS removal is at least `0.90`;
- the quartet is `CONFIRMED_COMPOSITE_OWNER` only if the full arm CONFIRMS
  and `Q_full<=0.01`;
- under a confirming quartet, `phi>=0.01` is
  `OWNED_POSITIVE_CONTRIBUTOR`, `phi<=-0.01` is `OWNED_CANCELLER`, and
  smaller magnitude is `BOUNDED_BELOW_OWNERSHIP_REMAINDER`;
- every Möbius interaction with absolute value at least `0.01` is MATERIAL;
- every noncandidate named term and the explicit faithful
  `assembly_remainder` must have gain `<=0.10` or the row remains open.

Unconfirmed-group Shapley values are descriptive only and must be stamped
`DESCRIPTIVE_UNADMITTED_GROUP_UNRESOLVED`.

## Continuation rule

If V confirms and every noncandidate is bounded, row 1.1 is fully disposed:
U by the accepted round-4 trio and V by this quartet.  A subsequent committed
preregistration may then hold row 1.1 at the exact all-eight-term NEMO forcing
endpoint and measure rows 1.2--6 in execution order from existing dumps.  If
the quartet fails, STOP on its first unbounded remainder; do not promote a
pressure-only edit or downstream row.

## Controls and provenance

The probe must run clean, CPU/fp64, `DINO_1226_LANE=d180`, with strict JSON,
exact masks/populations, all round-2 controls, the accepted round-4 artifact
hash, and unchanged dump hashes.  Plants must reject a changed locked corner,
breach a field interaction, exercise positive/canceller/bounded/unadmitted
labels, reach group CONFIRM and REFUTE, and reject a `0.11*RMS(r0)` remainder.

No new NEMO writer, run, build, GPU, MPI process, or SLOT allocation is
required.  Any future writer remains subject to the held-SLOT protocol.
