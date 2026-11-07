# Preregistration: GYRE kt=2 model-path accumulators and barotropic memory, round 48

Date: 2026-09-11. Frozen at legoESM `f826f1e6c8f9` before any round-48
measurement, oracle acquisition, or production edit. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round48/`. CPU/fp64 only.

## 1. kt=2 model-path accumulator score

The existing admitted round-46 record will be reused; no NEMO run is needed.
At each kt=2 stage, the gate will JIT legoESM's ordinary GYRE card and momentum
operator routes on the recorded NEMO stage-entry state. It will then accumulate
those production operator arrays in the compiled statement order, with an
optimization barrier after each boundary, and compare the complete accumulator
to the recorded `after_*` field. This is not an isolated-tendency subtraction.
The compiled HPG assignment overwrites Krhs because HPG is first
(`dynhpg.f90:400-427`), so the recorded pre-HPG scratch content is deliberately
not added.

The order is stage 1 HPG, LDF, VOR, ADV (`stp2d.f90:141-176`), stages 2 and 3
HPG, VOR, ADV (`stprk3_stg.f90:451-480`), with stage-3 LDF following the RK
stage update. LDF consumes `pu/pv(Kbb)`, not Kmm
(`dynldf_lev.f90:65-80,123-130`); the gate will therefore construct its LDF
operand from recorded `u_Kbb/v_Kbb`. For kt=2 stage 1, Kbb is the developed
kt=1 end state and is bit-identical to that stage's Kmm record. It is not the
stage-2 or stage-3 Kmm velocity.

Frozen prediction: the first non-bit accumulator is **stage 1 post-LDF**, and
**stage 2 post-VOR** and **stage 3 post-VOR**. The immediate predecessor
(post-HPG in all three stages) must be bit-identical. CONFIRM requires exactly
that first-boundary table. REFUTE if any post-HPG row differs, if stage-1
post-LDF is exact, if stage-2/3 post-VOR is exact, or if any earlier executing
boundary already differs. A failed prediction will be printed as REFUTED, not
reinterpreted.

## 2. Cross-step barotropic memory acquisition

The compiled RK3 card seeds `sshn_e/un_e/vn_e` from Kmm
(`dynspg_ts.f90:352-356`), resets Kaa and `un_adv/vn_adv` every call
(`:373-378`), consumes the six persistent b/bb histories in AB3 extrapolation
(`:456-489`) and SSH back interpolation (`:593-600`), rotates them every
substep (`:749-761`), and restart-writes all six (`:970-980`). The compiled
card contains no `ub2_b/vb2_b` read or write; those dead non-RK3 fields are not
admitted as inputs. legoESM already carries the six live histories as the
deviation-form `bt_hist` state (`state.py:648-681`) and reconstructs their raw
values at the next window (`barotropic_latlon_cgrid.py:2053-2084`). No new
state is preregistered.

A WRITE-only instrument will record two self-describing fp64 streams: kt=1
after barotropic finalization and kt=2 immediately after current-state seeding
and accumulator reset. Each carries the six raw histories, current external
mode, `un_adv/vn_adv`, and Kmm/Kaa barotropic/SSH arrays. Admission requires
bit-identical six-history payloads across the boundary, exact kt=2 current
seeds, exact zeros in reset accumulators, physical EOF, a clean producer
commit stamp, and byte/consumed-field identity of inherited outputs. Header,
truncation, boundary, seed, reset, twin, and stamp plants must exit nonzero.
The instrument only writes; it must not assign a model field.

No oracle executable will be built or run by the agent. The operator resumes
only after running the committed acquisition script.

## Decisions

| status | item | disposition |
|---|---|---|
| ASKED | measure model-path boundaries and prepare the kt=1→2 memory record | this round |
| UNASKED | new carried state, configuration, threshold, or physics choice | none |
