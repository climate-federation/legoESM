# ORCA2 round 133 — rung-0 step-36 FCT boundary

Date: 2026-10-04. Base `a21a3097b`. Final measurement commit
`2ac4b420b`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round133`.
Every scientific number is **independent**: legoESM starts from the hierarchy
rung-0 card's climatological T/S, zero velocity, and zero sea surface.
Decision 52's recorded-entry bridge is not used.

## Verdict

**HELD.** The merged baseline repeats round 132 exactly: it is finite through
step 35 and step 36 first returns non-finite temperature at zero-based
`(j,i,k)=(86,159,0)`. All exposed stage-1 face thickness, transport-average,
corrected-velocity, and final metric-transport operands are finite. Every
active tracer cell remains finite through the complete stage-2 tracer. The
first measured active-cell non-finite boundary is the stage-3 tracer-advection
content: 132 temperature and 134 salinity cells.

NEMO runs stages 1, 2, and 3 in that order at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:211-227`. At the named boundary the
compiled stage program calls `tra_adv` before the surface source at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:633-645`; the resolved ORCA2
dispatch takes `tra_adv_fct` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:525-535`. This names the first
measured executed call, not yet the first arithmetic statement inside FCT.
That internal source-ordered split is the next walk.

No configuration, forcing, carried state, stabilizer, sea-ice selector,
`unmeasured_features` entry, or `packages/` file changed. The shipped ORCA2
card and its ice tuple remain untouched.

## Frozen predictions

| ID | Result | Measurement |
|---|---|---|
| R133-P1 | **CONFIRMED** | finite through step 35; step-36 returned T first fails at `(86,159,0)`, twice bit-identically |
| R133-P2 | **REFUTED** | corrected velocity is finite; no exposed stage-1 transport operand is non-finite |
| R133-P3 | **REFUTED** | final stage-1 `zFu`, `zFv`, and `zFw` are all finite |
| R133-P4 | **CONFIRMED** | every write-only operand exposure preserves every ordinary returned-state field not used as its diagnostic slot |
| R133-P5 | **CONFIRMED** | measurement-only; no model or selector diff |
| R133-P6 | **CONFIRMED** | active stage-1 and stage-2 tracer boundaries are finite; stage-3 advection is first non-finite |
| R133-P7 | **CONFIRMED** | every measured active boundary from stage-3 advection through the returned state remains non-finite |

The refuted predictions remain part of the record. They moved the walk
downstream without changing the model.

## Stage-1 operand walk

At the step-35 entry, all five prognostic fields are finite. Maximum absolute
values are T `1.3914296032613976e5 K`, S `1.4709640491006407e4 g/kg`, u
`1.0200845911680202e10 m/s`, v `1.125719406976199e8 m/s`, and sea surface
`1.6596105440214956e1 m`. The state is already catastrophically amplified;
the admitted NEMO rung-0 run remains finite through step 240.

| source-ordered exposure | non-finite values | maximum absolute value |
|---|---:|---:|
| live U/V face thickness | 0 | `1.0000980586386111e3 m` |
| broadcast transport average | 0 | `1.9151821465550966e24 m3/s` |
| corrected stage velocity | 0 | `2.196681573349405e21 m/s` |
| metric `zFu/zFv/zFw` transport | 0 | `3.605854073039252e29` |

The compiled product statements are
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:282-283`. Their finiteness
refutes a stage-1 transport owner on the merged tree. The first-pass JSON and
all four synthetic plants are retained; both ordinary step-36 executions are
bit-identical in T, S, u, v, and sea surface, including NaN payload bits.

## Downstream active-cell walk

The first downstream run initially classified stage-1 advection as non-finite
because it counted 368,648 dry-cell NaNs in each scratch tracer. The complete
stage-1 tracer immediately cleared them. That classification is
**RETRACTED**: it was an instrument support bug, not a model owner. The gate
now scores the card's 430,552 active T cells and records dry cells separately.
Its corrected result is:

| boundary | active non-finite T / S | finite max abs T / S |
|---|---:|---:|
| after stage-1 advection | 0 / 0 | `6.421970430738307e22` / `1.9972316040413867e22` |
| after stage-1 surface source | 0 / 0 | `6.421970430738307e22` / `1.9972316040413867e22` |
| complete stage-1 tracer | 0 / 0 | `3.466868288625991e26` / `1.0781953899928912e26` |
| complete stage-2 tracer | 0 / 0 | `9.17083217605012e48` / `3.0075613372242536e48` |
| stage-3 advection content | **132 / 134** | `4.114566699799138e261` / `2.8613689353434972e259` |
| complete pre-implicit content | 132 / 134 | same as stage-3 advection |
| pre-implicit concentration | 138 / 139 | `4.4139591672965e263` / `5.948418806235251e261` |
| returned tracer | 210 / 187 | `1.7059666351441653e263` / `1.467666780266163e191` |

The first active non-finite is T at `(86,159,0)` in stage-3 advection. The
vertical solve is downstream propagation, not the first measured owner.

## Controls, tests, and review

Both operand plants and both downstream plants exit 2 with
`STATUS PLANT-FIRED`: one moves an earlier-boundary census and one breaks
write-only passivity in each gate. The focused round-130/131/133 battery passes
`24/24`. The separate read-only Codex review did not reach the diff:
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

Citation and broad-battery results are recorded in the final validation
commit after those controls run.

## OPEN

1. Split stage-3 FCT in compiled source order: effective transport, low-order
   update, antidiffusive flux, limiter, and final RHS accumulation. Name the
   first active-cell non-finite arithmetic statement at `(86,159,0)`.
2. The independent rung-0 month remains unmeasured at step 240. Do not merge
   the parked hierarchy decks or climb to rung 1 until rung 0 is finite.
3. Round 129's complete barotropic association arm remains held; its 31.363
   salinity exposure is not part of this tree.
4. The rung-7 first non-bit statement remains outside the hierarchy order and
   is unchanged by this measurement-only round.

## UNVERIFIED

- The first arithmetic operation inside stage-3 FCT that overflows.
- Every independent rung-0 terminal month RMS/max value and ranking.
- The independent-month effect of round 129's held source unit.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
