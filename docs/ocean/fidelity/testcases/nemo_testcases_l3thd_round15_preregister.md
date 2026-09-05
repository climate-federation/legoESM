# SI3 lane 3b round 15 preregistration — one-layer ZDF/end-step owner hunt

Date: 2026-09-04  
Tracker: `climate-federation/legoESM#1699`  
Parent: `dad9dd304872649d66044d1db45e3e68b1e63f45`  
State: **PREREGISTERED; UNMEASURED**

## Frozen observation and ordered stop

Round 14 left every registered step-1 boundary bit-identical but measured the
first continuous debt at `kt=2 PRE_SSM.u`, absolute/normalized
`1.0844585561836217e-4`.  No later register is used to infer the owner.  This
round inserts WRITE-only registers into the single unregistered interval and
stops at its first non-bit field in NEMO execution order.

The accepted scalar-math R14F config and data remain immutable.  A fresh
config-local copy will write only `kt=1`; its output will be calibrated by
confirming that every pre-existing physical stream, `ocean.output` numeric
lines, and restart remain byte-identical to R14F.  Every header count is
derived from the write list with `SIZE` and validated by the reader.

## Rule 0: executing source order

The active `key_RK3`, `key_qco`, `ln_drgimp`, `ln_dynspg_ts`, and
`ln_drgice_imp` path is:

1. `dynzdf.F90:119-140` constructs the thickness-weighted stage-3 momentum
   RHS at `Kaa`.
2. `dynzdf.F90:148-171` removes the already-solved barotropic velocity and
   adds bottom/top stress due to that barotropic component.
3. The one-wet-layer matrix has no interior viscosity recurrence.  Bottom
   drag changes `zwd` at `dynzdf.F90:293-297`; top drag changes the same
   diagonal at `:298-304` after `zdfdrg.F90:116-129` selected `rCdU_ice`.
4. The RK3 surface-stress statement updates the solve RHS at
   `dynzdf.F90:326-335` (`u`) and `:500-509` (`v`).  The terminal divide is
   `:341-345` and `:515-519`.
5. Back in stage 3, `stprk3_stg.F90:430` returns from `dyn_zdf`, then
   `:433-445` computes and applies the depth-mean correction that replaces the
   3-D mean by `uu_b/vv_b(Kaa)`.  The subsequent velocity halo update is
   `:486`; stage-3 T/S work and its final halo operations are `:494-645`.
6. `sbc_ssm` at `kt=2` is the next existing register and reads the committed
   state/time level.  The new end-step register will preserve the explicit
   `Kbb/Kmm/Krhs/Kaa` indices rather than naming a value merely “now”.

The fresh NEMO stream will record, for both u and v: the pre-drag matrix
diagonal; bottom and top diagonal increments separately; `zwi/zws/zwd`; the
pre-surface-stress RHS; the surface-stress increment; `e3u/e3v(Kaa)`; the
terminal solve output; the pre/post stage-3 barotropic correction; the
post-velocity-LBC value; and the value immediately before `kt=2 PRE_SSM`.

## Hypotheses and falsifiers

The shared card currently instantiates `surface_stress_implicit=False` and
`barotropic.nemo_stage_mean_imposition=False`, while the NEMO path above
unconditionally applies both operations in this composition.  This is a
measured construction fact from an instantiated fp64 card, not yet an owner
claim.

H15-A predicts that the first shared-path non-bit operand is the input to the
implicit momentum solve, because the card does not select NEMO's
`dynzdf.F90:328-330,503-504` surface-stress placement.  It is **CONFIRMED** if
the shared pre-solve diagonal is bit-identical but its post-surface-stress RHS
is the first non-bit row.  It is **REFUTED** if that RHS is bit-identical.

H15-B predicts that, after selecting the implicit surface-stress placement,
the next necessary identity is the stage-3 barotropic-mean replacement at
`stprk3_stg.F90:433-445`.  It is **CONFIRMED** only if the solve output is at
bar while the post-correction value is the first over-bar row.  It is
**REFUTED** if the first split precedes or follows that operation.

The primary preregistered target after any faithful fix is
`kt=2 PRE_SSM.u` **bit-identical (1/1)**.  The private arms toggle exactly one
of (a) implicit surface-stress deposition and (b) end-stage mean imposition;
each must reproduce its named red boundary before the corresponding change is
accepted.  A row-level `+1e-8` plant at every new boundary must exit nonzero.
Top-drag scaling is evaluated before attribution but cannot own `kt=1` if the
dump reconfirms its coefficient is exactly zero.

Only after `kt=2 PRE_SSM.u` is bit-identical will the continuous walk advance
through `kt=2..8760`; it stops at the next first over-bar register and reports
no values downstream of that stop.

## Round-13 review and hygiene predictions

The pinned Round-13 exchange prefix will be re-scored.  The expected corrected
statement is **448,950/448,950 bit-identical** with no stale `t_bo` debt.  A
missing required stream must raise the gate's named `GateError`, never a Python
`UnboundLocalError`.  The eight exploratory roots ending `_oracle` through
`_oracle_h` will be listed as flagged for future cleanup; none will be deleted.

The assumed-shape lower-bound issue is treated strictly as an instrument
correction: an assumed-shape dummy defaults to lower bound 1 even when its
actual argument originated at a halo bound.  Any affected rows will be named
and re-measured from a fresh config-local writer; physical NEMO statements are
unchanged.

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| fresh config-local `kt=1` operand and end-step writer | ASKED | preregistered above |
| one-variable arms in NEMO order | ASKED | two selectors, evaluated separately |
| shared-path fix and continued first-over-bar walk | ASKED | conditional on measured owner |
| Round-13 448,950-row promotion and named missing-stream error | ASKED | included |
| flag eight exploratory roots without deletion | ASKED | included |
| edit shipped NEMO, delete retained files, add a second solver, GPU, `mpirun`, or push | UNASKED | forbidden / not planned |
| claim a source-statement owner before the new register | UNASKED | withheld |

## CONFIRMED / PLAUSIBLE

**CONFIRMED:** Round 14's registered boundary and continuous measurements; the
executing NEMO order and branches cited above; and the instantiated card's two
false selectors.  **PLAUSIBLE, UNMEASURED:** H15-A and H15-B.  They remain
hypotheses until the new stream and one-variable arms discriminate them.

## Addendum R15-C — preregistered operand discriminator

Status at registration: the first fresh writer and private shared-path trace
have **CONFIRMED** that selecting `nemo_stage_mean_imposition` removes the
`1.0844585561836217e-4` owner, but leaves a sub-bar, non-bit `u` row.  They also
showed, post-hoc for hypothesis design, that NEMO's stage-3 pre-strip value is
`3.9043073711832223e-05` whereas the shared driver hands zero to its ZDF
composition.  No Kbb/RHS operands have yet been dumped, so the cause of that
input split remains unmeasured.

H15-C predicts that the first operand split is the time level used by the
stage-3 RHS construction at `dynzdf.F90:119-140`: either `puu(Kbb)` is nonzero
and absent from legoESM's cold-start stage input, or `puu(Krhs)` is nonzero and
absent from its stage tendency.  A fresh config-local writer will record both
terms, `r3u(Kbb/Kmm/Kaa)`, the two rounded numerator terms, and their resolved
sum before the existing `us0` boundary.  It is **CONFIRMED** only if replaying
that first differing operand accounts for all of `us0`; otherwise it is
**REFUTED**.  The accepted change must be a shared WS-RK3 time-level identity,
not a one-layer or slab-specific shortcut.

The primary target remains `kt=2 PRE_SSM.u` bit-identical.  Only after it is
1/1 will the ordered walk stop at the already observed candidate next boundary,
`kt=2 PRE_SSM.temperature`; that candidate is not promoted until the velocity
chain closes.
