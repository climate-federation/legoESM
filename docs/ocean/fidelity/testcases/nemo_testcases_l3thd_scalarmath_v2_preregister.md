# Lane 3b scalar-math oracle V2 preregistration

Date: 2026-09-04  
Tracker: `climate-federation/legoESM#1699`  
Status: PREREGISTERED before either scalar-math build or run

## Question and one-variable build arm

This round measures the consequence of user Decision 4: rebuild the accepted
C1D_OMIP_L3 SAS-ice oracle with gfortran math-call vectorization disabled.
The sole build-variable arm is the added `-fno-tree-vectorize` flag in
`arch-conda-scalarmath.fcm`; every config-local `MY_SRC` file, the executed
`EXP_SASICE` deck, and preprocessor-key file must be byte-identical to oracle
V1.  Two independent source-tree copies, configurations, executables, and
full-year run directories will be created.  Nothing existing will be
overwritten or deleted.  Each executable must contain zero dynamic `_ZGV*`
symbols according to `nm -D`.

The supplied scalar-math arch file and its complete flags are registered inputs
and will be SHA-256 pinned.  The executable, copied source/deck/key files,
forcing inputs, all binary instrument streams, restarts, and numeric
`ocean.output` content will also be pinned.  The two V2 runs must agree
byte-for-byte for every binary stream and restart; reproducibility failure is a
blocking finding, not a value to average.

## V1-to-V2 stream predictions

The byte audit covers every `oracle_si3_*.bin` file found in either run, not
only the four streams consumed by the gates.  It also covers ocean and ice
restarts and numeric tokens in `ocean.output`.  Before observing V2:

| artifact | preregistered prediction | reason |
|---|---|---|
| bulk operand stream | DIFFERENT | active ice saturation uses Goff-ice `LOG10` and real exponentiation in `sbc_phy.F90:665-679,727-749`; its derivative calls the same exponentiation at `:693-711,771-788` |
| exchange stream | DIFFERENT | it carries the bulk outputs and later thermodynamic flux/state values (`icestp.F90:245-268`) |
| thermodynamics frames | DIFFERENT | their entry fluxes inherit bulk differences and active BL99 radiation/conductivity evaluates `EXP`/`LOG` (`icethd_zdf_bl99.F90:217,228-230,314-315`) |
| ZDF input stream | DIFFERENT | it registers the changed thermodynamic inputs to BL99 |
| other registered SI3 operand streams | DIFFERENT where downstream of the first changed operand; otherwise IDENTICAL | attribution must distinguish a direct transcendental owner from upstream carry |
| ocean and ice restarts | DIFFERENT | accumulated state inherits the changed bulk/thermodynamic trajectory |
| `ocean.output` numeric results | DIFFERENT | the printed diagnostics/restart state inherit the trajectory; build labels and timing text are excluded from the numeric-token comparison |

`ice_alb` has active `LOG` and `EXP` paths at `icealb.F90:125,130,160,167-175`.
Only the executing branch may own a measured difference: disabled pond terms
and branch arms whose inputs do not reach them cannot be cited as owners.
For every differing stream the audit will report the first frame, registered
field, bit/absolute difference, and either (a) the directly executing
`EXP`/`LOG`/real-power/`TANH`/`SIN`/`COS` source line or (b) **upstream carry**
with the earlier owning field.  A downstream difference will not be relabeled
as a fresh transcendental difference.

The two V2 streams are declared **REPRODUCIBLE** only on byte identity.  V1 is
retained and labeled oracle V1; V2 is a new, separately pinned oracle, not an
in-place replacement.

## Rung 3.5b bit score

The existing bulk parser and formula implementation will be reused.  In
addition to its pointwise normalized bar
`abs(lego-nemo)/max(abs(nemo),1) <= 1e-15`, every registered row will be
compared as a binary64 bit pattern.  Non-bit-identical rows will be grouped by
their first direct function owner.  Each group reports its row count and the
maximum conventional relative error `abs(lego-nemo)/abs(nemo)` for nonzero
oracle values; exact-zero oracle rows are counted separately and never hidden
behind a denominator of one.

Rows whose only remaining direct difference is JAX versus scalar-glibc
`exp` or `tanh` are labeled **AWAITING_LIBM_POLICY**.  This lane will neither
add a second library-exact implementation nor alter the shared precision
policy.  `log`, `sin`, and `cos` are expected to be bit-identical under the
user's supplied control evidence, but that expectation does not substitute for
this gate's row-wise result.  A bar pass is not called bit-identical.

## Closed column re-score

The existing full-year operator and continuous-trajectory gate will consume
oracle V2 with fp64 policy on CPU.  It will remeasure the closure debts:

- independent exact-entry `kt=4242 POST_DH.h_i` injection;
- positive-subnormal-snow rows and their largest registered row;
- year-end continuous `t_su`, `e_i`, and `h_i` errors; and
- all six phenomenology rows.

A V1 debt is classified **explained by vectorized math** only if the V1-to-V2
oracle displacement owns the same registered operand/path and the V2 score
removes that debt to its declared bar (or accounts for it in sign and
magnitude where no binary bar is defined).  Mere numerical movement is not an
explanation.  The rung remains CLOSED / MIXED DEBT; this round re-pins rather
than silently reopening or upgrading its verdict.

## Controls

The committed gates must exit nonzero for: a one-bit mutation in a V2 stream;
substitution of a V1 stream under a V2 hash; a missing stream-inventory row;
a deliberately wrong bit-owner group; a bulk output perturbation; and a V1
thermodynamics root presented as V2.  The pre-existing bulk planted controls
remain required.  A deliberately selected vectorized executable must fail the
zero-`_ZGV*` binary check.  Controls compare independent artifacts and perturb
nonzero registered values.

## Decisions

| Choice | State | Disposition |
|---|---|---|
| Rebuild every NEMO oracle with math-call vectorization off | ASKED | This round creates and pins C1D oracle V2. |
| Use `arch-conda-scalarmath.fcm` | ASKED | Only added physics-relevant build flag is `-fno-tree-vectorize`. |
| Rebuild and rerun twice | ASKED | Independent byte reproducibility check. |
| Add row-wise bit comparison to rung 3.5b | ASKED | Kept distinct from the normalized bar. |
| Wait for the shared library-exact exp/tanh policy | ASKED | Differences solely owned by those calls are `AWAITING_LIBM_POLICY`. |
| Re-score the closed column and retain MIXED DEBT | ASKED | Debt table and phenomenology are re-pinned to V2. |
| Reopen or implement rung 3.6 | UNASKED | Forbidden this round; design remains under review. |
| Implement a lane-local exp/tanh replacement | UNASKED | Forbidden; shared policy work belongs to the GYRE lane. |
| Delete or overwrite V1 artifacts | UNASKED | Forbidden; V1 remains retained evidence. |
| Modify shipped NEMO source/configuration | UNASKED | Forbidden; all work is in copied trees/configs. |
| Push commits | UNASKED | Forbidden. |

## Flagged for future deletion

Nothing is deleted.  Oracle V1, both V2 reproducibility runs, their copied
source trees, build products, and all historical run roots remain retained and
will be named explicitly in the receipt.
