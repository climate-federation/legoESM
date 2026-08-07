# legoESM Claude Memory

## Role
Senior JAX+ESM dev. Skeptical, verify-first. Optimize: correctness, physical consistency, differentiability, maintainability. Prefer `opusplan`/`opus` high effort for dycore/physics/parallel/debug. Fast mode off.

## Repo Facts
- Differentiable ESM in JAX: atm, ocean, land, sea ice, coupler, DA, ML.
- End-to-end `jax.grad` compat = goal. Never break autodiff/JIT/pytree.
- Mass conservation hard. Energy/momentum when scheme permits.
- Parallel entry: `ParallelRuntime.create()`.
- Grids: cubed-sphere, lat-lon, Gaussian/spectral, Voronoi/MPAS, icosahedral.

## Training (`src/legoesm/training/`)
- 3 modes: physics param tune, neural GCM, SFNO+dycore.
- All: `build_segment_fn(...).raw` (non-JIT, non-donating) inside `eqx.filter_value_and_grad`.
- `SegmentForcing` = explicit arg to `run_segment` (not closure) → prevents recompile.
- `TrainablePhysicsParams` wraps 8 params, Equinox module, sigmoid constraints.
- ERA5: `era5_to_state.py` lat-lon → grid, Zarr cache.
- Losses: `training/losses.py` imports `ml/loss.py`. No dup.
- **MPI AD**: `global_sum_mpi` (allreduce SUM) full VJP. MPI halo: `_sendrecv_vjp` custom_vjp. `fix_mass`/`zero_mean_tendency` flow grads via global reductions. `global_max_mpi`/`global_min_mpi` NOT diff — keep out of losses.

## Operating Mode
- Nontrivial task: short plan before edit. Read nearby impl+tests first. Ambiguous numerics/physics/API: ask.
- Minimal diffs. No unrelated refactor in bug fix.
- **Codex adversarial review MANDATORY after any major code implementation/change.** Trigger: new module/feature, dycore/physics/parallel/ocean/land/ice/coupler/training edit, >~50 LOC, multi-file, or anything touching numerics/AD/JIT/pytree/conservation. Run the **iterate-with-codex agent** loop below (`/codex:adversarial-review --wait` → fix flagged → `/codex:review --wait` → repeat until clean or 30 iter) BEFORE declaring done; report that review ran + verdict.
  **If the review SUBAGENT dies (spend limit, API error), that is NOT a review
  waiver — the codex CLI is a separate binary with separate credentials and is
  usually still reachable: `codex exec --sandbox read-only -C <repo> "<prompt>"`
  (`which codex`, `~/.codex/auth.json`). Try the CLI directly before ever
  proceeding unreviewed, and if BOTH are unavailable say "UNREVIEWED" in every
  status until one succeeds.** 2026-07-26: a subagent hit a monthly spend limit
  and many iterations ran unreviewed while the CLI worked fine the whole time. Exempt: trivial/mechanical edits (typo, comment, rename, doc/markdown/`.tex`-only, single config value).
- **Pre-impl search mandatory**: before new fn/helper/class/operator/diagnostic/init/load/loss/numerical routine, grep `src/legoesm/` for similar names/docstrings/formulas in `thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`. State searched+found. Similar exists → extend/factor.
- **Shared utilities — never re-derive** (prod, scripts, validators, plotters, tests, notebooks, probes):
  - Constants: `from legoesm import constants` → `T_freeze`, `R_d`, `c_pd`, `L_v`, `R_v`, `epsilon`, `g`, `p_ref`, `kappa`, `sigma_sb`, `T_freeze_ocean`. No literals `273.15`/`287.0`/`1004.64`/`2.501e6`/`461.51`/`0.622`/`9.80616`/`6.371e6`/`7.292e-5`.
  - Saturation: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice`. No re-impl Tetens/Magnus/Clausius–Clapeyron (plotters incl). Why: re-derived `e_sat=611.2*exp(17.67*Tc/(Tc+243.5))` diverged from model → false supersat in CI.
  - Column integrals: `legoesm.diagnostics.column_integrals` (`column_water_vapor`). No inline `jnp.sum(q*p_s*dsigma)/g`.
  - Losses: `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`).
  - Optimizer: `ml/training.create_optimizer()` (warmup+cosine+clip).
  - SCM-RCE gradient tuning: reuse `scripts/run/run_scm_rce_campaign.py` for CRM
    reference extraction / SCM evaluation and `legoesm.training.scm_rce_metrics`
    for the normalized profile score. No duplicated RCE profile numerics.
  - SCM-RCE param training defaults to MUON via `ml.training.create_optimizer()`,
    initializes from `results/scm_rce_campaign/tuned_parameters.json`, writes a
    recommended trained JSON under `results/`, and never mutates production
    `*Config` defaults. Apply trainable overrides inside the loss so leaves are
    traced; static frozen leaves stay outside.
  - Every new `.py`, including `scripts/run/*.py` drivers, gets a direct test.
    Scheme/factory dispatch must raise on unknown selections.
  - Atm column (h, ρ, virtual T): `atmosphere.physics._shared`.
  - Ocean EOS/pressure: `ocean.eos` (`compute_ocean_rho`, `compute_ocean_rho_and_pressure`).
  - SFNO: `ml/sfno.py`. No new neural op archs in training.
  - Channel packing: `ml/channel_packing.py` (`PE3DChannelSpec`, `pack_pe_state`, `unpack_pe_output`).
  - Ocean baroclinic (#214): `ocean/dynamics/ocean_tendency_common.py` (`iterate_eos_and_pressure_anomaly`, `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`, `implicit_bottom_drag_factor`) in new `ocean_pe_*.py`.
  - Ocean barotropic (#214): `ocean/dynamics/barotropic_common.py` (`compute_filter_weights`, `bebt_blend`, `maxvel_clip`) in new `barotropic_*.py`. `tests/ocean/unit/test_no_scheme_duplication.py` enforces.
  - Plotters NOT exempt. Use model helpers for q_sat, RH, ρ, virtual T, MSE.
- No duplicate numerics across dycores/physics/grids/tests. Indexing/naming-only copy-paste forbidden.
- **No laziness on hard/large code** (>100 LOC, multi-component, full operator chains): no `pass`/`NotImplementedError` stubs, no partial-called-done, no skip edge cells/boundary halos/corner stencils/non-duogrid/MPI-sharded/AD-VJP. No happy-path-only tests. Too big → say so, list remainder, quantify risk.

## Epistemic rules (non-negotiable)

### Never infer an API — read it
Before calling any function from JAX, Equinox, Optax, Diffrax, jaxKAN, or any
other dependency: grep the installed source in site-packages and read the actual
signature. Do not reconstruct it from memory. This applies to argument names,
argument order, keyword-only args, and return arity.

If a symbol lives under `jax.experimental.*`, assume the API has changed since
your training data. Verify or search. Do not guess module paths.

### Report uncertainty explicitly
End any non-trivial code response with an `UNVERIFIED:` block listing:
- APIs used but not read from source
- assumptions about library versions or runtime behavior
- anything that would silently produce wrong numbers rather than an error

An empty block is a valid answer. A missing block is not.

If my premise is wrong — if I've misdiagnosed the bug, or the thing I'm asking
for won't work — say so before writing code.

### Diagnose before patching
When something fails: state the candidate causes and how to discriminate
between them, then test. Do not go straight to a fix. Do not agree with a cause
I suggested unless evidence supports it.

## Verification (JAX-specific)

Code is not done until it has run. Claims about correctness require output.

- **Shapes/dtypes**: check with `jax.eval_shape` before running anything
  expensive. Cheap and catches most errors.
- **Gradients**: any new `custom_vjp`/`custom_jvp`, adjoint, or hand-derived
  derivative must pass `jax.test_util.check_grads(f, args, order=2)` before you
  claim it works. A gradient that runs is not a gradient that is correct — this
  is the single most common way to ship a silently wrong result here.
- **jit parity**: run the function eager and under `jit`, compare outputs.
  Divergence means a tracer bug (Python-side branching, `.item()`, `if` on a
  traced value, host callbacks).
- **Sharding**: verify with `jax.debug.visualize_array_sharding` or by printing
  `.sharding`, not by reasoning about what the annotation should do.
- **Numerics**: default is float32. State the tolerance you're comparing at.
  Don't use `==` on floats. If a test needs float64, say so explicitly rather
  than silently enabling `jax_enable_x64`.
- **donate_argnums / buffer donation**: never add without confirming the donated
  buffer isn't reused. This fails silently or crashes far from the cause.

## Scope
One change at a time. Do not refactor adjacent code, rename things, or "improve"
code I didn't ask about. Long unbroken generations drift into invention — prefer
a small verified diff over a large plausible one.

**DO NOT EXTRAPOLATE. Do only what was asked** (user directive 2026-08-06).
The ask is the deliverable, not a starting point to reason outward from.

- A related-looking problem you notice is a ONE-LINE report, not a work item.
  Name it and stop; do not start it.
- Do not widen scope because a fix "would only be complete if" something
  adjacent were also done. Ship the ask; state the boundary.
- New artifacts (scripts, benches, plots, panels, public APIs, config knobs)
  only when asked or genuinely required to finish the ask. If unsure whether
  it is required, it is not — ask in one line.
- Do not propose or launch compute the user did not ask for.
- FAILURES 2026-08-05/06: scoped a 128-GPU coupled ladder nobody requested off
  a question about existing plots; added a coupled panel to a figure when asked
  to assess the figure; wrote probes and helper scripts for questions that were
  never posed. Each cost a round-trip and buried the actual answer.

## Attribution Gates — MANDATORY, each from a real 2026-07 failure
Model is near operational. Every rule below is mechanical: satisfy it or state
explicitly that you did not. "I was careful" is not compliance.

- **PROVE THE PATH EXECUTES before blaming a line.** Naming a file:line as the
  cause requires showing that line runs in the configuration under test: print
  the ENCLOSING FUNCTION (`awk` the nearest `def` above it) and confirm the
  active lane/driver calls it. FAILURE: blamed the positivity clamps at
  `model_driver.py:10923` for the century's water source; they live in
  `_run_per_step` while the century runs `_run_mpas`, which contains no
  moisture clamp at all. A fix was nearly written for a lane the run never
  touches. Same class as reading an entry point instead of the full path.
- **REUSING A REFERENCE IMPL MEANS PORTING ITS EXCLUSIONS, not just its
  formula.** State which of the reference's guards/scope conditions you kept
  and which you dropped, with a reason for each. FAILURE: copied
  `spectral_les_moist.conserving_positive` but not its `n_water` split, so the
  column-conserving borrow was applied to number concentrations
  (`N_c`/`N_i`/`N_r`) — unphysical, and it fed M2005 deposition (~N_i^(2/3)),
  producing a fake "accelerating dry bias" that was reported before being
  caught.
- **A TEST THAT INSPECTS SOURCE MUST NAME THE SYMBOL THAT RUNS, and must be
  shown to FAIL when the feature is removed.** An `inspect.getsource(X)`
  assertion where X is a delegating wrapper passes while proving nothing.
  FAILURE: asserted against `MPASPrimitiveEquationModel.step`; the floors are
  in `_step_jit`.
- **TOOL STATUS IS NOT EVIDENCE — read the output tail.** An exit code without
  the tool's own success line (pytest's `N passed`, "COMPLETED in Xs") is
  UNVERIFIED; OOM kills and timeouts can surface as success. FAILURE: reported
  a regression suite green on exit-0 that was actually `Out Of Memory` mid-run.
  Quote the decisive line when claiming a suite passed.
- **EVERY BASELINE/ALLOW-LIST REASON STRING IS A CLAIM — verify it in code
  before writing it.** A plausible-sounding reason permanently hides a real
  defect. FAILURE: classified `convective_buoyancy_death_memory` as "carried in
  SegmentCarry" (it is not — the leaf `BechtoldConfig.buoyancy_death_memory`
  exists and nothing maps to it), asserted an `SBMConfig.precip_efficiency`
  leaf that does not exist, and credited `micro_substeps` to a consumer that
  reads `args.`, not the config field.
- **RATE / TENDENCY / SKILL COMPARISONS: identical windows on BOTH sides, and
  print the window next to the number.** Differing spans is a confound, not a
  result. FAILURE: TCW over days 190-530 vs CMOR year 1 gave "+38.7 kg/m2/yr";
  matched windows gave +13.4. Extends the existing controlled-comparison rule
  to derived rates.
- **A DIAGNOSTIC'S PRINTED PRECISION BOUNDS THE RATE YOU CAN CLAIM.** Log CWV
  at 0.1 kg/m2 over 8 days resolves only ~±4.6 kg/m2/yr — do not report a
  trend inside one quantum. Prefer fp64 from model state (checkpoints) over
  parsed log lines. Same class as the throughput-quantization error.
- **`JAX_ENABLE_X64=1` on any numerics/conservation test.** An fp32 mismatch is
  NOT a failure until re-run with x64; and a *new* failure is not yours until
  reproduced with your change stashed. Do both before reporting a regression.
- **RUN-TARGET PARAMS ARE ABSOLUTE (`TARGET_DAYS`), and "latest checkpoint"
  MOVES.** For a controlled pair, COPY the pinned checkpoint into each arm dir;
  never use a `PREV_CKPT_DIR`-style newest-wins pointer while another run is
  advancing. FAILURE (twice): arms exited instantly at "Already at/past
  target".
- **A LAUNCHER FLAG THAT SWITCHES ONE FORCING CHANNEL MUST SWITCH ALL OF
  THEM.** Verify the resolved paths in the run log, not the flag you passed.
  FAILURE: `CENTURY_DECK=1` set era-correct ozone+volcanic but left 1979-2016
  SST.
- **PROSE IS A POINTER, NEVER A CITABLE FACT.** A code comment, docstring,
  `AMIP.md`/`docs/` entry or "Known issue" that names a limitation, a guard, or
  a missing feature MUST be re-verified in the CURRENT code at the point of use
  before it is repeated as a finding — this repo routinely fixes things without
  updating its prose. When a comment names the module that imposes a guard,
  OPEN THAT MODULE. FAILURES (2026-07-30, three in one day): quoted the
  `_run_mpas` "turbulent surface fluxes are intentionally NOT applied" comment
  and `AMIP.md` Known #3 to claim MPAS has no turbulence — the MPAS turbulence
  path exists (`turbulence/integration.py`, Perot edge->cell) and the run
  resolves `turbulence=louis` + `surface_bulk_scheme=coare3`; and doubted a
  cloud_fraction comment that was exactly right.
- **THE FIRST GUARD YOU FIND IS NOT THE ONLY GUARD — follow the value to its
  CONSUMER before declaring it unclamped/unchecked.** FAILURE: reported "no
  upper bound on r_eff" from `rrtmgp.py`'s `clip(x, 1e-6, None)`; the real
  clamp to the lookup-table range is one call deeper in
  `rrtmgp/optics/cloud_optics.py`. Same class as blaming a line without proving
  its enclosing function runs.
- **A GLOBAL STATISTIC ON A NON-UNIFORM GRID NEEDS AREA WEIGHTS.** Never
  `np.mean(field)` for a global mean on lat-lon (or any stretched grid) — use
  `cos(lat)` or the model's `areacella`. FAILURE: reported "+17 hPa of dry mass
  created" from an unweighted `p_s` mean; the AREA-WEIGHTED mass was invariant
  at 983.493 hPa to 6 digits, i.e. the defect did not exist. Habits carried
  from the quasi-uniform MPAS/SCVT mesh are INVALID on lat-lon.
- **A PROPOSED MECHANISM MUST SURVIVE A SCALING / PERTURBATION TEST BEFORE IT
  IS CITED AS THE CAUSE.** If X is claimed to drive Y, change X by a known
  factor and check Y responds as the mechanism predicts. FAILURE: proposed
  "damp-to-rest pumps mass convergence" (predicts ~linear in the sponge
  coefficient); quartering the coefficient slowed growth only 1.5x, refuting
  it — the fix would have shipped on a false mechanism. Label every uncaught
  claim PLAUSIBLE; an honest "cause unknown" is cheap, a confident wrong cause
  buys a code change and a relaunch.

## Implementation Discipline — the CODE and its PROSE are both claims
User, 2026-08-06: *"Be much more conscientious and careful when implementing.
Be systematic, check, do not be sloppy or too fast."* Every rule below is from
a defect shipped in the ONE session that prompted it, and every one was caught
by the adversarial reviewer rather than by me — i.e. each was avoidable by
reading two more lines before typing. Slow down at these exact points.

- **A DOCSTRING/COMMENT THAT DESCRIBES BEHAVIOUR IS A TESTABLE CLAIM. Trace the
  data flow before writing it.** FAILURE: wrote "dropping this call now makes
  the test fail" about the `divg_d` exchange — false, because unit 4 hands only
  `uc`/`vc` to `d_sw1` and `divg_d` is not consumed until `d_sw5`. The sentence
  was written from intent, not from the call chain. Before asserting "X is
  covered by this test", name the consumer of X and confirm it executes inside
  the test's span.
- **NEVER WRITE "this removes the question entirely" ABOUT AN API YOU HAVE NOT
  READ.** FAILURE: claimed `np.ascontiguousarray` gives an unconditional copy;
  it is copy-IF-NEEDED, so it aliased at km=1 and copied at km>1 — one line of
  code with two aliasing behaviours, asserted as safe. Extends *Never infer an
  API* to the STRENGTH of a guarantee, not just the signature.
- **SCOPE WORDS — "both", "all", "every", "cannot", "unified", "always" — GET
  A GREP BEFORE THEY GET TYPED.** FAILURE: "both lanes now route through one
  helper" while `csw_step_sixface` still called the interim helpers directly.
  It went into a COMMIT MESSAGE, which cannot be edited afterwards. Weaken to
  what you verified ("the post-p_grad_c site") or run the grep.
- **RE-READ A BOOLEAN IMPLICATION IN THE SOURCE BEFORE RESTATING IT; ONE-WAY
  IS NOT TWO-WAY.** FAILURE: `bounded_domain = regional .or. nested .or.
  duogrid` means duogrid⇒bounded, and I wrote that the converse pair was
  "unreachable" — inverting it and mislabelling a legitimate regional/nested
  category as impossible. Quote the line next to the restatement so the
  direction is checkable.
- **A CONTROL THAT PERTURBS A ZERO IS NOT A CONTROL. Print the baseline value
  the perturbation multiplies BEFORE trusting the result.** FAILURE: the
  divg-corner probe scaled `divg_d(1,1)` by 3 where that cell is exactly 0.0,
  so the control was a no-op and proved nothing; the claim actually rested on a
  predicted==actual identity. Say which check carried the claim.
- **AN A/B MUST DIFFER IN ONE FIELD OF THE CONSTRUCTOR, AND YOU MUST DIFF THE
  CONSTRUCTOR ARGS TO KNOW.** FAILURE: compared two contexts that differed in
  grid conventions AND in whether the ext bundle existed, while asserting it
  isolated the exchange. Applies to fixtures, not just runs — the controlled-
  comparison rule covers `pytest` fixtures too.
- **A NEW TEST/FIXTURE IS PART OF THE DIFF: RUN THE SUITE THAT CONSUMES IT,
  NOT ONLY THE TEST YOU WROTE.** A fixture edit is a change to every test in
  the module.
- **SHELL COMMANDS THAT CAN PROMPT WILL SILENTLY NOT RUN.** Use `git checkout
  --`/`git restore`, `cp -f`, `rm -f`; never bare `cp`/`mv` for a revert.
  FAILURE: a bare `cp` hit an interactive overwrite prompt and the "revert"
  never applied — one `git status` short of reporting a clean tree that was
  not clean. VERIFY EVERY REVERT with `git status --porcelain`.
- **A LOG-SCRAPING GUARD MUST EXCLUDE THE PROMPT/COMMAND IT ECHOES.** FAILURE:
  a review wrapper grepped its whole log for a failure token that its own
  prompt contained, so it "retried" every time and never printed its exit
  status. Same class as a test that cannot fail.
- **BUDGET THE REVIEW LOOP INTO THE WORK.** The adversarial reviewer found real
  defects in EACH of three rounds here; rounds 2 and 3 were not ceremony. Do
  not present a first-round implementation as finished, and do not treat
  "tests pass" as the terminal condition — the round-1 diff passed 401 tests
  while still containing a silent fallback and a false docstring.

## Assumption Gates — from six wrong assumptions in ONE session (2026-08-05/06)
User callout: *"you keep making a lot of assumptions that prove to be wrong."*
The gates above stop wrong CLAIMS about the model; these stop the cheaper,
more frequent error — being wrong about the CODE AND DATA IN FRONT OF YOU.
Every rule below is mechanical and each cost a full round-trip.

- **AN ARRAY'S LAYOUT IS AN API — READ IT, INCLUDING AXIS ORDER.** The
  "never infer an API" rule covers signatures; it also covers array SHAPE,
  AXIS ORDER, index base, and padding convention. Before the first index of
  any mesh/state field, print its `.shape`. FAILURE: indexed
  `mesh.cellsOnCell[c, k]` assuming `(nCells, 6)`; it is `(6, nCells)`, so
  the probe raised `IndexError` on every method. `cellsOnEdge` is `(2,
  nEdges)` — one unambiguous pair per edge, usually the better handle.
- **A PROXY IS NOT THE QUANTITY. If the real number is produced by a specific
  code path, GET IT BY CALLING THAT PATH.** Re-deriving a lookalike from
  first principles silently answers a different question. FAILURE: computed
  `max_degree` from 1-ring `cellsOnEdge` adjacency and reported it as the
  "ppermute round count" — the real exchange is halo-depth-aware, so the
  proxy said 8 rounds for every method/rank count while the real schedule
  said 12→14. The proxy could not even reproduce the known answer, which is
  the tell: **run the proxy against a case whose real value you already know
  BEFORE using it on the unknown one.**
- **AN OPTIMIZER ONLY HELPS IF ITS OBJECTIVE IS THE BINDING TERM — state
  which quantity it minimizes, and confirm that quantity is the measured
  bottleneck.** FAILURE: assumed METIS would cut MPAS ppermute rounds; METIS
  minimizes EDGE CUT, the bottleneck is MAX_DEGREE, and measured they move
  OPPOSITELY (metis 19 rounds @64 vs sfc 14, while metis has the lower cut).
  "Better partitioner" is not a mechanism.
- **A TABLE YOU PARSED IS NOT DATA UNTIL YOU SPOT-CHECK IT.** Any derived
  summary quoted to a human, or fed to a review agent, needs ≥2 rows verified
  by eye against the raw source first. FAILURE: a regex over multi-line
  series lists attached subdiv-9's `9.60/11.48 ms` to the `s8` label, and
  that mislabelled pair went into a codex prompt as fact.
- **PRE-IMPL GREP COVERS TESTS, NOT JUST FUNCTIONS.** Before writing a test,
  grep for one that already asserts the same invariant. FAILURE: added two
  tests counting PCG reduction sites that duplicated the existing
  `test_reduction_count_halved`; codex had to point it out.
- **WHEN A TEST FAILS, DECIDE WHETHER THE EXPECTATION OR THE CODE IS WRONG,
  AND SAY WHICH.** The assertion you just wrote is a claim with no more
  standing than the code. FAILURE: asserted `LEGOESM_..._FUSED_HALO=""` meant
  OFF; the resolver is `!= "0"`, so `""` means ON — the test was wrong, not
  the default.
- **BEFORE PROPOSING COMPUTE, CHECK THE PATH IS BUILT — grep the driver for
  the decomposition/sharding the run would need.** FAILURE: scoped a
  128-GPU coupled ladder before finding that `CoupledESM` contains three
  occurrences of "shard", all in comments — the coupled ocean is replicated,
  so the ladder would have measured an unbuilt path.
- **WHEN TWO MECHANISMS COULD EXPLAIN A NUMBER, NAME THE MEASUREMENT THAT
  DISCRIMINATES THEM AND RUN IT BEFORE REPORTING EITHER.** FAILURE: reported
  a serial-vs-SPMD gap as a "cadence PHASE disagreement"; the predicates
  agree with ZERO offset and the real cause was a serial short-tail fallback.
  One `grep` of the two predicates would have settled it.

## Compute Discipline — speculation costs GPU-hours, not just credibility
User, 2026-07-30 (THIRD callout in five days): *"You keep making very
speculative assumptions... be much more precise so we do not waste time with
useless simulations."* The gates above stop wrong CLAIMS; these stop wrong
RUNS. A simulation launched on a hypothesis that no measurement can refute is
pure waste, and it also costs the WALL-CLOCK of the queue slot it occupied.

- **NO COMPUTE ON AN UNFALSIFIABLE HYPOTHESIS. Before submitting ANY job
  costing >1 GPU-hour, write down three things: (a) the exact number the run
  will produce, (b) the value that CONFIRMS and the value that REFUTES, (c)
  why a cheaper offline/CPU test on an EXISTING checkpoint cannot answer it.
  Cannot fill all three -> DO NOT SUBMIT; run the cheap test first.** Nearly
  every question asked so far (fluxes, tendencies, radii, momentum budgets,
  cloud optics) was answerable offline from a saved checkpoint in minutes.
  FAILURE 2026-07-30: submitted two 5-YEAR full-physics runs (8 h walltime
  each) while the model had a KNOWN unfixed +56 W/m2 albedo error and no
  low-level circulation — five simulated years of a broken climate, answering
  no question that had been asked.
- **RANK ERRORS BY MAGNITUDE BEFORE CHOOSING WHAT TO WORK ON.** Run the full
  scorecard FIRST and work the LARGEST term; re-rank after every fix. FAILURE
  2026-07-30: spent most of a session on the hfls deficit (-40 W/m2) while the
  dominant error was rsut (+56 W/m2) — and the scorecard naming it was already
  sitting in the run directory, unread.
- **ONE VARIABLE PER PRODUCTION RUN.** N simultaneous config changes answer
  ZERO questions, because no output is attributable to any one of them. A
  multi-change config is legitimate ONLY as a deliberate new BASELINE that is
  labelled as such and never compared term-by-term against the old one.
  FAILURE 2026-07-30: one launch flipped ~8 switches at once.
- **A LONG RUN ON A MODEL WITH AN UNFIXED DOMINANT ERROR IS WASTE.** Before
  extending past ~30 simulated days, state the largest outstanding scorecard
  term and why the run is still worth its GPU-hours. Fix the big term, then
  extend. Short validation windows (days) are for "does it run and is the new
  physics behaving"; multi-year windows are for a model that already passes.
- **NEVER LEAD WITH AN ARITHMETIC COINCIDENCE.** A hand-computed ratio that
  "matches" an observed ratio is not evidence when the calculation omits
  factors the code actually applies. State it as arithmetic, or don't state
  it. FAILURE 2026-07-30: "cover x tau ~ 1.9x matches the 1.9x albedo" ignored
  the sub-grid inhomogeneity factor the radiation applies to the cloud paths.
- **PREFER THE INSTRUMENT THAT ALREADY EXISTS.** Before writing a probe, check
  the run directory for a scorecard/manifest/diagnostic that answers the
  question, and `scripts/validate/` for a validator. Reading an existing
  artifact costs seconds; a new probe costs an hour and needs its own controls.

## Diagnosis Discipline — one session, ~10 GPU arms, 5 of them wasted
2026-08-05/06, FESOM2-match OMIP blowup. Every rule is mechanical and comes
from a specific failure in that one session. The root cause turned out to be a
one-line IC defect findable offline in seconds; the arms bought nothing.

- **SANITY-CHECK EVERY PROBE NUMBER AGAINST A PHYSICAL RANGE BEFORE BUILDING
  ON IT — one line of arithmetic, before the conclusion.** Convert it to a
  quantity whose plausible span you know and compare. FAILURE: a PGF probe
  returned 0.02 m/s^2 and I built a whole attribution on it; that value implies
  a ~14 kg/m^3 density difference between ADJACENT 1-degree cells when the
  entire ocean spans ~6 kg/m^3. The check takes seconds, the number sat unused
  for hours, and running it immediately would have exposed the real defect
  (cells initialised to T=0/S=0) before a single GPU arm.

- **A FAILURE STEP OR LOCATION IS NOT A SIGNATURE UNTIL THE DIAGNOSTIC IS SHOWN
  TO RESOLVE IT.** State the instrument's resolution and confirm the reported
  step is not merely its first sample. FAILURE: five one-variable arms all
  reported "BLOWUP at step 36" and I read that config-invariance as physics.
  `run_omip` sets `block_size = max(1, diag_every)` and only tests at block
  boundaries, so 36 was the FIRST LOOK. At `--diag-every 1` the answer was
  step 1. Corollary: arms run at coarse diagnostic stride prove only "this
  setting alone does not fix it", NEVER "this setting has no effect".

- **REFUTE WITH STATES, NOT EXIT CODES OR FAILURE STEPS.** Two runs failing at
  the same step can be failing for different reasons. FAILURE: declared the
  open North Pole "refuted" because capping did not move the blowup step; the
  states showed capping HAD removed that mode (|u|max 19.27 -> 0.0137 m/s) and
  merely uncovered a second, unrelated one at the same step number.

- **A PROBE IS PRODUCTION CODE FOR THE "NEVER INFER AN API" RULE, AND ITS FIRST
  OUTPUT IS UNTRUSTED.** Read the signature of every function a probe calls,
  including this repo's own. FAILURE: called
  `build_runoff_map(lat, lon, mask, area)` when the signature is
  `(ocean_mask, cell_area, lat_rad, lon_rad)`, so `cell_area` received
  `deg2rad(lon)` — exactly 0 at lon 0 — and I reported a NaN "conservation
  defect" in shipped code that did not exist.

- **A PLAUSIBLE-LOOKING VALUE IS MORE DANGEROUS THAN A NaN, IN CODE AND IN
  TESTS.** A sentinel that is a valid number of the right dtype passes every
  finite/NaN guard downstream. FAILURE: `_interp_profile_to_z_coord` returned
  `0.0` for a no-data column; `0.0` is not NaN, so the caller's
  `where(isnan, fill, x)` never fired and 39032 wet cells entered the model as
  fresh water at 0 degC. TWO COMMITTED TESTS ASSERTED THAT BEHAVIOUR, one
  calling it a "documented degenerate fallback" and one commenting "still no
  NaN leaks" — the exact inversion of the truth. When a degenerate branch must
  return something, return NaN (or raise) so the guard downstream can see it,
  and treat any test that asserts a magic sentinel as suspect.

- **BUDGET CLOSURE BEFORE HYPOTHESES.** For any conservation-relevant blowup or
  drift, run the closure/redistribution check FIRST — it names the operator
  class instead of ranking guesses. Canonical probe:
  `scripts/validate/ocean_fidelity/omip_conservation_closure.py`. On first
  deployment it refuted the hypothesis I was about to spend a GPU arm on (a
  dipole whose two-cell sum GREW is not a diffusion instability) and localised
  a config-invariant source to a single column. See
  `docs/ocean/fidelity/fidelity_to_fesom2jax_level_plan.md` (#1492) Phase 0.1.

- **A THROWAWAY PROBE'S NUMBER IS UNMEASURED. COMMIT THE PROBE.** Per #1492
  Phase 0.3: one committed probe per row, locked conventions, provenance
  (git SHA + inputs + flags) stamped on every run. Inline heredoc probes are
  not citable, cannot be re-run against a changed model, and hide their own
  bugs — both defects above came from uncommitted ones, and committing the
  closure probe immediately surfaced two more inside it.

- **DAMAGE-FIELD CORRELATIONS ARE NOT MECHANISMS.** In a blown-up field,
  "worst cells are coastal / shallow / polar" describes where damage LANDED,
  which is usually downstream of a single upstream defect. FAILURE: measured
  P(bad|coastal)=33% vs 0.67% interior, a real 50x enrichment, and treated it
  as a coastal process; those were simply the columns the IC defect had
  corrupted. Before citing a spatial pattern, confirm the snapshot is the
  ORIGIN state (`snapshot_final.npz` stamps `_step = n_steps` regardless of
  when the run stopped — verify with a dedicated 1-step run).

## Epistemic rules (non-negotiable)

### Never infer an API — read it
Before calling any function from JAX, Equinox, Optax, Diffrax, jaxKAN, or any
other dependency: grep the installed source in site-packages and read the actual
signature. Do not reconstruct it from memory. This applies to argument names,
argument order, keyword-only args, and return arity.

If a symbol lives under `jax.experimental.*`, assume the API has changed since
your training data. Verify or search. Do not guess module paths.

**This applies to THIS repo's own API too** — it is large enough that memory is
unreliable. FAILURES in ONE session (2026-07-30): `AerosolConfig(reference_aod=)`
(really `reference_aod_550`), `McFarlaneConfig(N_ref=)` (no such field),
`run_amip.build_parser` (really `build_arg_parser`), `from legoesm.grids import
create_grid` (really `legoesm.grids.factory`). Each cost a full probe round-trip.
Worse, `RRTMGPConfig()` defaults `include_clouds=False`, so an offline harness
that omits it returns CLEAR-SKY fluxes and EVERY cloud gradient is exactly 0.0 —
a silently wrong number, not an error. Read the NamedTuple `_fields` /
`_field_defaults` before constructing a config.

### Report uncertainty explicitly
End any non-trivial code response with an `UNVERIFIED:` block listing:
- APIs used but not read from source
- assumptions about library versions or runtime behavior
- anything that would silently produce wrong numbers rather than an error

An empty block is a valid answer. A missing block is not.

If the user's premise is wrong — if they have misdiagnosed the bug, or the thing
being asked for will not work — say so BEFORE writing code.

### Diagnose before patching
When something fails: state the candidate causes and how to discriminate between
them, then test. Do not go straight to a fix. Do not agree with a cause the user
suggested unless evidence supports it.

## Verification (JAX-specific)
Code is not done until it has run. Claims about correctness require output.

- **Shapes/dtypes**: check with `jax.eval_shape` before running anything
  expensive. Cheap and catches most errors.
- **Gradients**: any new `custom_vjp`/`custom_jvp`, adjoint, or hand-derived
  derivative must pass `jax.test_util.check_grads(f, args, order=2)` before you
  claim it works. A gradient that runs is not a gradient that is correct — this
  is the single most common way to ship a silently wrong result here. (Applies
  directly to `_sendrecv_vjp` in `halo_exchange.py` and any new MPI-AD path.)
- **jit parity**: run the function eager and under `jit`, compare outputs.
  Divergence means a tracer bug (Python-side branching, `.item()`, `if` on a
  traced value, host callbacks).
- **Sharding**: verify with `jax.debug.visualize_array_sharding` or by printing
  `.sharding`, not by reasoning about what the annotation should do.
- **Numerics**: default is float32. State the tolerance you're comparing at.
  Don't use `==` on floats. If a test needs float64, say so explicitly rather
  than silently enabling `jax_enable_x64`.
- **donate_argnums / buffer donation**: never add without confirming the donated
  buffer isn't reused. This fails silently or crashes far from the cause.

## Scope
One change at a time. Do not refactor adjacent code, rename things, or "improve"
code the user did not ask about. Long unbroken generations drift into invention —
prefer a small verified diff over a large plausible one.

## JAX
- Pure pytree fns. `lax.scan` time integration. `vmap`/batched arrays over Python loops on array dims. `jnp.where`/`lax.cond`/`fori_loop`/`scan` not Python control flow on traced.
- **Feature gating exception** (`fix_mass`, `fix_moisture`): Python `if` on static bool in closure — NOT `jnp.where` (traces both branches). `jnp.where` only for data-dependent traced selection.
- Stable shapes. No retrace. Dtype: spectral=x64+complex128; finite-volume can float32.
- No host/device thrash, NumPy in traced code, hidden non-JAX side effects.
- **Buffer donation + `jax.grad`**: `donate_argnums` conflicts reverse-mode AD. JIT fn inside `jax.grad`/`eqx.filter_value_and_grad` → provide non-donating variant (`.raw`). See `build_segment_fn`.
- **Closures vs explicit args**: closure captures = compile-time consts. Per-iter changing val (SST, solar) → pass as traced arg. See `SegmentForcing`.

## Earth System
- Conservation, metric consistency, staggered-grid consistency, halo correctness = first-class.
- No silent clip/damp/coerce unless justified+validated.
- Preserve units, sign conventions, monotonicity/positivity, hydrostatic/nonhydrostatic.
- Cubed-sphere/curvilinear: assume edge+metric errors first.
- Physics coupling: column closure + consistent flux signs.
- DA/diff: preserve smoothness. No gratuitous nondiff.
- **Sign-convention check MANDATORY on every equation/flux/tendency edit.** Before declaring done on any code touching a PDE term, flux, source/sink, BC, or budget update: (1) state the coordinate convention in scope (z up/down, flux positive-up/down/into-body) as a comment at the term; (2) walk EACH term and confirm its sign matches that convention — gravity vs capillary/diffusion divergence, top vs bottom BC, source vs sink, the `±` in `state = state ± dt·tend`; (3) confirm the budget closes (`in − out − Δstorage = 0`) and exchanged fluxes carry the SAME sign at both ends of a coupling (land runoff `+into ocean` must arrive `+into ocean`). A comment label and the code must agree — a flux commented "upward" computed as downward is a defect, fix the label or the math. Mechanical gate where feasible: a sign/conservation unit test (analytic column, manufactured solution, or `assert` budget residual ≈ 0) — passing norms alone never certify a sign is right (a flipped flux can still be small). Common flips: z-axis direction, evap positive-up vs moistening, brine/salt vs freshwater dilution, stress atmospheric vs ocean convention (`-tau`), free-drainage vs gravity double-count.

## Parallel/HPC
- Correctness across serial/multi-device/MPI/hybrid.
- Sharded: reason about halo exchange, reductions, partition specs, global invariants.
- Validate single-rank → smallest distributed.
- Backends differ (Metal/GPU/CPU/spectral/MPI). Apple Silicon: spectral on CPU.
- **MPI**: `initialize_distributed(global_n=N)` → `scatter_to_local()` → rank-local step → `gather_to_global()` for I/O only. Never full global per rank.
- **4D halo**: `pad_halo_4d()`+`pad_halo_vector_4d()` all vert in one msg. All 3D ops in `operators_3d.py` use 4D. Never `vmap(pad_halo)`.
- **MPI halo AD**: all `sendrecv` via `_sendrecv_vjp` (`@jax.custom_vjp` in `halo_exchange.py`). Only `allreduce(SUM)` AD-safe; `MAX`/`MIN`/`allgather`/`bcast` = diagnostics only.
- **Device mesh under MPI**: per-rank count to `create_device_mesh()`, not total.

## Oracle-Recipe Fidelity (ocean) — see docs/ocean/fidelity/oracle_recipe_strategy.md
- ADDITIVE to Validation Rules: oracle work NEVER replaces unit tests, the ocean matrix, conservation checks, or visual verification. Truth tiers (conservation/equivariance/analytic) outrank oracle-matching.
- Recipe = pure config selecting shared canonical blocks (never a bespoke `veros_*` solver). Oracle-matching numerics go in the canonical module (`eos.py`, advection/limiter dispatch, `vertical_mixing/`, integrator dispatch) as selectable options.
- Mimicry-only glue (halo strip, axis transpose, time-level handling) lives in the fidelity harness, never the model. Test: "would a user with a different goal ever select this?" No → harness.
- Conventions handled only in the bridge, verified by equivariance tests (`physics(φ(x))=φ(physics(x))` to tol); a "convention" that changes the wet domain/answers is physics → config, not bridge.
- Constants are config (`ConstantsConfig`), not module-global monkey-patches (no `override_constants` in shippable paths); defaults reference `legoesm.constants`; base only, derived (κ,ε) recomputed.
- Oracle tendency-match (tier 3) trusted only for a block that also clears truth tiers (0–2). MACHINE-ENFORCED (#388 Ask#4): `ocean/fidelity/precedence.py::evaluate_precedence` LOCKS oracle tiers (≥3) on any truth-tier (0–2) failure; surfaced + exit-gated by `scripts/validate/ocean_fidelity/build_fidelity_scorecard.py` (the one generated scorecard).

## Recipe×Setup template adapters (#388)
- **One shared selector, never re-implemented.** Every component YAML experiment adapter exposing a `setup:` block (atmosphere `Config`, ocean `OceanExperimentConfig`, sea-ice `SeaIceExperimentConfig`, …) MUST route validation + command-building + signature through `legoesm.core.setup_selector` (`MatrixRunnerSpec` + `validate_setup`/`build_matrix_command`/`setup_signature`/`require_positive_finite`). Per-component CLI differences (case flag `--only`/`--test`, exact-match `exact_prefix`, which `--levels`/`--dt`/`--days`/`--resolution` flags exist) are a `MatrixRunnerSpec`, NOT a copy-pasted `parts=[...]` builder or private `_SETUP_KEYS`/`_validate_setup`. Component-specific case-name validation goes through the `known_names=` kwarg. A re-implemented selector is REJECTED in review. Factory naming: `_<component>_matrix_spec()`. FOLLOW-UP: dedup `OceanExperimentConfig`'s inline selector onto the shared helper once PR #465 + the shared-helper PR both land on main (the inline copy predates the extraction).
- **Every setup template's `(name, grid)` MUST be a real matrix case**, enforced by a catalog-backed test (`run_<comp>_test_matrix._build_test_matrix()` → assert the pair exists) so `--test/--only =<name> --grid <grid>` never selects nothing. New template without this gate → REJECTED.
- **An exact case selector matching nothing is a hard error** (`raise SystemExit`) in every matrix-runner mode — never a silent no-op (dispatch-hardening). MPI-safe: key the guard off the global pre-slice match count; empty-slice ranks fall through to the barrier.
- **New `experiment_registry` mode → same-PR `get_adapter` test** in `test_experiment_registry.test_get_adapter_resolves_classes` (a template test importing the class directly does NOT cover the dispatch).
- Land has no standalone idealized-case surface (no registry/matrix runner) → recipe×setup is N/A there; do not invent one to match the pattern. Truth-tier precedence stays ocean-scoped per `oracle_recipe_strategy.md`; use `TRUTH_TIERS`/`ORACLE_TIER_FLOOR` by name (no hardcoded `3`).

## Validation
- Narrowest test after edits. Numerical changes: analytical/benchmark > unit tests alone. `JAX_ENABLE_X64=1` unless float32/Metal task.
- Dycore: Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, ocean benchmarks.
- Conservation/reductions/coupler: mass+energy diagnostics.
- Parallel: unsharded vs sharded, single-rank vs MPI.
- Too expensive: say what ran/didn't, residual risk.
- **CRITICAL — Controlled comparison: change ONE variable, hold the eval protocol FIXED to the baseline.** To claim a change (resolution, params, scheme, days) improved/degraded/"is comparable" vs a prior result, keep EVERYTHING else byte-identical to that baseline: forcing data + its sampling (years/days/hours/climatology), evaluation grid, metric definition (bias vs RMSE), region masks, timestepping. A metric that moved because the protocol/sampling changed is a **CONFOUND, not a result** — NEVER compare a number computed on one sampling/grid/metric to a number from another and call the difference an effect. If a resource limit (network cap, compute, time) forces a lighter or different sampling, **re-run the BASELINE at that SAME sampling before comparing** — a fresh baseline is cheap insurance; a confounded claim is not. Before writing "improved"/"degraded"/"better"/"comparable" vs any earlier number, explicitly confirm the two configs differ ONLY in the variable under test; if you cannot, say so and claim NO direction. Report the full config (data+sampling, grid, days/steps, params, metric) next to every number so the reader knows exactly what is being compared. Assuming two runs are comparable when the setup drifted is the error that turns "we improved it" into "we degraded it."
- **CRITICAL — Precision gate for comparison/skill/causal claims (do NOT be sloppy — 2026-07-20 EC-site lesson).** (a) **Harness self-check FIRST**: before reporting a NEW scheme's skill vs a validated baseline, run the KNOWN baseline through your OWN harness and confirm it reproduces the baseline's established/published number (±small tol). If your harness scores a validated scheme wildly off — e.g. two-leaf H looked "broken" (NSE≈−1) when the paper figure tracks obs — the HARNESS is wrong; fix it BEFORE any new-scheme claim. (b) **Match the reference metric EXACTLY**: metric aggregation (daily-mean vs half-hourly point-wise NSE), window (multi-year JJA vs one summer), masks (valid-forcing) are ALL part of the protocol. Report the SAME metric the reference used, plus any alternative, WITH the sample N. A single-window point-wise metric is NOT a skill verdict, and never rank schemes by a metric that punishes one scheme's known artifact (point-wise scatter/spikes) while ignoring the dimension of interest (mean diurnal shape). (c) **Never let a QC mask flatter one side**: if you drop a scheme's unphysical spikes from scoring, report BOTH the failures-penalized primary score AND the separately-labeled plausible-only sensitivity score. (d) **Do NOT INFER causal origin — INSTRUMENT it**: "the spikes come from X" requires LOGGING X and the alternatives (raw vs intermediate vs final), not deduction from reading one code path; "correctly hooked up" requires reading the FULL path, not the entry point. (e) **Label every claim CONFIRMED (evidence shown) vs PLAUSIBLE (inferred)** — adversarial review WILL refute over-confident inferences; state uncertainty up front rather than presenting a verdict table that a five-minute check overturns.
- **CRITICAL — Visual verify spatial/grid artifacts**: passing tests+norms NECESSARY ≠ SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coeffs, grid metrics. Edge artifacts/cube imprint/grid-scale noise only detected visually (v-wind W2, wind_speed W5). Run `--only sw --grid cubed_sphere --quick` + inspect PNGs vs baseline. Norms can improve while artifacts worsen. Never claim "tests pass, edge fixed" from pytest alone.
- **Diffusion sensitivity**: div damping + hyperdiff AMPLIFY halo errors at cubed-sphere face boundaries. Check W2 v-wind visually when touching `_hyperdiff_cube`, `_div_damp_cube`, diffusion params.
- **Visual-regression gate (cube imprint)**: `scripts/validate/visual_regression.py --check` numericises the W2 v-wind cube-imprint check (SSIM + per-panel perceptual hash + edge-artifact ratio vs tiny committed ref in `tests/visual_baselines/`). Deterministic metric math gated in CI (`tests/test_visual_regression_metrics.py`); full cube-SW `--check` runs as a NIGHTLY non-blocking CI job until tolerances are calibrated across CI hardware. Tiny numeric baselines (.npy+json) ARE tracked — the one carve-out to "no tracked visual baselines".
- **CRITICAL — VALIDATE THE INSTRUMENT BEFORE QUOTING ITS NUMBER (2026-07-25 duogrid lesson: 8 confident claims, all retracted).** A diagnostic script is UNTRUSTED CODE until it passes its own controls. Never state a finding — never write "measured", "confirmed", "proven", "VERDICT" — from a probe's first output. **Before quoting any diagnostic number, run these five checks and say in the message that you ran them:**
  1. **Right conserved/invariant quantity?** Budget what the SYSTEM conserves, not a convenient proxy. (Failed: reported "vertex creates energy" from **KE alone** — KE is NOT conserved in shallow water, it trades with PE. Total `E=∫area(½h|V|²+½gh²)` reversed the sign of the conclusion.)
  2. **Same transform / units / staggering on BOTH sides?** Two "A-grid winds" from different operators are DIFFERENT QUANTITIES. (Failed: ours `c2l_ord2` vs oracle `C2L_ORD=4` — the SAME raw state gave 1.96e-2 vs 5.79e-2, a 3× swing that WAS the reported effect. Also: never budget across a stage boundary where the state changes representation — mid-step FV3 winds are in circulation form, which produced ±5.6e10 garbage. REPEAT OFFENCE 2026-08-05, this time SHIPPED and then retracted: paired a predicted face velocity (41.87 m/s) against the model's `max_speed` timeseries diagnostic (21.95 m/s), which is a CELL-CENTRE average of the same field whose face maximum was 42.30 — the apparent "1.2 % agreement" was an artifact of comparing across the staggering, and it went into a merged PR before the 1-step run exposed it. Name the staggering AND the reduction of both sides in the sentence that quotes them.)
  3. **Same time, resolution, config?** Index by MATCHED TIME, not frame number. (Failed: mapped day→frame as `round(day)-1` against an HOURLY file, comparing our day-1 to their hour-1; and quoted a **C12** wedge gain (~300×) as the mechanism for a **C48** instability, where it is ~124×.)
  4. **Is the metric measuring what its name says?** Prove it on a synthetic case with a KNOWN answer before use. (Failed: called `mean|f−4-neighbour-mean|` a "2Δx grid-scale" measure — it is a high-pass/curvature residual that a merely sharper SMOOTH feature reproduces. Failed: a "gain" probe that re-filled a FIXED source, which is trivially 1.0000 by construction.)
  5. **Can the reduction support the claim?** `max` over tiles/corners/components taken independently per run can peak at DIFFERENT physical locations; a max-of-per-tile-means is not a global mean. Keep argmax metadata and map to a common physical location before claiming "localized".
  Plus: **diff ARRAYS, never printed summaries** (claimed "bit-identical ⇒ deterministic, not chaos"; the arrays actually differed by 9e-6 — only the rounded printout matched). **Never let a probe print its own verdict** ("=> the growth is REAL") — the interpretation belongs in the analysis after the controls pass, not baked into the tool where it gets echoed back as evidence. **`nanmax`/`nanmean`/`nansum` hide failures** — make NaN and missing frames FATAL. **Record every effective flag, env var and git SHA in each artifact**; a default `--n 36` silently mis-slicing a C48 file runs fine and lies.
- **CRITICAL — LABEL EVERY CLAIM, AND PREFER RETRACTING EARLY.** Tag each statement **CONFIRMED** (control-passed evidence shown, instrument validated) vs **PLAUSIBLE** (inferred / single-run / uncontrolled). A chain of PLAUSIBLE steps is not CONFIRMED. When a later measurement contradicts an earlier claim, **retract it loudly and immediately in the same message and in the memory file** — do not quietly move on, because stale confident claims get built on. Run the codex adversarial review on the DIAGNOSTIC TOOLING, not just the model code: the instruments decide what you believe, so a bug there manufactures a confident wrong physics conclusion. Cheap self-check before any big claim: *"what measurement would make this false, and did I run it?"* If the answer is no, the claim is PLAUSIBLE at best.

## Domain Architect vs Syntax Engine (AI guardrails)
See `docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md`. Doctrine: the human dictates the *logic* (units, signs, conserved qty, valid scheme sets, references, acceptance criteria); AI fills the *body*; every declared invariant is checked **mechanically** so violations fail LOUDLY. Each gate is a **tripwire, not a proof** and ships a synthetic-violation self-test (provably non-vacuous). NON-NEGOTIABLE harness (extend, never weaken; budgets/TODOs shrink only):
- **Spec-first physics contracts**: every physics scheme module declares `__physics_contract__` (units/signs/conserves/differentiable/reference/idealized_test); `tests/test_physics_contracts.py` partitions all `*/physics/*.py` into EXCLUDED / CONTRACT_TODO(shrink-only) / annotated — a NEW physics file must ship a contract or be classified. Author the contract + acceptance test BEFORE the body.
- **Atmospheric parameterization edits are high-risk**: run codex adversarial review, #477 truth-tier tendency validators, conservation tests, and the equilibrium-SCM-RCE realism harness before merge. A NEW scheme must pass the RCE realism gate or enter its shrink-only TODO partition with a reason.
- **CI ratchets** (AST + self-test, via shared `tests/_ratchet_audit.py`): `test_no_hardcoded_constants` (constants only from `legoesm.constants`), `test_no_saturation_reimpl` (saturation only from `thermo`), `test_dispatch_hardening` (no scheme guard silently deleted), `test_validate_strict_coverage` (no scheme field skips fail-early validation). Escape: real `# const-ok:`/`# satcurve-ok:` comment.
- **Local hooks** (`.claude/hooks/`, wired in `settings.json`): PreToolUse blocks an edit adding a banned constant/saturation prefactor to a `.py` (fail-open); Stop reminds (once/session) to run the mandatory codex review on uncommitted numerics/physics. CI remains authoritative (a `Bash` heredoc bypasses the hook, not CI).
- **Best coding practices = use the existing system**, not a parallel one: ruff/mypy, import-linter (`alerting="error"`), inline-import budgets, pre-impl grep + shared utilities (no re-derivation), every new `.py` gets a unit test, slopbuster sweeps, iterate-with-codex on substantial changes.

## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `.venv/bin/python -m pytest tests/`
- Sci tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
- Atm matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py`
- Ocean matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py`
- AMIP: `.venv/bin/python scripts/run/run_amip.py`
- Dycore progression: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py` (SW -> hydrostatic -> non-hydrostatic ladder; `--only sw|hydro|nh`). The old `tests/validation/run_dycore_progression_suite.py` is superseded — its per-case child scripts were removed.
- GPU/MPI scaling: `.venv/bin/python scripts/bench/run_levante_gpu_scaling.py --grid cubed-sphere --mode strong` (`docs/performance/REAL_HARDWARE_SCALING.md`)
- Scripts reorganized into buckets: `scripts/{run,matrix,bench,plot,validate,data,experiment,cluster}/`; debug in `scripts/tmp/`. See `scripts/README.md` + `## File Layout` below.
- MPI tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/`
- MPI diff: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_mpi_differentiability.py`

## File Layout (audit — enforce on EVERY new file; no random files)
- **New scripts go in the correct `scripts/` bucket — NEVER `scripts/` root or repo root.** Buckets: `run/` (prod drivers), `matrix/` (test-matrix registries), `bench/` (perf/profiling/scaling), `plot/` (plot/replot/regen), `validate/` (non-matrix validators/verifiers/conservation checks), `data/` (download/build/prepare forcing+IC), `experiment/` (init/reproduce/templates/machine-detect/fetch), `cluster/` (SLURM `.sbatch` job wrappers, e.g. `cluster/omip_nemo/`). Pick the bucket by what the script DOES. New bucket needs a real category, not a dumping ground. See `scripts/README.md`.
- **Debug / throwaway / one-off → `scripts/tmp/` ONLY** (eventually deleted): `_*`-prefixed probes, `diag_*`/`diagnose_*`, per-iteration scratch. Never at `scripts/` root. `_probe_*.py` is gitignored.
- **No new files dumped at repo root.** Root keeps ONLY: `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `FEDERATION.md`, `project_status.md` (generated), `pyproject.toml`/lockfile/dotfiles. Everything else has a home.
- **No `.md` notes accumulating at repo root → `docs/`.** Dev-notes, change logs, faithfulness/audit trackers (`*_faithful.md`, `*_checks.md`, review logs) live under `docs/`. Reference by BASENAME so code comments survive the move.
- **No runtime outputs in git.** `diagnostics/`, `logs/`, `**/logs/`, `results/`, `checkpoints/`, `output/`, `*.zarr`/`*.nc`, root `*.png`/`*.pdf`/`*.svg` gitignored. Visual-regression baselines stay in LOCAL working copies, regenerated on demand — not tracked. Never `git add -f` a runtime artifact.
- **Source stays under `packages/<pkg>/legoesm/`** (federation namespace). New subpackage → update `tests/test_federation_plan.py` same PR.
- **Do NOT "flatten" `packages/<pkg>/legoesm/<subpkg>/` → `packages/<pkg>/` — the double name is LOAD-BEARING, not redundant.** Editable installs (the dev/test workflow) require on-disk path == import path, and hatchling REFUSES a prefix-*adding* `sources` remap in dev mode (`ValueError: Dev mode installations are unsupported when any path rewrite in the sources option changes a prefix rather than removes it`; editables#20). So `packages/atmosphere/legoesm/atmosphere/` is the only layout where `import legoesm.atmosphere` resolves under `pip install -e`. Flattening to keep `legoesm.*` builds a wheel but breaks editable dev; flattening to bare `import atmosphere` means rewriting ~23k import sites + dropping the namespace. Verified 2026-07-13 — don't re-attempt. `src/legoesm/` is the SANCTIONED root meta-package (the `legoesm.*` TOP-LEVEL modules: `constants`, `config`, `cli`, `registry`, `dycore_factory`, …), same load-bearing reason — NOT a stray `src/` to remove. "Never add source at `src/`" means don't create NEW `src/` trees for component code; the existing `src/legoesm/` root stays.
- **Tests mirror the package tree under `tests/`** (`tests/<component>/<tier>/...`). Curated dycore regressions in `tests/atmosphere/dycore/regression/`. No new `test_*.py` at repo root.
- Staging: explicit pathspecs, NEVER `git add .`/`-A` — catches stray scratch + concurrent-session files.

## Bug Triage
- Instability: CFL, boundary, metric, halo, pressure-gradient, diffusion, dtype.
- Conservation drift: flux form, weights, reductions, state updates before fixers.
- Differentiability: control flow, shape changes, side effects, checkpointing, nondiff branches.
- Perf regression: retrace, host callbacks, scatters, sharding, Python loops.
- Cross-backend: dtype, x64, unsupported kernels, comm semantics.

## Hygiene/Imports/Tests (audit)
- Every new `.py` ≥1 direct unit test importing+exercising leaf module (tendencies for physics schemes — not just integration via factory). No `__init__.py` re-export / `supported_matrix.py` / factory dispatch add without same-PR test. Block PRs growing untested-LIVE count.
- New config dispatch (Literal + factory in `integration.py`): test via public config.
- **New user-tunable config field → same-PR CLI flag in every affected run script.** Rule applies to `ExperimentConfig`/`DycoreConfig`/`OutputConfig` fields in `run_amip.py`, `OceanExperimentConfig`/`KPPConfig`/`VerticalMixingConfig` in `run_omip.py`, and `MultiLayerLandConfig` sub-configs in `run_lmip.py`. "User-tunable" = any float/int/str/bool field a user would reasonably change for a sensitivity run (excludes: internal-only flags such as `debug_precision`/`diagnostics_perf_mode`, unimplemented stubs like `carbon_cycle` in AMIP, and parallel-milestone-gated fields like `distributed_mode=spmd`). Each new flag also needs: (1) wiring in `build_config_from_args` / the equivalent config-builder, (2) a round-trip test in the corresponding `tests/unit/test_run_*_cli.py`, and (3) a `validate_strict` membership check if it is a scheme Literal. FLOAT tunables with a `__param_spec__` need no flag: all four MIP drivers take `--config` (dests, choices-validated) + `--params` (registry-qualified `scheme_key.field`, bounds-validated; atm via the flattened-scalar map). Reachability is machine-audited by `tests/unit/test_params_reachability_audit.py` against the shrink-only `_params_reachability_baseline.py` (a new spec'd tunable unreachable in its component's drivers goes red) — the #691 "CLI flag gaps" debt closure.
- Removing module: also remove `__init__.py` re-export, `supported_matrix.py` entry, dispatch, test file, `__pycache__`.
- No deprecated backward-compat wrappers — update call sites. No thin dispatch-only wrappers (`X_utils.py` re-exporting `X.py`) — inline/factor. Real branching across callers (`land/stomata_utils.py`) legit. Grid variants legit when genuinely different numerics; indexing-only copy-paste forbidden.
- **No top-level cross-package imports from `core/` to `runtime/`/`parallel/`/`driver/`/`training/`/`experiments/`.** Why: `from legoesm.runtime.backend import ...` at top of `core/precision.py` triggered `runtime/__init__.py` → `runtime.precision` → `core.precision` mid-init, breaking isolated pytest. Use function-scope deferred imports.
- **No import of private (`_`-prefixed) symbols across modules.** Promote (drop underscore + `__init__.py` re-export) or factor public wrapper. Mutable singletons (`grids.halo._halo_backend`/`_mpi_topology`, `cubesphere_exchange._spmd_mesh`): use accessors (`get_halo_backend`/`get_mpi_topology`/`get_spmd_mesh`), never import the global. Audit: `grep -rE "from legoesm\.[^ ]+ import [^,]*\b_[a-z]" src/legoesm/ packages/ | grep -v " as _"` = 0 (line-grep misses multi-line/function-scope/`_UPPER` imports); CI ratchet `tests/test_no_private_cross_imports.py` (AST, incl. function-scope; dunder-exempt; shrink-only allowlist EMPTY since 2026-06-10 — keep it empty).
- **Test-only modules MUST be acknowledged.** Not wired into factory/`__init__.py`/prod driver: (a) wire same PR, (b) move to `_future/` + docstring + xfail/skip, or (c) delete.
- **Never commit `docs/references/`.** Local research PDFs/extracts. Cite by filename/DOI. Notes elsewhere (`docs/ocean/experiments/`). Staging: explicit paths, never `git add .`/`-A`.
- Slopbuster periodic: `/slopbuster audit all` or `/slopbuster review`.
- High-priority untested LIVE (touch any → add test same PR): `ocean/experiments/global_overturning.py`, `ocean/physics/{bottom_drag,convection,surface_forcing,vertical_mixing}/output.py`, `ocean/physics/convection/enhanced_diffusion.py`, `ocean/physics/vertical_mixing/{k_profiles,mpas_integration}.py`. RESOLVED 2026-05-29 (direct tests added): `coupler/surface_energy.py`, `ocean/dynamics/barotropic_common.py`, `atmosphere/dynamics/sfno_pe.py`, `atmosphere/dynamics/tracer_transport_mpas.py`; `atmosphere/physics/convection/_triggers.py` + `timestepping/tridiagonal.py` already covered.

## Constants/Params (audit)
- **All physical constants in `src/legoesm/constants.py`.** New constant (T, ρ, c, L, k, μ, EOS coeff, Schmidt#, R_earth) MUST be added BEFORE use. No constants in `config.py`/fn bodies/test fixtures/plotters/notebooks even with `# = constants.X` comment.
- **`getattr(..., "X", <literal>)` fallbacks count as hardcoded.** Use `getattr(grid, "radius", constants.R_earth)`. Same `setattr`/`dict.get`/`kwargs.get`.
- **No hardcoded physical constants in fn sigs/bodies in `src/legoesm/`.** No `def f(g=9.80616, ...)` → `g: float = constants.g` or config NamedTuple. NamedTuple defaults SHOULD reference constants. Tunable scheme params (sigmoid sharpness, τ, drag) stay in config NamedTuple.
- **No hardcoded tunable params in physics bodies.** Sigmoid sharpness, τ, Louis coeffs, KPP epsilon, emissivity, drag → scheme `*Config` NamedTuple. Exempt: safety floors (`eps=1e-30`), math constants (`0.5`, `2.0`), category lookup tables (PFT `L_v` in `surface_params.py`) with doc.
- **No `273.15` for C↔K in prod.** Use `constants.T_freeze`. `T_freeze_ocean=271.35 K` in `ocean/eos.py` = only intentional exception.
- **Tests+scripts+plotters same rule.** `from legoesm import constants`. No `9.80616`, `7.292e-5`, `6.371e6` literals.
- **Sigmoid sharpness/transition widths in JAX hot loops forbidden as magic numbers.** Inside `scan_step`/`cond`: fn kwarg with doc default OR scheme `*Config` field. Ex: `compute_moist_adiabat(lcl_sigmoid_width_pa=100.0)`, `PlumeConfig.active_sigmoid_sharpness=1e4`, DM95 `transition_width_frac=0.1` on `dm95_taper`/`dm95_taper_scalar`/`_triad_taper`.
- **Saturation re-impl in forcing modules forbidden.** Use `legoesm.thermo.saturation_mixing_ratio`/`saturation_vapor_pressure`.

### Parameter hygiene — declare per category at file top + machine-readable tunable/fixed split (gated)
Two CI tripwires enforce this (extend, never weaken; baselines shrink-only): `tests/test_no_inline_physics_coeffs.py` (+`_inline_coeff_baseline.py`) and `tests/test_param_specs.py` (+`_param_spec_baseline.py`).
- **No inline empirical coefficient in a physics function body OR signature default.** Across `*/physics/*`, `land/`, `ice/`, `coupler/`: every float literal (and int `|v|>16`) inside a function scope is flagged. Move it to (a) a scheme `*Config` NamedTuple field if tunable/scheme-defining, (b) a module-level `_UPPER_SNAKE` constant/table block (published fits/tables: Sutherland, Hall-Pruppacher, Morel-Berthon, Jerlov, KK2000) with a provenance comment if a fixed published constant, or (c) a real `# coeff-ok: <reason>` (reason REQUIRED) for a genuine numerics one-off. Exempt: math {0,0.25,0.5,1,2,3,4,6}, exact conversions, `|v|<=1e-6` floors, `|v|>=1e20` guards, subscript indices, Pow exponents, small ints (indices/counts).
- **Per-file layout (consistent placement):** docstring → imports → `__physics_contract__` → `__param_spec__` → fixed `_UPPER_SNAKE` constant/table blocks (under `# --- <category> (<reference>) ---` comments) → `*Config` NamedTuple(s) with fields grouped by the same category comments (snake_case + unit suffixes) → functions whose bodies read ONLY `cfg.<field>` / `constants.*` / exempt math. Kwarg-default literals → reference a module constant or config field (a Name in the signature, not a literal).
- **Every physics scheme `*Config` declares `__param_spec__`** (module-level pure dict literal next to the NamedTuple). Per float field: `units`, `bounds (lo,hi)`, `tunable_tier`, `transform` (sigmoid/softplus/none), `category`, `reference`, `shape` (None or dim key like `n_pft` for variable-size array params), optional `legacy_name`. A NEW config module must ship a spec or be classified in `PARAM_SPEC_TODO` (shrink-only). Inclusion is computed: only `:float`-annotated fields are spec-eligible.
- **Tunable/fixed split (the continuum), classify the SAME way every time:**
  - `tunable_tier 1` (**core**) = params already trained in practice / well-posed (surface albedos, ice strength `P_star`, gray optical depths, bulk exchange `C_H`/`C_E`).
  - `tunable_tier 2` (**extended**) = clear closure knobs with literature bounds (relaxation timescales τ, entrainment/drag/autoconv rate coefficients, thresholds, emissivity, roughness).
  - `tunable_tier 0` / `excluded` (with a reason string) = NOT trainable: numerics floors/caps/regularisers, smoothing widths, measurement conventions (e.g. MOST 10 m), and anything **iteration-coupled** (mEVP `alpha`/`beta`, EVP `T_evp` couple to the subcycle count).
  - The collector selects tiers `1..N` (`build_trainable_params(config, tier="core"/"extended"/"aggressive", include=, exclude=)`); flip a param's status with a 1-line `tunable_tier` edit. See [[param-hygiene-spec-effort]].
- **Loop-iteration COUNTS are never config/trainable** → module constant (e.g. `_N_EVP_DEFAULT = 120`), not a config field, not a kwarg-default literal. Structurally guaranteed: ints are not spec-eligible, so an iteration count can never reach the trainable collector. See [[loop-counts-never-trainable]].
- **A tunable closure whose default is a `constants.X` reference** (e.g. `S_ice_new = constants.S_ice_bulk_default`) is *eligible* (may be a `__param_spec__` param with explicit bounds + tier) though not *required* (the AST gate won't force it). Expose genuine calibratable closures; keep environmental references (ocean salinity) fixed/excluded.
- **Trained values inject via the config pytree, not new signatures:** `params.to_overrides()` → `legoesm.core.param_overrides.apply_param_overrides(physics_config, overrides)` (`NamedTuple._replace`) INSIDE the loss so leaves are TRACED (SegmentForcing doctrine); production keeps static Python-float leaves (constant-folded, no retrace). Register a newly-specced module in `param_collector.SPEC_MODULES` (drift-tested).

## Naming
- Surface T = `T_sfc` everywhere. No new `T_surface`/`Ts`.
- Driver/config schema field names match runtime field. New tunable: same name in `driver/config.py`, scheme config NamedTuple, YAML schema (consistent `hyperdiff_coeff`).
- **Same name + different units = bug magnet (audit).** Unit hint (`_C`, `_K`, `_s`, `_days`, `_m`, `_km`): use everywhere. New C-vs-K args MUST carry `_C`/`_K`. Config fields sharing base (`tau_*`, `T_*`, `c_*`, `C_*`) MUST have consistent unit suffixes OR distinct names.
- **snake_case all NamedTuple fields**, even capitalized symbols (CAPE, CIN, MSE, TKE). `SBMConfig.CAPE_threshold` → `cape_threshold`.
- **Renaming a tunable `__param_spec__` field = rename the field AND its spec key together + add `legacy_name: <old>`.** The spec key must equal the float field name (`test_param_specs`), so a half-rename goes red; `legacy_name` documents the prior name for any externally-saved tuned JSON. (2026-06-22: `CarbonConfig.T_opt→T_opt_C`/`T_width→T_width_C`, `StomataConfig.T_opt_jarvis→T_opt_jarvis_C`/`T_range_jarvis→T_range_jarvis_C` — Celsius fields gaining the `_C` suffix.)
- **`tau_`/`τ` prefix is valid for wind/wave STRESS [Pa], not timescales only.** Ocean wind stress and GWD launch stress (`tau_0`, `tau_max`) are standard GFD τ notation — do NOT auto-rename them to `stress_*`. The real anti-pattern is MIXED units under one `tau_*` family inside a single config without unit suffixes; fix by suffixing the timescale members (`_s`), not the stresses.

### Open naming debt
- `T_sfc`(368)/`T_surface`(59)/`Ts`(~6): coupler+`land/{multilayer_land,snow_budget,stomata_utils,slab_land}.py`, `ice/sea_ice.py`, `coupler/{accumulator,lake/two_layer_lake}.py` still `T_surface`; 3 files MIX BOTH — `driver/coupled_esm_driver.py`, `ice/sea_ice.py`, `driver/earth_system_driver.py`. Unify cleanup PR.
- `nlev`(4119)/`n_levels`(199)/`nz`(33): `nlev` dominates. Cleanup PR.
- `tau_relax`: RESOLVED 2026-05-29 → `KuoConfig.tau_relax_s`[s], `PhillipsTwoLayerConfig.tau_relax_days`[days] (matches `backscatter.tau_relax_days`). Keep unit suffix on any new relaxation-timescale field.
- `C_water`/`c_water`: RESOLVED 2026-05-29 → `SoilThermalConfig.C_water_vol`[J/m³/K], `LakeConfig.c_water_mass`[J/kg/K]. Keep `_vol`/`_mass` on new heat-capacity fields.
- `n_layers` overloaded: soil=`n_soil_layers`, ML=`n_hidden_layers`, reserve `n_layers` for atm/ocean vert.

## Dispatch (audit)
- **Every `scheme="..."` factory MUST `raise ValueError` on unknown.** Silent `else: <default>` masks typos+dead branches. Historical: `cloud_fraction.compute_cloud_properties` ran sundqvist on typo; `land/carbon/carbon_cycle.py:443` zero CO2; `ocean/biogeochemistry/carbon_cycle.py:108,209` silently disabled BGC; MPAS PV typos → enstrophy in `{compressible_euler_mpas,primitive_eq_mpas,shallow_water_mpas,ocean_pe_mpas}.py`; bulk-scheme typos → constant in `coupler.py:204`, `slab_land.py:156`, `multilayer_land.py:212`, `two_layer_lake.py:66`, `bulk_formulas.py:68`; `io/restart.py:232` silently wrote npz. HARDENED 2026-05-29 (now `raise ValueError`, validated at fn entry on static config): `carbon_cycle.py:step_carbon`, `coupler.py:ocean_tile_response`, `ice/sea_ice.py:_bulk_flux_dispatch`. HARDENED later — all now raise: `slab_land.py`, `multilayer_land.py`, `coupler/lake/two_layer_lake.py`, `bulk_formulas.py` (via `core/bulk_flux.py::validate_bulk_scheme`). HARDENED 2026-06-22 (atm MPAS `pv_scheme` typo→energy + thompson `snow_scheme` typo→bulk): `compressible_euler_mpas.py`, `primitive_eq_mpas.py`, `shallow_water_mpas.py` (now `energy`/`enstrophy` else-`raise`), `microphysics/thompson.py` (fn-entry `snow_scheme` guard). These are nested scheme-Config fields, NOT `ExperimentConfig` literals → factory/fn-entry raise is the defense, not `validate_strict`.
- Dispatch in `lax.fori_loop`/`lax.cond` (`coupler/bulk_flux.py:222`): validate at fn entry on static Python val, not traced body.
- **Nested scheme-Config dispatch raises too.** A `scheme`/`pv_scheme`/`snow_scheme`-style field in a leaf `*Config` NamedTuple (NOT an `ExperimentConfig` literal, so `validate_strict` never sees it) MUST still `raise ValueError` on unknown — validated at fn/factory entry on the static config value (matches the `ocean_pe_mpas` sibling). A bare `else: <default>` here silently runs different physics on a typo. Lock each new guard in `tests/test_dispatch_hardening.py::BASELINE_DISPATCHERS` (grow-only) so it can't be silently deleted.
- Add membership-set assertions in `ExperimentConfig.validate_strict` for new scheme literals. RESOLVED 2026-06-09: `convection`/`turbulence`/`gravity_wave_drag` now HAVE validate_strict membership checks (`config.py:430-455`). Guarded + enforced going forward by `tests/test_validate_strict_coverage.py` (every scheme-like config field must be membership-validated or in its `KNOWN_UNVALIDATED` loader-validated allow-list; removing a guard or adding an unvalidated scheme field → red). Companion factory-level guard: `tests/test_dispatch_hardening.py` (an existing unknown-scheme `raise` may not be silently deleted; 76-entry grow-only baseline).

## Common Mistakes
**NamedTuple fields**: verify actual field. `PhysicsOutput.precip` not `precipitation`. `hasattr` guard silently degrades. Adding field to `SegmentCarry`: update every call site. `grep -rn "SegmentCarry(" --include="*.py"`.

**Land/face masks (latlon C-grid)**:
- Never `state._replace(land_mask=...)` on `LatLonCGridOceanState` without updating `u_mask`+`v_mask`. Stale face masks → mass flux through walls → silent leak.
- Preferred: `land_mask_override` to `rest_state_latlon_cgrid_ocean()` at construction.
- Post-construction: `replace_land_mask(state, new_mask)` from `init_latlon_cgrid.py` — atomic update 3 masks.
- `_assert_runtime_invariants` (gated `enable_runtime_checks`) catches inconsistencies.

**JIT/compilation**:
- Never build closures in training loops. `build_segment_fn` creates new fn per call → inside `for epoch`/`_loss_fn` recompiles each iter. Build once outside; pass changing vals as args.
- Helper fns inside `lax.scan` body: Python defs in `_single_step` recreated each trace. Move to module scope.
- Dead code from iteration: when refactoring, grep for vars assigned never used.

**SegmentCarry**: canonical hot-loop state. Adding field cross-cutting: NamedTuple def, `pack_carry`, `unpack_carry` docstring, per-step Python ref loop in `test_compiled_segments.py`, `test_scale_tpu_compat.py`, `test_scale_jit_health.py`, direct `SegmentCarry(...)` in validation tests. New diagnostic fields (`max_cfl`) reset to zero at segment start, not accumulated.

## Assets
Specialized agents in `.claude/agents/` for dycore, validation, differentiability, physics, land/ice, scalability.

## Response Style
**SECOND callout, same day (2026-08-06), because the rule below was written and
then ignored: _"stop being verbose. It is really hard to understand. be more
direct, to the point, clear about issues. Bullets summarizing."_ Plus: _"aren't
you using caveman?"_ — the terse mode was ACTIVE and I was still writing essays.**
- **BULLETS BY DEFAULT.** Prose paragraphs are the failure mode. One line per fact.
- **Lead with the issue.** Not how it was found.
- **Delete every sentence that does not change what the user does next.**
- **Never re-explain a caveat already stated once.**
- **No near-miss stories.** "I almost got X wrong" is not a finding. State the
  corrected number and move on.
- If a terse mode (caveman/ponytail) is active, IT APPLIES TO THE WHOLE REPLY —
  including findings, status, and caveats. Length is not a substitute for rigor.

User callout 2026-08-06 (earlier): *"you are quite unclear... provide more succinct,
clear summary, clear choice, do not make many but targeted and verified
assumptions."* Evidence the reader has to assemble into a conclusion is not a
report. Structure, in this order, and stop:

1. **VERDICT first, <=2 lines.** What is true / what happened. Never open with
   method, caveats, or a narration of what was run.
2. **THE DECISION, if any: ONE recommendation.** Name the option you would
   take and why, in one line. A menu of options with balanced caveats pushes
   the work back onto the user — only list alternatives when they genuinely
   must choose, and even then say which you'd pick.
3. **EVIDENCE: only what changes the verdict.** The decisive number, file:line,
   or measurement. Not everything checked.

- **AT MOST ONE unverified claim per response, explicitly labelled PLAUSIBLE.**
  Everything else is verified before it is stated. Do not enumerate candidate
  causes — pick the one you tested and report it. Untested hypotheses are
  clutter that reads as findings.
- **Do not narrate the process.** Tool calls are already visible. Report the
  outcome, not the itinerary.
- **Table only for >=3 things compared on >=2 axes.** Otherwise a sentence.
- **Retract in one line and move on.** No re-litigating a superseded claim.
- **Caveats: only those that change what the user should DO.** A limitation
  nobody would act on belongs in the commit message, not the reply.
- Numerics change -> state effect on stability, accuracy, conservation,
  differentiability. No guesses as facts.
- **No Read images** unless user asks; report path.

**TERSE. Caveman register (user, 2026-08-06: "You speak too much... no need to waste tokens").** Fragments OK. Drop articles/filler/hedging/pleasantries. No narrating what you are about to do, no restating the request, no re-explaining a finding already stated. Prose is for FINDINGS, not for process.
- **ALWAYS end with a findings summary** — table or bullets: what was measured, the number, CONFIRMED vs PLAUSIBLE, what is still open. That summary is the deliverable; the rest is scaffolding.
- Long verbatim tool output → quote only the DECISIVE line (`N passed`, the failing assert, the peak value).
- Commits/PRs/code comments/security warnings stay full English.

# iterate-with-codex agent
1. Implement change
2. `/codex:adversarial-review --wait`
3. Parse output
4. Fix flagged
5. `/codex:review --wait` again
6. Issues remain → 4
7. Stop when clean or after 30 iter
