# SCM-RCE convection-scheme intercomparison — protocol and findings

Ranking all 10 legoESM convection schemes (`sbm`, `dca`, `kuo`, `mass_flux`,
`edmf`, `zhang_mcfarlane`, `kain_fritsch`, `emanuel`, `tiedtke`, `bechtold`)
against a plane-CRM reference under single-column radiative-convective
equilibrium, in three parameter arms and two kernel configurations.

Every claim below is tagged **CONFIRMED** (evidence run and read) or
**PLAUSIBLE** (inferred, not yet measured).

---

## 1. Protocol (pinned — all arms share it byte-for-byte)

The only quantities allowed to vary between arms are the **parameter set** and,
for the secondary table, the **transport kernel**. Everything else is pinned in
`scripts/cluster/scm_rce_paper/*.sbatch` and never left to a default.

### 1.1 CRM reference

| item | value | why |
|---|---|---|
| driver | `scripts/run/run_rcemip_plane.py` | — |
| grid | 128 x 128 x 30, `dx` = 2 km, `H` = 33 km | validated iter-229 production config |
| timestep | `dt` = 20 s, `N_ACOUSTIC` = 12, off-centering 0.2, semi-implicit | ditto |
| advection | van Leer TVD scalars, centered momentum | ditto (script defaults) |
| SGS | Smagorinsky `c_s` = 0.2, hyperdiff 5e6 | ditto |
| microphysics | Kessler warm rain | ditto |
| **radiation** | **RRTMGP**, refresh every 90 steps = 1800 s | **deliberate deviation** (see below) |
| IC seed | `--theta-noise-amp 0.5 --seed-kind band_noise --seed-kmax 6` | **required** (see below) |
| length | 345 600 steps = 80 sim-days | equilibrium with margin |
| 3-D volumes | days 62,64,…,80 (10 files) | `--last-reference-files 5` uses days 72–80 |
| surface snapshots | every 0.5 day | supplies the CRM precip reference |
| precision | float64 | — |

Two deliberate deviations from the validated iter-229 recipe, stated so they
are not mistaken for it:

1. **RRTMGP instead of gray radiation.** The SCM campaign runs RRTMGP, and
   `run_scm_rce_campaign.py:1934-1943` warns that a gray CRM against an RRTMGP
   SCM is a radiation confound. Matching the two sides removes it.
2. **`--radiation-interval 90` (1800 s)** rather than 150 steps: the script
   itself warns above 1800 s (the SAM/WRF/CM1 recommended maximum).

**The θ′ seed is not optional. CONFIRMED:** an unseeded 1-day pilot (job
9248485) stayed *laminar* — `max|w| ≈ 9e-4 m/s`, i.e. no convection at all —
while θ′ grew monotonically under radiative destabilisation. With the band-noise
seed the same configuration convects: at day 1.4, `max|w| = 1.39 m/s`,
θ′ ∈ [−10.2, +31.4] K, dry-mass drift 1.85e-16 (machine zero). For scale, the
repo's own "convects normally" reference for this case is `max|w| = 0.64`
(`docs/physics-notes/les_crossgrid_regression_2026-07.md:152-154`).

**CONFIRMED throughput:** ~5 steps/s on one Quadro RTX 8000 at this
configuration (fp64, RRTMGP, 128²×30) ⇒ ~19 h for the 80-day run. RRTMGP at
128² fp64 does **not** OOM on a 48 GB card, contradicting the script's own
warning banner (which is calibrated for 24 GB).

### 1.2 SCM arms

| flag | value |
|---|---|
| `--days` | 100 |
| `--dt` | 600 s |
| `--analysis-days` | 5 |
| `--last-reference-files` | 5 |
| `--radiation` | rrtmgp |
| `--tune-evals` | 48 |
| `--tune-seed` | 20260705 |

* **Arm (a) literature** — shipped `*Config` defaults, provenance audited (§3).
* **Arm (b) derivative-free** — the campaign's own sampler over
  `tunable_tier <= extended` parameters.
* **Arm (c) gradient** — `scripts/run/train_scm_rce_params.py`, initialised
  from arm (b)'s `tuned_parameters.json` **for the same kernel arm**.

### 1.3 The two kernel configurations

* **PRIMARY** — `--subsidence-solve implicit_flux`: every scheme that owns the
  knob is forced onto the conservative flux-form transport, isolating scheme
  physics from the transport discretisation.
* **SECONDARY** — `--subsidence-solve as_shipped`: what users get today.

---

## 2. The matched-kernel work, and what it uncovered

Before this campaign, the mass-flux family did **not** share a transport
kernel: Bechtold and EDMF defaulted to the conservative `implicit_flux` solve,
Kain-Fritsch hardcoded it, and Tiedtke / Zhang-McFarlane / Arakawa-Wu
`mass_flux` silently inherited the leaky `advective` **function default** with
no way for a caller to change it. A ranking across that set would have partly
measured the kernel.

The two solves are **not** two discretisations of one operator:

* `advective` emits `dq_c = δ₀ M q_c,u / ρ` with **no** paired vapor sink.
* `implicit_flux` transports in telescoping flux form **and** books
  `dq_v -= dq_c`, and therefore **owes** the condensation warming
  `+(L_v/c_p) dq_c`.

Threading the selector surfaced three defects that the adversarial review
(`codex exec`, 3 rounds) and the new gates caught:

| # | finding | status |
|---|---|---|
| 1 | **Emanuel's shipped path never calls the shared kernel.** `use_genuine_mixing=True` (the default) is a buoyancy-sorting mixing matrix; only the legacy surrogate branch calls `apply_mass_flux_kernel`, and that branch's `sort_multiplier` rescaling would leave an unpaired vapor debit. | **CONFIRMED** by 4 independently failing gates (job 9248502). Emanuel is reported as **outside** the matched-kernel family, not silently kernel-matched. |
| 2 | **Tiedtke / ZM / mass_flux lacked the condensation warming** under `implicit_flux`, leaving column vapor-MSE short by `L_v ∫dq_c` (a pure cooling bias). Kain-Fritsch and Bechtold already added the term inline (in-package measurements: −549 and −56.85 W/m²), so deferring it to microphysics is *not* self-consistent — `dq_c_conv_dt` is already-condensed cloud. | **CONFIRMED**; fixed via the shared helper `mass_flux.release_detrained_condensate_latent`. |
| 3 | **EDMF — a real bug in shipped code.** EDMF ships `implicit_flux`, so the kernel debits vapor, but `edmf_convection` never supplied the matching warming. Every EDMF call was short by `L_v ∫dq_c`. | **CONFIRMED** by codex round 2. Fixed. **This changes shipped EDMF behaviour** (adds previously-missing heating) and would have biased the published ranking, since the campaign advertised `forced:edmf=implicit_flux`. |
| 4 | **Bechtold over-heated on the advective arm** — its latent term was unconditional, so under the non-default `subsidence_solve="advective"` (which never debits vapor) it added `L_v ∫dq_c` of unowed heat. | **CONFIRMED**; now gated. Bechtold's shipped default is byte-identical. |

### Independent confirmation of the EDMF fix (controlled, baseline-vs-branch)

`scripts/validate/validate_convection_physics.py` is a *different* probe from
the unit gates (its own tropical sounding, its own diagnostics). Running the
**shipped validator at `cf/main`** (job 9249390) and on this branch (job
9249138) — same probe, same sounding, only the code differs:

| scheme | H [W/m²] | Q_v [W/m²] | vapor-MSE residual H+Q_v |
|---|---:|---:|---:|
| edmf @ `cf/main` (before) | **−0.1** | −21.3 | **−21.4** |
| edmf @ this branch (after) | **+21.3** | −21.3 | **≈ 0.0** |
| emanuel (both) | 26.7 | −25.9 | 0.8 (unchanged) |
| tiedtke (both) | 38.3 | −24.5 | 13.8 (unchanged) |

**CONFIRMED.** Before the fix EDMF dried the column by 21.3 W/m² while
producing essentially *zero* heating — the vapor-MSE budget was off by the full
latent throughput. After the fix the two balance to ≈0. Emanuel and Tiedtke are
**bit-identical** across the two runs, which is the expected control: Emanuel
was reverted entirely and Tiedtke's default is `advective`, where the helper is
a documented no-op.

**Validator status, stated honestly:** the validator exits non-zero on *both*
sides. `cf/main` already fails with 3 Test-4 magnitude issues (edmf, emanuel,
tiedtke); this branch reports 5. The two extra are **`dca`**, which this branch
*added* to the validator's scheme list (it had never been covered — the
validator ran 9 of the 10 schemes). Both are newly-exposed pre-existing DCA
properties, not regressions:

* `dca` fires in a CAPE ≈ 0 column (peak 112 K/day) — its trigger gating on the
  stable sounding is poor;
* `dca` column heating is ≈ 0 while its magnitude ratio vs SBM is ≈ 0.

Test 4's magnitude band `[0.01, 100]` vs SBM is failed identically at baseline
by emanuel and tiedtke, so it is a pre-existing property of that test's design
on this probe, not something this branch introduced.

**Also confirmed pre-existing, not mine:** the two
`test_no_hardcoded_constants` failures (`packages/ocean/.../constants_config.py`,
`tests/da/test_gen_be.py`) reproduce identically at `cf/main`
(`2 failed, 3488 passed`), and `git diff --name-only cf/main...HEAD` shows this
branch never touches either file.

### A claim I retracted

I initially judged the missing condensation warming to be a self-consistent
"defer the latent to microphysics" convention and planned to report it as a
caveat. That was **wrong**, and I retract it: the review showed *Bechtold* also
adds the term, so the package convention is unambiguously that convection owns
the release. The gate that would have caught my error is now in the test suite
(`test_implicit_flux_closes_vapor_mse`).

I also mis-specified that gate's own invariant on the first attempt — it
included `+L_v dq_c`, which double-counts the enthalpy the warming just
deposited. The gate then reported a residual ratio of **exactly 1.0** for all
four schemes, which is the fingerprint of a *correct implementation measured
against a wrong invariant*. Corrected to the vapor MSE `c_p dT + L_v dq_v`;
non-vacuity is now exact (deleting the helper moves the ratio from ~0 to 1.0).

---

## 2b. Conservation residual per scheme, per kernel arm (DELIVERABLE)

**CONFIRMED**, job 9249138, `scripts/validate/validate_convection_physics.py`
Test 5; machine-readable copy in `results/scm_rce_paper/kernel_conservation.json`.
Tropical probe sounding, `dt` = 1800 s. `vaporMSE` = ∫(c_p dT + L_v dq_v) dp/g
[W/m²], target 0. `water` = ∫(dq_v + dq_c + dq_r) dp/g [kg/m²/s], target 0.
The residual is column-dependent, so the stable-sounding column is given too.

| scheme | vaporMSE as-shipped | vaporMSE implicit_flux | water as-shipped | water implicit_flux | kernel arm status |
|---|---:|---:|---:|---:|---|
| **mass_flux** | **52.5** | **−0.024** | −7.9e−05 | −2.7e−10 | forced |
| **zhang_mcfarlane** | **134.7** | **+0.0056** | −1.2e−05 | −1.1e−10 | forced |
| **tiedtke** | **13.8** | **+0.0097** | −2.7e−06 | −8.1e−10 | forced |
| kain_fritsch | −0.054 | −0.054 | −8.8e−10 | −8.8e−10 | forced (already implicit) |
| bechtold | −0.013 | −0.013 | −4.1e−10 | −4.1e−10 | forced (already implicit) |
| edmf | −0.060 | −0.060 | −7.3e−10 | −7.3e−10 | forced (already implicit) |
| emanuel | 0.84 | 0.84 | −8.5e−06 | −8.5e−06 | **not_applicable** |
| sbm | 28.2 | 28.2 | 1.9e−10 | 1.9e−10 | **not_applicable** |
| dca | −4.6e−04 | −4.6e−04 | 0 | 0 | **not_applicable** |
| kuo | 0 | 0 | 0 | 0 | **not_applicable** |

Stable probe (all schemes near zero except): `mass_flux` as-shipped **215.6**
W/m² → −0.006 forced; everything else ≤ 0.03 W/m².

Three things a ranking must be read against:

1. **The secondary (as-shipped) table is measurably confounded.** The three
   newly-threaded schemes leak 14–135 W/m² of vapor MSE and 1e−6–8e−5 kg/m²/s
   of water on the tropical probe (and `mass_flux` leaks 216 W/m² on the stable
   one). Forcing the conservative kernel drops both by **3–4 orders of
   magnitude**, to ~0.01 W/m² and ~1e−10 kg/m²/s. Any as-shipped score
   difference between, say, `zhang_mcfarlane` and `bechtold` is partly a
   134 W/m² energy-leak difference, not scheme physics.
2. **`sbm` cannot be equalised by either arm.** It leaks 28.2 W/m² of vapor
   MSE and owns no mass-flux kernel, so even the PRIMARY table does not put it
   on the same conservation footing as the mass-flux family. It is the campaign
   baseline scheme, which makes this worth stating prominently.
3. **`emanuel` likewise** (0.84 W/m², 8.5e−06 kg/m²/s water) — outside the
   matched family for the structural reason in §2.

**Consequence for arm (b)/(c) interpretation, stated in advance:** if tuning
improves a *leaky-kernel* scheme a lot in the SECONDARY table, the honest
reading is that the optimiser may be compensating for an energy/water leak
rather than improving physics — the leak is a free parameter the tuner can
exploit. The PRIMARY table is the one where that confound is removed, which is
why it is the primary. This will be checked, when the campaign lands, by
comparing each scheme's (b)−(a) improvement between the two tables.

---

## 3. Literature-provenance audit of arm (a)

Generated by AST-parsing `__param_spec__` in
`packages/atmosphere/legoesm/atmosphere/physics/convection/config.py`.
Full table: `results/scm_rce_paper/provenance_raw.md`.

**CONFIRMED: 114 spec'd tunable parameters across the 10 schemes; 6 lack a
traceable literature origin — all 6 in EDMF**, whose `reference` strings are
self-referential rather than citations:

| scheme | parameter | default | reference as written |
|---|---|---|---|
| edmf | `M_b_max` | 0.05 kg/m²/s | "EDMF mass-flux stability cap" |
| edmf | `a_u_init` | 0.1 | "EDMF scheme default" |
| edmf | `cape_activation_scale` | 10.0 J/kg | "EDMF scheme default" |
| edmf | `cape_threshold` | 70.0 J/kg | "EDMF scheme default" |
| edmf | `delta_0` | 2.0e-3 1/m | "EDMF scheme default" |
| edmf | `tau_a` | 1800.0 s | "EDMF scheme default" |

These are **flagged, not attributed**: calling EDMF's arm (a) a "literature"
configuration would overstate it. The other 108 carry author+year or a named
oracle source (IFS `sucumf.F90`/`cuascn.F90`, E3SM/CAM `zm_conv.F90`,
`convect43c.f`, Kain 2004, Tiedtke 1989, Emanuel 1991, Frierson 2007,
Arakawa & Wu 2013, Sundqvist 1978, Gregory et al. 1997, Manabe et al. 1965,
Kuo 1974, Zhang & McFarlane 1995, Bechtold et al. 2008/2014).

---

## 4. Gradient-arm feasibility (arm c)

All 10 schemes have a `__param_spec__` and the module **is** registered in
`param_collector.SPEC_MODULES`, so no scheme is silently unreachable — an
unregistered module fails loudly rather than producing an empty "trained"
result.

**CONFIRMED** tier ≤ 2 trainable-parameter counts (AST):

| scheme | tier ≤ 2 params | gradient arm |
|---|---:|---|
| bechtold | 20 | strong |
| tiedtke | 17 | strong |
| kain_fritsch | 12 | strong |
| emanuel | 12 | strong |
| mass_flux | 6 | ok |
| edmf | 6 | ok |
| zhang_mcfarlane | 6 | ok |
| sbm | 3 | ok |
| **kuo** | **2** | **structurally weak** |
| **dca** | **1** | **structurally weak** |

Caveats to report with any arm-(c) ranking:

* `dca` has a **single** trainable parameter (`cape_threshold`) and `kuo` two;
  both are at real risk of tripping the nonzero-gradient preflight. A weak
  arm-(c) result for these two is a property of the scheme's exposed parameter
  surface, not of gradient tuning.
* `dca` additionally **cannot reach** the 7 nested `AhmedNeelinDCAConfig`
  parameters: `campaign._tunable_subconfig` unwraps only CLUBB, so the
  collector sees `DCAConfig` and returns 1 parameter.
* `bechtold` requires `--bechtold-policy train_deterministic`; under the
  default policy its scheme key is discarded and the job would train nothing.
  The driver now exits loudly rather than writing an empty bechtold result.

---

## 4b. A confound in the campaign harness itself (found late, MUST be reported)

**CONFIRMED** by direct read of `scripts/run/run_scm_rce_campaign.py:149-156`
and `:657-659`:

```python
SCM_CONVECTION_SUBSTEP_SCHEMES = (
    "dca", "zhang_mcfarlane", "kain_fritsch", "emanuel", "tiedtke", "bechtold",
)
...
    if convection_scheme in SCM_CONVECTION_SUBSTEP_SCHEMES:
        return requested_substeps
    return 1
```

The SCM sub-steps convection for **6 of the 10 schemes only**. At the pinned
`--dt 600` with the default `--scm-convection-substeps 10`, those six run
convection on a **60 s** effective step while `sbm`, `kuo`, `mass_flux` and
`edmf` run it on **600 s** — a 10× difference in the convective time step,
which for a CAPE-relaxation closure directly changes how much CAPE is consumed
per outer step.

This is **not** something the kernel arms control for, and it is not something
this branch introduced — it is a pre-existing property of the shared harness.
It is nevertheless an uncontrolled per-scheme numerical difference sitting
underneath any ranking produced by this campaign, so:

* it is stated next to the tables rather than discovered later, and
* a **substep control arm** is provided: re-run with
  `--scm-convection-substeps 1`, which gives every scheme the identical 600 s
  convective step, and check whether the ranking order changes.
  `scripts/cluster/scm_rce_paper/arms_ab_substep_control.sbatch` is that arm.

If the ranking is stable between the default and the control, the asymmetry is
harmless for the conclusion and can be reported as such. If it is not, the
default-substep ranking is confounded and the control is the one to publish.
**This must be resolved before any ranking is published.**

---

## 5. Anti-confound machinery

* `--subsidence-solve` is part of `_run_signature` in the intercomparison
  driver, so a cached per-scheme checkpoint from one kernel arm can **never**
  be reused for the other. The gate asserts the two signatures differ *and* are
  identical in every other key, so it fails if the field is dropped.
* The arm and its per-scheme status string (`forced:…`, `not_applicable:…`,
  `as_shipped`) are written to the per-scheme JSON, the CSV (before every
  metric), `run_meta.json`, and a labelled line in `summary.md`. A mixed output
  directory self-reports `MIXED`/`CONFOUNDED`.
* Every sbatch runs from a **pinned detached git worktree** and echoes its SHA
  as the first log line, so no job can read a tree that is being edited.
* **`run_cached`'s cache key is a SHA-256 over a payload containing
  `"config": _to_jsonable(cfg)`** (`run_scm_rce_campaign.py:694-705`) — the full
  `PhysicsConfig`, so `subsidence_solve` is in the key and the two kernel arms
  cannot collide in the in-process cache. Independently verified.

### Adversarial-review verdict on the anti-confound machinery

Final focused review (`codex exec`, job 9249487), six targeted questions:

| # | question | verdict |
|---|---|---|
| Q1 | per-scheme convection substepping | **BLOCKER** — see §4b; documented + control arm queued |
| Q2 | `run_cached` cache key | **CLEAN** — full `PhysicsConfig` in the key |
| Q3 | `_SIGNATURE_FIELDS` / checkpoint stamping | **CLEAN** |
| Q4 | the two arms differ only in `--subsidence-solve`; arm (c) reads the matching arm's `tuned_parameters.json` | **CLEAN** |
| Q5 | the new `--convection` SystemExit guard can it false-positive | **CLEAN** — cannot |
| Q6 | any other per-scheme harness asymmetry | **CLEAN** — none found |

Q1 was found independently by direct code read before the review returned, and
is the one open item; it is mitigated by the control arm rather than hidden.

---

## 5b. Test evidence (decisive lines quoted)

| suite | result | job |
|---|---|---|
| `test_convection_subsidence_solve_threading` + `test_scm_rce_subsidence_solve_override` | **85 passed** | 9249138 |
| `test_scm_rce_convection_intercomparison_cli` + `test_train_scm_rce_params_cli` | **59 passed** | 9249089 |
| `test_tiedtke` | 19 passed, 1 xfailed | 9249398 |
| `test_emanuel` | 16 passed | 9249398 |
| `test_zhang_mcfarlane` | 16 passed, 1 xfailed | 9249398 |
| `test_kain_fritsch` | 18 passed | 9249398 |
| `test_bechtold_implicit_flux` | 12 passed | 9249398 |
| `test_edmf_convection_824` (invariant updated) | 8 passed | 9249398 |
| `test_bechtold` | 98 passed | 9249398 |
| `test_convection` (hydrostatic) | 68 passed, **1 failed** — see below | 9249398 |

The scheme regression was run **split per file**: an earlier combined run
crashed with a hard fault dump that `tail` truncated, which would have looked
like a pass. Exit codes are quoted per file.

**The one failure is PRE-EXISTING, confirmed by control** (job 9249950): the
same test, run at `cf/main` (ed03b0af1) and on this branch (3279a2e4a) in the
same job, fails identically on both with `assert 0.0 > 0.0`:

```
######## scmrce_baseline  sha=ed03b0af1 ########   1 failed
######## scmrce_wt6       sha=3279a2e4a ########   1 failed
```

Likewise the two `test_no_hardcoded_constants` failures reproduce at `cf/main`
(`2 failed, 3488 passed`) in files this branch never touches.

---

## 6. Reproduce

```bash
# Phase 1 — CRM reference (~19 h, 1 GPU)
sbatch scripts/cluster/scm_rce_paper/crm_reference.sbatch

# Phase 3 — arms (a)+(b), both kernel configurations
sbatch --export=ALL,ARM=implicit_flux scripts/cluster/scm_rce_paper/arms_ab.sbatch   # PRIMARY
sbatch --export=ALL,ARM=as_shipped    scripts/cluster/scm_rce_paper/arms_ab.sbatch   # SECONDARY

# Phase 3 — arm (c)
sbatch --export=ALL,ARM=implicit_flux scripts/cluster/scm_rce_paper/arm_c_gradient.sbatch
sbatch --export=ALL,ARM=as_shipped    scripts/cluster/scm_rce_paper/arm_c_gradient.sbatch

# Conservation residual per scheme per kernel arm (both probe soundings)
python scripts/validate/validate_convection_physics.py \
    --json-out results/scm_rce_paper/kernel_conservation.json

# Figures
python scripts/plot/plot_scm_rce_convection_paper.py
```

---

## 7. Status

* **Phase 1 CRM reference: running** (job 9248515). Convecting, dry-mass drift
  at machine zero, ~19 h wall.
* **Phase 2 kernel threading + four defect fixes: complete.** 3 rounds of
  `codex exec` adversarial review; 85 threading/override tests pass.
* **Phase 3: queued as an automatic dependency chain** — it fires the moment
  the CRM job completes, no manual step required:

| stage | PRIMARY (`implicit_flux`) | SECONDARY (`as_shipped`) | depends on |
|---|---|---|---|
| arms (a)+(b), 10-way array | 9249429 | 9249430 | `afterok` CRM 9248515 |
| merge + figures | 9249431 | 9249432 | `afterany` arms |
| arm (c) gradient, 10-way array | 9249433 | 9249434 | `afterok` merge |

* **Phase 4: NOT YET FILLED.** The two ranking tables and the tuned-parameter
  columns of the provenance table require the campaign above to finish.

> **No ranking numbers appear in this document, and none should be quoted from
> it, until this section says the Phase-3 chain has completed.** What is
> established so far is the *protocol*, the *instrument* (the kernel threading,
> the conservation gates and the four defects they exposed), and the
> *feasibility* audit — not the ranking.
