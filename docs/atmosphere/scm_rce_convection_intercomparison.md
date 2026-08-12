# SCM-RCE convection-scheme intercomparison — protocol and findings

> **2026-08-10 — THE PROTOCOL BELOW §1.1 IS SUPERSEDED for the current
> campaign.** Three things changed, and none of the numbers in this document
> predate all three, so nothing here may be compared against a new table
> without re-reading this box.
>
> 1. **The reference is no longer our own CRM.** Every arm through 2026-08-06
>    scored against `results/rcemip1_n128_ocean` — the artifact under
>    investigation, whose column water vapour was 73-81 mm against RCEMIP's
>    42.2 mm because its initial condition was supersaturated. The oracle is
>    now `results/rcemip_ref_sam300`, SAM_CRM RCE_small300 from the published
>    RCEMIP archive (Wing et al. 2018).
> 2. **The IFS/SAM saturation treatment is live.** Ice: pristine air below
>    235 K may hold ice supersaturation up to `rh_homo = 2.583 - T/207.8`,
>    withdrawn where cloud ice already exists (`thermo.
>    homogeneous_freezing_rh_factor`, on the deposition target of
>    morrison/thompson/p3, ON by default). Liquid: the in-scheme iterated
>    saturation adjustment, opt-in, now carried by every factory-reachable
>    scheme. `scripts/validate/check_ifs_supersaturation_cap.py` verifies both
>    in all three lanes (plane CRM, SCM campaign, global model).
> 3. **Microphysics is `morrison` (SAM M2005 flavour), not `kessler`.**
>    Kessler is warm-rain only, so the ice allowance is inert under it and the
>    upper troposphere is biased for every scheme alike — and the optimiser can
>    compensate for the missing ice phase by distorting convection parameters.
>    `--microphysics` and `--hard-saturation-adjustment` are now campaign
>    flags, and both are in the checkpoint signature, so a kessler run can
>    never be merged into a morrison table.
>
> The RCEMIP1 analytic initial condition also moved back to the PUBLISHED
> protocol (`T_v0 = T_sfc(1+0.608 q0) = 303.400466 K`, `Γ = 0.0067 K/m`,
> `q0 = 18.65 g/kg` at SST 300 K). The alternative calibration fitted to
> gSAM's own sounding is retained as `GSAM_SND_{T_V0,GAMMA,Q_SFC}` and must be
> selected as a triple or not at all.
>
> **Measured cost** (jobs 9356449_0/_1, bechtold + morrison + RRTMGP, CPU):
> ~139 s of compile per distinct config and ~2.2 ms/step, i.e. ~171 s per
> 100-day evaluation and ~11.4 h for a 240-evaluation scheme; peak RSS 2.4 GB
> flat across evaluations after `run_cached` began clearing the JIT caches.


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

---

## 8. The 2026-08-10 campaign — protocol, instrument, and what is still open

### 8.1 What the arm runs

`scripts/cluster/scm_rce_paper/convtune_arms.sbatch`, one array task per
convection scheme, every flag pinned in the script:

| item | value |
|---|---|
| reference | `results/rcemip_ref_sam300` (SAM_CRM RCE_small300, RCEMIP archive) |
| length / step | 100 days, `dt` = 600 s, last 5 days analysed |
| radiation | RRTMGP, `S_0` = 551.58 W/m², fixed cos(zenith) = 0.7425 |
| microphysics | `morrison` (SAM M2005 flavour) + `--hard-saturation-adjustment` |
| turbulence / GWD | Louis / none |
| surface | fixed SST 300 K, prescribed 5 m/s wind |
| large-scale forcing | **none** — no imposed subsidence; the column reaches RCE through radiation, convection and surface fluxes alone |
| convective substeps | 10 (60 s effective), identical for every scheme |
| transport kernel | `--subsidence-solve implicit_flux` (PRIMARY) |
| tuning | derivative-free, `12 × n_params` evaluations clamped to [48, 240], seed 20260810 |
| objective | `sqrt((T² + qv² + cloud² + precip²)/4)`, each profile term normalised by the reference's own mass-weighted standard deviation, precipitation by 3 mm/day |

The tuning budget is **computed from the live parameter registry**, not a
table. The reason is in the sampler: it spends its first `1 + 2·n_params`
evaluations on a deterministic one-at-a-time sweep and the rest on joint
random draws, so a flat 48 gives a 1-parameter scheme 45 joint draws and a
19-parameter scheme 7 — the high-dimensional schemes would be ranked by
optimiser under-convergence rather than by attainable fit. Acceptance is
best-so-far with a fixed seed and one uniform draw per parameter, so a
48-evaluation result is a strict prefix of a 240-evaluation one; the two
budgets are comparable without re-running (gated by
`tests/unit/test_scm_rce_tuner_sampling.py`).

Parameters whose bounds are strictly positive and span ≥ 2 decades are sampled
**geometrically**. Linear draws over, say, `[1e-6, 1e-2]` put ~90 % of the
candidates in the top decade and never visit the bottom three, which biases
every rate coefficient high by construction.

### 8.2 Confounds that survive this design

Both reviewers (Codex, GLM-5.2) were asked for these independently and agreed
on the ordering; they are stated here rather than discovered later.

1. **The matched kernel does not cover the whole field.** `sbm`, `dca`, `kuo`
   and `emanuel` have no compensating-subsidence kernel at all, so even the
   PRIMARY arm compares mass-flux schemes on `implicit_flux` against
   adjustment/mixing operators. Forcing the conservative solve also moves
   Tiedtke, Zhang-McFarlane and `mass_flux` off the kernel they ship with.
2. **Louis turbulence is not neutral.** EDMF already represents plume
   transport, so pairing it with Louis risks double-counting the boundary
   layer; trigger-sensitive mass-flux schemes inherit a boundary layer they
   were not calibrated against.
3. **Equal wall-clock is not equal tuning opportunity** even with the scaled
   budget: 240 evaluations in a 19-dimensional box is sparse, and the joint
   draws are i.i.d. uniform rather than a space-filling design.
4. **A single fixed convective substep count** (60 s) suits fast adjustment
   closures and moves slower ones away from their intended call interval.
5. **The realism/equilibrium gate is reported, not enforced**, so a scheme can
   in principle win on RMSE from an unphysical or still-drifting column. Every
   table therefore carries the drift, moist-adiabat and cold-point columns
   next to the score.
6. **One CRM, one SST.** The target imprints SAM's own microphysics, radiation
   and numerics; a scheme that resembles SAM is favoured, and the ranking need
   not generalise.

### 8.2b Two schemes have (almost) no tuning surface — MEASURED

The per-scheme budget is computed from the live registry, and the first arms
reported it: **`dca` has ZERO extended-tier tunable parameters** and `sbm` has
**two** (job 9356595, tasks 0-1). §4 of this document claims `dca` has one
(`cape_threshold`) and `kuo` two; the `dca` claim is stale.

The consequence has to be read into any ranking rather than discovered from
it: `dca`'s "tuned" column equals its "a priori" column **by construction**,
not because tuning failed to help it. A ΔRMSE of exactly zero for a scheme
with no exposed parameters is a property of the scheme's config surface, and
the `#params` / `#evals` columns are in the table so that is visible in the
same row as the number.

### 8.2c Where the cap is applied — MEASURED per scheme

Reaching the scheme and being correctly gated does not prove the allowance is
applied to the *deposition target*, which is what gSAM does. Section 2b of the
probe pins that with an equivalence: if the only effect is
`q_sat_i -> rh_homo*q_sat_i` in the deposition driving term, a cell at
`q_v = S*q_sat_i` **with** the allowance carries the same driving
supersaturation as one at `q_v = (S - rh_homo + 1)*q_sat_i` **without** it.

| scheme | residual, nucleation live | residual, `N_i0 = 0` |
|---|---:|---:|
| morrison | 237 748 % | **0.000 %** |
| thompson | 0.000 % | **0.000 %** |
| p3 | 35.670 % | **35.415 %** |

Two things follow, and the first is about the instrument. Morrison's enormous
first-column residual was the *construction*, not the physics: moving `q_v` to
the deposition-equivalent `S'` also moves the nucleation source, which is
gated on ice supersaturation (p3 names its gate: `cooper_supi_min = 0.05`,
sigmoid sharpness 200, and `S' - 1 = 0.0516` sits on it). The numbers say so
without inference — morrison's `OFF(S')` equalled thompson's answer *exactly*,
so with nucleation quiet the two schemes agree and the extra 5.35e-8 was
entirely the nucleation sink.

**CONFIRMED:** morrison and thompson reduce exactly to a deposition-target
multiplier, and are now pinned there with round-off slack only.

**OPEN:** p3's 35 % residual survives silencing Cooper, so p3 carries a second
`q_v`-dependent term at this cell that the allowance does not pass through.
Its absolute deposition is also ~6x smaller than the other two
(3.59e-12 vs 2.25e-11 kg/kg/s), i.e. a different capacitance/PSD path.
Attributing it needs p3's per-process tendencies instrumented; the cause is
UNKNOWN, not "small". p3 is therefore held only to a gross bound.

### 8.2d What the objective is actually minimising — MEASURED, and it is not T and q_v

The first four arms (job 9356595) expose two properties of the score that have
to be read with any ranking from it.

| scheme | prior | tuned | T | q_v | condensate | precip | precip [mm/d] |
|---|---:|---:|---:|---:|---:|---:|---:|
| sbm | 11.446 | 1.698 | ~0.42 | 1.065 | **3.096** | 0.798 | 3.0e-05 |
| mass_flux | 4.561 | 2.385 | — | 0.570 | **4.664** | 0.798 | 2.4e-18 |
| kuo | 4.936 | 4.936 | — | 0.373 | **9.828** | 0.798 | 6.3e-18 |
| dca | 9.459 | 9.459 | — | 0.364 | **18.897** | 0.798 | 4.2e-18 |

**1. The precipitation term is inert.** Every scheme scores exactly 0.798,
which is `2.395 / 3` — the CRM reference divided by the normalisation — i.e.
the SCM precipitates nothing (1e-18 to 3e-5 mm/day against the CRM's 2.395).
A term identical across every scheme and every tuning candidate cannot
discriminate; it adds a constant to every score. The column is nevertheless in
steady state (`drift_qv` 1e-7 to 4e-5), so water is not accumulating either —
what the campaign scores as "precipitation" is the microphysics diagnostic
alone: `run_scm_rce_campaign.py` zeroes the convective contribution whenever
convection is sub-stepped and microphysics is not `none`, to avoid
double-counting detrained condensate that the microphysics is expected to
sediment. Under the SCM's morrison configuration that sedimentation does not
reach the surface. **Cause not yet attributed** — the diagnostic, the
detrainment, and the sedimentation are all candidates.

**2. The score is dominated by CONDENSATE, not by T and q_v.** Each profile
term is normalised by the reference's own mass-weighted standard deviation,
and the CRM's condensate spread is small, so a condensate mismatch is worth
3-19 sigma while q_v is worth 0.4-1.1 and T less. The tuner is therefore
optimising the condensate profile with T and q_v as minor terms.

That matters for how a ranking is read: the deliverable ranks schemes on
temperature and humidity RMSE in physical units, but the tuning minimised a
combined score in which those two are the SMALL terms. The two need not agree,
and any "after tuning" T/q_v improvement is a by-product rather than the
target. The CSV carries all four components for both arms, so the
decomposition is checkable per scheme rather than taken on trust.

### 8.2e The precipitation was real; the DIAGNOSTIC was not — SOLVED

The inert precipitation term of 8.2d is a harness defect, found by asking why
the same physics package precipitates normally in the global model.

**The defect.** `run_scm_rce_campaign.py` reported precipitation from a
SECOND, diagnostic-only invocation of the microphysics — not the invocation
whose tendencies advanced the column — and that closure was built with the
OUTER timestep (600 s) while the applied operator runs at `dt/substeps`
(20 s). The global model reads the value from the applied call
(`physics_pipeline.py:1473`), which is exactly why it never showed this.
Diagnosed by codex (job 9361578), confirmed by direct read.

**The measurement that forced it.** Evaporation was measured by calling the
shipped `compute_surface_fluxes` on the equilibrium column the model reached,
so it is the model's own number rather than a hand estimate:

| configuration | E | storage | P before | P after | residual |
|---|---:|---:|---:|---:|---:|
| sbm | 1.559 | +0.072 | 4.1e-18 | **1.125** | 0.362 |
| mass_flux | 0.906 | −0.301 | 2.4e-18 | **0.971** | 0.236 |
| convection = none | 0.933 | −0.854 | 1.0e-07 | **1.352** | 0.435 |

76-80 % of the previously-unaccounted water is now reported, in every
configuration INCLUDING convection switched off — which is what localised the
defect to the microphysics diagnostic rather than to convective routing.

**Two suspects killed by controls, not by argument:**

* The in-scheme hard saturation adjustment. A one-variable A/B moved the sink
  by 0.042 mm/day with convection off and **0.001** with sbm. Not the sink.
  It was the leading hypothesis; it is retracted.
* Frozen precipitation being excluded from the reported flux. Morrison's
  `precipitation = precip_r + precip_i + precip_s + precip_g` — all four
  species. Refuted by reading before claiming.

**The residual (0.24-0.43 mm/day) is probably the probe, not the model.** E is
a SNAPSHOT at the final state while storage and P are WINDOW MEANS; where the
column is far from equilibrium those are not comparable, and the residual is
largest exactly there (convection=none, days 5-10) and smallest for the most
settled case. Tightening it needs E averaged over the window.

**Consequence for the ten arms.** The diagnostic is read-only — it never
entered the state — so no column's evolution was perturbed and the RANKINGS
STAND. The precipitation column in those arms is invalid and is marked so. A
re-run with the fix is not a cosmetic redo: a live precipitation term would
discriminate between schemes and would therefore change what the tuner
optimises.

**Still open, in the same lane:** the standalone convection bridge
(`convection/integration.py:670`) routes an ADJUSTMENT scheme's column drying
into `q_c` because it "has no surface-precip accumulator", while the coupled
`PhysicsPipeline` precipitates that vapour sink directly
(`physics_pipeline.py:1568`). Its own comment calls a convective-precip path
here "a tracked follow-up". That is a genuine SCM-vs-global semantic
difference for sbm/dca/kuo, separate from the defect fixed above.

### 8.2f The lowest model level is PINNED to the SST — SHF is identically zero

Found while measuring E for 8.2e: the surface-layer call returns
`SHF = 0.000 W/m^2` exactly, in every configuration, because the lowest
level's temperature is `T_a = 300.000 K` exactly — the prescribed SST.

```python
DEFAULT_SCM_RCE_BL_TOP_M = 0.0
...
z_above_lowest = jnp.maximum(z_profile - z_profile[-1], 0.0)
bl_mask = z_above_lowest <= DEFAULT_SCM_RCE_BL_TOP_M     # 0 <= 0 is TRUE
```

A boundary-layer depth of **zero** reads as "anchor disabled", and with `<`
it would be. With `<=` it anchors EXACTLY the lowest model level, which
`apply_surface_sst_anchor` then resets to `300.0 - 6.5e-3 * 0 = 300.0 K` on
every step, at both stepping bodies (`:1244`, `:1286`).

Consequences, none of them small for an RCE column:

* the sensible heat flux is **identically zero by construction** — it is
  proportional to `(T_sfc - T_a)`, and that difference is pinned at 0;
* the near-surface air temperature cannot respond to radiation, convection or
  turbulence, so the sub-cloud lapse rate is prescribed rather than simulated;
* `q_sat` at that level is therefore fixed, which constrains the near-surface
  RH and hence the evaporation measured in 8.2e (0.9-1.6 mm/day against the
  CRM's 2.23).

NOT hot-patched: the eight completed arms ran with this anchor, and changing
it would leave the published numbers describing code that no longer exists.
It belongs with the re-run that also carries the precipitation fix, where it
should be one labelled variable rather than a silent difference.

### 8.2g RESULT — all ten schemes, a priori vs tuned (job 9356595, merge 9369383)

Physical-unit, mass-weighted RMSE against SAM_CRM RCE_small300. Ordered by
tuned temperature RMSE.

| scheme | T [K] a->t | q_v [g/kg] a->t | #par | #ev | verdict |
|---|---|---|---:|---:|---|
| dca | 4.779 -> 4.779 | 1.656 -> 1.656 | 0 | 48 | unphysical |
| zhang_mcfarlane | 10.683 -> **5.847** | 4.984 -> **3.421** | 5 | 60 | unphysical |
| mass_flux | 10.158 -> **6.870** | 1.931 -> 2.592 | 5 | 60 | unphysical |
| emanuel | 7.342 -> 7.678 | 1.629 -> 2.397 | 11 | 132 | physical |
| edmf | 8.326 -> 8.305 | 1.648 -> 1.644 | 5 | 60 | unphysical |
| kuo | 9.344 -> 9.344 | 1.697 -> 1.697 | 2 | 48 | unphysical |
| bechtold | 15.415 -> **9.686** | 3.383 -> **2.174** | 19 | 228 | physical |
| kain_fritsch | 11.814 -> 12.305 | 4.093 -> 3.927 | 12 | 144 | physical |
| sbm | 15.003 -> 13.662 | 3.890 -> 4.847 | 2 | 48 | physical |
| tiedtke | 13.693 -> 13.693 | 6.321 -> 6.321 | 16 | 192 | unphysical |

Four readings that a ranking table alone would hide:

1. **Tuning made four schemes WORSE on the reported quantity.** emanuel
   7.34->7.68 K, kain_fritsch 11.81->12.31 K, sbm q_v 3.89->4.85, mass_flux
   q_v 1.93->2.59. Not a defect: the tuner minimised the combined score, which
   §8.2d shows is condensate-dominated, so it traded T and q_v away. This is
   the objective-vs-deliverable mismatch, quantified.
2. **`dca` leads on temperature with ZERO tunable parameters and is flagged
   unphysical** — it does not convect (§8.2a). "Dry convective adjustment wins"
   would be an artifact of scoring a non-convecting column.
3. **`tiedtke`: 16 parameters, 192 evaluations, no improvement at all.** Its
   optimum is its default under this objective, or the sampler cannot reach it.
4. **Absolute errors are 4.8-13.7 K.** With the lowest level pinned to the SST
   (§8.2f) and evaporation 30-60 % below the CRM's precipitation (§8.2e), these
   columns are not close to the reference, and the ranking orders schemes
   within a biased configuration rather than certifying any of them.

Largest genuine gains: bechtold (-5.7 K, 19 params) and zhang_mcfarlane
(-4.8 K, 5 params).

**Every number here carries the two harness defects of §8.2e and §8.2f.** The
precipitation column is invalid; the rankings stand because the diagnostic
never entered the state, but the physics being ranked is not yet right.

### 8.2h THE BUDGET CLOSES — and a third defect, in shared code

Co-sampled water budget, every term read from the APPLIED tendencies over the
same window (jobs 9370465/9370478):

| configuration | window | E | P | dS/dt | residual | % of E |
|---|---|---:|---:|---:|---:|---:|
| sbm | 20→25 | 1.202 | 1.125 | +0.072 | +0.005 | 0.4 % |
| mass_flux | 20→25 | 0.683 | 0.971 | −0.301 | +0.013 | 1.8 % |
| convection=none | 5→10 | 0.720 | 1.352 | −0.854 | +0.222 | 31 % |
| convection=none | **20→25** | 1.618 | 1.757 | −0.110 | −0.029 | **1.8 %** |

`d(CWV+CWC)/dt = E - P` closes to <= 1.8 % in every SETTLED configuration.
The one 31 % row is the same control at an UNSETTLED window, and the last row
is the controlled test of that: identical configuration, days 20→25 instead of
5→10, residual 31 % → 1.8 %. E and P are averaged over the last 2 days of each
run while dS/dt spans the whole interval, which is not the same window when
storage drains at 0.85 mm/day.

**The third defect, found by this measurement.** The budget read E = 0.0000 on
a column whose bulk formula gives 1.5 mm/day, twice, through two of my own
"fixes" that were downstream of the real problem: the HYDROSTATIC turbulence
path built `HydrostaticTendencies` WITHOUT `shflx_sfc`/`lhflx_sfc`. Only the
MPAS construction attached them — and its comment says they exist "for the
CMOR hfss/hfls feed", so on the hydrostatic lane that feed had nothing to read
and `evspsbl` (= lhflx / L_v) was unavailable for ANY hydrostatic run, not
just this SCM. Now exported symmetrically, None-guarded, gated by an AST test
over every construction in the module (a grep for `lhflx_sfc` passed
throughout, because the field existed — on the other lane).

**Revised, from measurement:** the earlier "evaporation 30-60 % below the CRM"
came from bulk snapshots at unsettled windows. With the applied flux at a
settled window, E = 1.62 against the CRM's 2.23 — a 27 % deficit. Still real,
smaller than stated.

### 8.2i The anchor-off arm — precipitation is real and discriminating

Second baseline (job 9376354, outdir `arm_implicit_flux_morrison_noanchor`),
differing from §8.2g in exactly two LABELLED variables: the precipitation term
reports the applied microphysics evaluation (§8.2e), and the SST anchor is off
(`--bl-anchor-top-m -1`, §8.2f) so the sensible heat flux can be nonzero. Both
are in the checkpoint signature, so the two tables cannot be merged.

First two schemes:

| scheme | prior→tuned | T | q_v | cloud | precip RMSE | P mm/d | E mm/d |
|---|---|---:|---:|---:|---:|---:|---:|
| dca | 9.314 → 9.314 | 0.210 | 0.168 | 18.625 | 0.128 | 2.779 | 2.778 |
| sbm | 11.922 → 7.645 | 0.294 | 0.578 | 15.276 | 0.028 | 2.310 | 2.317 |

Two things the numbers establish.

**The precipitation term now discriminates.** It was 1e-18 mm/day for every
scheme, giving all of them an identical error of 0.798; here dca precipitates
2.779 and sbm 2.310 against the CRM's 2.23-2.40, with errors of 0.128 and
0.028. sbm is within 3 % of the reference precipitation, and the term
separates schemes rather than adding a constant to every score.

**E ~ P in both columns** (2.778 vs 2.779; 2.317 vs 2.310), i.e. they are in
water balance at roughly the CRM's rate. The 27 % evaporation deficit measured
under the anchored configuration (§8.2h) is gone, which is what removing the
SST pin was expected to do: the near-surface air can respond again, so the
bulk flux is no longer throttled by a frozen q_a.

**DO NOT read these scores against §8.2g.** sbm is 11.92 → 7.65 here and
11.45 → 1.70 there, but the objective now contains a live precipitation term
and the column has a working sensible-heat flux. Two configurations, not a
delta.

### 8.2j kuo is INACTIVE in the SCM, and a retraction about bit-identity

**kuo contributes nothing.** Scored against a `convection=none` column under
the campaign's own configuration (job 9379420):

```
kuo    score=5.736693549527354  T=0.306305 qv=0.323231 P=1.9509 E=1.9454
none   score=5.736693549527354  T=0.306305 qv=0.323231 P=1.9509 E=1.9454
BIT-IDENTICAL: True      max|dT_profile| = 0.000e+00 K
```

This is by design, and the code says so: `convection/integration.py:435` —
Kuo needs a large-scale moisture-convergence operator, "so on a state with no
`v` (e.g. ... a single-column SCM) Kuo is deliberately OFF rather than falling
back to a proxy". Its 48 tuning evaluations were spent on a disabled scheme,
and **its row in the ranking is the no-convection baseline, not a scheme
result.**

**That baseline is itself useful:** a no-convection column scores 5.737 in the
anchor-off configuration, so any scheme scoring above it is worse than having
no convection at all. `dca` (9.314) is.

**RETRACTED — bit-identity is NOT a code-path signature.** This investigation
began from "tiedtke and kuo returned tuned scores bit-identical to their
defaults, across 192 and 48 evaluations, which cannot be weak sensitivity".
That premise is wrong. In `tune_category_winner`, `best_run` is initialised to
`default_run`, and when no candidate beats it the returned `tuned` IS that same
object. Bit-identity therefore means exactly "no candidate improved on the
default" — the expected output of an unsuccessful search.

**tiedtke needs no fix.** It is active and strong: 1.282 against the 5.737
no-convection baseline, with 14 of its 16 parameters changing the tendency in
the reachability probe. Its unmoved score is a SEARCH-BUDGET result — 192
samples over 16 dimensions — not a reachability failure. The honest response is
a larger or smarter budget, not a code change.

### 8.3 The instrument

`scripts/validate/check_ifs_supersaturation_cap.py` is the committed probe for
the saturation treatment. Its ice criterion is deliberately NOT a magnitude
threshold: measured on a 215 K, 200 hPa, RH_ice = 1.60 cell, the absolute
effect of the allowance is identical in morrison and thompson
(2.389e-10 kg/kg/s) but is 91 % of thompson's total vapour tendency and 0.44 %
of morrison's, because morrison's single-step tendency there is dominated by
Cooper nucleation — which the allowance does not gate. A relative floor would
have failed morrison for having more physics. The criterion is instead two
internal controls that no correct implementation can fail and no ignoring (or
unconditional) one can pass: above the `qci` gate, and above the 235 K ramp
cutoff, the ON and OFF configurations must be **bit-identical**.
