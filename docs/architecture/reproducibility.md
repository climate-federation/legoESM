# Experiment reproducibility: assessment and proposal

Status: **assessment, not implemented** (2026-10-12). This page describes what
legoESM records about a run today, what is lost, and what is the smallest set
of changes that would close the gaps. Nothing here has been built yet.
Each item that changes behaviour is listed under [Decisions](#decisions);
all were approved on 2026-10-12.

Target principle:

> Every legoESM simulation should leave behind a self-describing record of the
> resolved scientific configuration, software versions, input data, and
> execution context needed to understand and reconstruct it.

## Four kinds of reproducibility

These are different guarantees. Recording the configuration does **not**
guarantee identical numbers.

| Kind | Question it answers | What it needs | Status today |
|---|---|---|---|
| 1. Configuration | *What* was run? | Fully resolved scientific configuration, with every default written out | Partial: driver-level fields only; physics-scheme parameters are not recorded (G2) |
| 2. Software / environment | *With what code* was it run? | Commit, uncommitted diff, lockfile, jaxlib/CUDA, environment variables, devices | Partial: commit, dirty flag, and python/jax/numpy versions only (G5) |
| 3. Bitwise numerical | Is the rerun *bit-identical*? | 1 + 2, plus the same hardware, device count and decomposition, XLA version and flags, and deterministic kernels | `legoesm reproduce --check` compares the **final-state digest only**, **atmosphere only**, on whatever code is checked out now (G1) |
| 4. Scientific / statistical | Does a rerun give the *same climate*? | 1 + 2, an ensemble, and a metric with a tolerance (e.g. ClimateEval scores) | No mechanism (the release matrix, item 11, is the proposed step toward it) |

Even with identical software and hardware, bitwise equality is not
automatic: GPU atomics and scatter reductions can be non-deterministic. A
change of GPU model, device count, domain decomposition or XLA/jaxlib version
usually changes reduction order and fusion, so the realistic target across
those changes is level 4. A record that satisfies levels 1 and 2 tells you
*which* of these changed. It does not promote one level to another.

## 1. Current state

### Three ways to configure a run

1. **`legoesm run <yaml>`** (atmosphere and idealized cases).
   `src/legoesm/config.py:Config.from_yaml` deep-merges the user YAML over
   `DEFAULT_CONFIG`, rejecting unknown keys (`_require_known_keys`).
   `Config.to_experiment_config` then maps the result onto `ExperimentConfig`
   (`packages/coupler/legoesm/driver/config.py`). `ExperimentConfig` is a flat
   NamedTuple of about 350 fields, with three nested sub-configs: `GridConfig`,
   `DycoreConfig` and `OutputConfig`.
2. **Driver command line plus `--config`** (`run_amip.py`, `run_omip.py`,
   `run_coupled.py`, `run_lmip.py`).
   `driver/run_config_yaml.py:load_yaml_config` turns the YAML into argparse
   defaults, so the precedence is command line > file > parser default.
   `read_yaml_with_includes` merges a recursive `include:` chain, with cycle
   detection.
   `--require-config` forbids running on bare parser defaults.
   `--params` (`load_params_config`, `apply_params_to_config`) overrides
   physics-scheme parameters and checks each against the scheme's
   `__param_spec__` bounds.
3. **Templates.** These live in `config/templates/<category>/*.yaml`, each
   with an `experiment:` metadata block.
   `scripts/experiment/init_experiment.py` takes a template, dotted `-o`
   overrides and a machine profile. It runs `validate_strict`, then writes
   `config.yaml`, `run.yaml` (template, overrides, machine, creation time) and
   `run.sh`.
   A second, unrelated `scripts/run/init_experiment.py` does the same for
   land/LMIP and writes an INI `experiment.tag` instead.

Machine profiles (`config/machines/*.yaml`, resolved by
`scripts/experiment/lego_detect_machine.resolve_machine`) are separate from
the science, with one exception: `init_experiment` copies the machine's
`precision` into the run's numerical precision, which is a scientific choice.

### What a run records today

- **`run_manifest.json`.** Built by `driver/restart.py:build_run_manifest` and
  written atomically by `write_run_manifest`. The schema version is
  `RUN_MANIFEST_SCHEMA_VERSION = 1`. It contains:
  - `legoESM`: ref and commit of the *imported* package
    (`io/git_provenance.py:git_provenance`).
  - `reproducibility`: runner tag, python/jax/numpy/legoesm versions, x64
    flag, platform, `git_dirty` (a boolean), and patch names.
  - `config`: `resolved_config`, `config_hash` (SHA-256 of the sorted JSON),
    and `params_applied`.
  - `result`: `state_digest`, RNG seeds, dataset provenance (path, size and
    mtime; sha256 is optional and never passed), and model-weights provenance.
  - `run`: argv and creation time.
- **`legoesm reproduce [--check]`** (`src/legoesm/cli.py:cmd_reproduce`).
  It rebuilds `ExperimentConfig` from `resolved_config`, reruns through
  `ModelDriver`, and compares `state_digest`. Ocean manifests are validated
  but not rerun.
- **Restart metadata** (`driver/restart.py:RestartMetadata`): config hash,
  state digest, git hash and versions. `verify_reproducibility` compares two
  restarts.
- **Who writes a manifest:**
  - `ModelDriver` (`_write_run_manifest`), which covers `legoesm run`,
    `run_amip.py`, `run_aimip_latlon.py` and `reproduce`.
  - `run_omip_core2.py`, through `ocean/config.py:OceanRunRecord`. A failed
    write there only produces a warning.
  - `run_fesom_core2.py`, which writes its own JSON in a different format.
  - **No manifest:** `run_omip.py`, `run_lmip.py`, `run_coupled.py`,
    `run_dino.py`, and the test-matrix runners. The matrix runners record only
    commit and python version, in CSV rows.

### legoESM-ocean-runners (separate repository)

legoESM-ocean-runners is an experiment layer that sits on top of legoESM.
It is worth keeping, not replacing. It provides:

- `legoESM_version.txt`, which pins the legoESM commit.
- Validated `legoesm-<hash>-<date>` git tags, enforced by a CI job
  (`repro-tag-check`).
- Numbered `patches/` holding changes not yet merged upstream.
- A `data_catalog.yaml` and `scripts/fetch_data.py`.
- Machine profiles that generate scheduler-specific `run.sh` files
  (slurm/oar/pbs/lsf).
- A per-experiment INI `experiment.tag`, written by
  `runners/run_experiment.py:generate_tag`.
- `scripts/reproduce.py`, which checks out the runner tag, checks out the
  legoESM commit, applies the patches and checks the data.

The runners repository does the job of a CIME case or an ESM-Tools setup:
it chooses a model version and records what was asked for. legoESM's
manifest records what the model actually did. **The two records are not
linked today** (gap G7).

## 2. Gaps

Each gap is CONFIRMED by reading the code at commit `207686288`, unless
marked otherwise.

- **G1. Old configurations can silently take on new meaning.**
  Rebuilding a stored configuration (`experiment_config_from_dict`,
  `ocean_config_from_dict`) drops fields that no longer exist and fills
  missing fields from *today's* defaults. `restart.py:config_hash_matches`
  goes further and accepts a stored manifest as identical whenever every
  newly added field sits at its current default.
  The `hines_launch_p` case (gravity waves launched at the surface because
  the fix's default kept the bug) is the shape this lets through: if a new
  field's default is not the old behaviour, the old run is "matched" under
  new physics.
  Separately, `legoesm reproduce` never compares the recorded commit with the
  checked-out one, so it reruns old manifests on new code without warning.
  The same applies to restart-chain continuation, which only checks
  `config_hash_matches`.
  Code changes that touch no config field at all (bug fixes, new lookup
  tables, changed constants, kernel rewrites) are invisible to every
  config-level check. Only the commit records them.
- **G2. Physics parameters are not in the record.** `resolved_config` is
  `ExperimentConfig` alone. Physics-scheme configurations are rebuilt from
  code defaults at run time and only partly overridden. For example,
  `driver/physics_pipeline.py:_resolve_turbulence` builds
  `GravityWaveDragConfig(scheme=scheme)`. A change to any scheme default
  (Morrison, CLUBB, Bechtold, GWD, land, …) therefore changes the model
  without changing `config_hash`. Only the `--params` overrides are recorded
  (`params_applied`).
  Two further losses:
  - `config_to_dict` deliberately writes `turbulence_override` as `None`, so a
    run driven by a full turbulence override loses it.
  - Array-valued ocean config leaves are stored as summaries
    (`ocean/config.py`, e.g. the runoff depth map), which cannot be rebuilt
    from the manifest.
- **G3. Manifest coverage is incomplete.** OMIP (non-CORE2), LMIP, coupled,
  DINO and the test-matrix drivers write no manifest. FESOM writes an
  incompatible format. The ocean CORE2 write is best-effort.
- **G4. The fingerprint mixes science and execution.**
  - `config_hash` covers execution-only fields: `output.output_dir`,
    `n_devices`, `distributed_mode`, `max_wallclock_seconds`. The same
    science run in another directory, or on a different number of GPUs, gets
    a different hash.
  - The runners' `config_hash` hashes the template after `-o` overrides,
    before legoESM resolves defaults, so it can never be compared with the
    upstream hash.
- **G5. Environment is thin.**
  - Not recorded: the `uv.lock` hash, the jaxlib / CUDA-plugin / equinox /
    optax / mpi4jax versions, `XLA_FLAGS`, `JAX_*` and `NCCL_*` variables,
    the device model and count, MPI size, hostname, and machine profile.
  - `git_dirty` is a boolean, so the uncommitted diff itself is lost.
- **G6. Input data is identified by path only.**
  - Both `fetch_data.py` verifiers already honour a `sha256` field, but no
    entry has a usable value. Upstream `config/data_catalog.yaml` has none;
    the runners' 25 entries are all `sha256: pending`.
  - The manifest records path, size and mtime. A re-staged or regridded
    forcing file under the same path is invisible.
- **G7. The input record and the result record are not linked.**
  - The template name, the overrides and the original YAML text (including
    the `include:` chain) are not in the manifest. Only argv is, and argv
    holds the file path, not its content. `init_experiment`'s `run.yaml`
    sits beside the manifest, not inside it.
  - The runners' `experiment.tag` does not point at the manifest.
  - Patches are recorded by name, not by content. A deleted patch makes an
    old tag reproduce silently unpatched.
- **G8. Outputs do not point back to their run.** CMOR files
  (`io/cmor_output.py`) carry the standard CMIP global attributes, including
  a fresh `tracking_id`. They carry no legoESM commit, config hash or
  manifest reference, so a copied file loses its link to the run that made
  it.
- **G9. Duplicate machinery.** There are two `init_experiment.py` scripts,
  two data catalogs with different schemas, and two tag/manifest formats.
  Each new provenance feature would have to be built twice.

## 3. Minimal proposal

Extend `run_manifest.json`. **Do not add a new framework or a second file
format.** The manifest is already schema-versioned, written atomically,
consumed by `reproduce`, and covers the most-used drivers.

The layout suggested in the task maps onto what exists today:

```
experiment/
├── config.yaml          # user input (init_experiment already writes it)
└── <output>/
    ├── config_input/    # NEW: verbatim copy of every YAML read, incl. include: chain
    └── run_manifest.json
          ├── config.resolved_config   ≈ resolved_config.yaml (now incl. physics schemes, G2)
          ├── config.science_hash      # NEW fingerprint (G4)
          └── legoESM / reproducibility / environment / inputs / run   ≈ manifest.yaml
```

Splitting the manifest into two YAML files would force a change in every
existing reader (`reproduce`, `read_run_manifest`/`validate_run_manifest`,
the restart-chain check) and buy nothing. If a human-readable view is wanted,
`legoesm manifest show <run_manifest.json>` can print the resolved
configuration as YAML from the one file.

The proposal, change by change:

1. **Record the resolved physics** (fixes G2; protects *future* manifests
   against G1).
   - After *all* resolution (`--params`, surface-flux config, per-lane
     resolvers), serialise every `*Config` actually used into
     `config.resolved_physics`, including `turbulence_override`.
   - Replay **builds** each `*Config` from the record and fails on a missing,
     extra or renamed field. It does not overlay the record on today's
     defaults: an overlay would silently fill new fields from the new code.
     There is no existing loader that does this. `--params` routing only
     covers registered tunables, so this is new code.
   - Closure check at run start: re-resolve from the record and compare with
     it. That detects resolver drift cheaply.
   - Replayed values land in the terminal lane. Both the scalar fields and
     any full override (`turbulence_override`, `gravity_wave_drag_override`)
     come from the record, so a rebuilt override cannot mask them. Composite
     schemes (e.g. `mcfarlane+hines`) are replayed per component.
     `validate_strict` passes, or the replay fails loudly.
   - `__param_spec__` bounds are check-only on replay: an out-of-bounds value
     is refused, never clamped.
   - Ocean array leaves are saved as arrays beside the manifest, with their
     sha256, not as summaries. Their source file alone is not enough, because
     the array is derived from it.
   - Limits:
     - Manifests written before this change have no physics record and stay
       exposed. For those, only the commit check (item 2) helps.
     - Replay pins *values*, not *meaning*. A field whose units,
       coordinate or interpretation changed is replayed with the old number
       and new semantics. The `hines_launch_p` fix had exactly this shape.
2. **Make version drift loud** (fixes G1).
   - `reproduce` compares the recorded commit **and code-delta sha256**
     (dirty diff plus applied patches, item 5) with the imported code. Either
     differing counts as drift.
   - On mismatch it prints a field-by-field diff table: the stored resolved
     configuration against today's code's reconstruction of it (fields
     dropped, fields filled from new defaults). It refuses unless `--allow-version-drift` is given.
   - Restart-chain continuation applies the same commit and code-delta
     check.
     `config_hash_matches` keeps its schema-growth tolerance, but logs which
     new fields it accepted. Tolerance alone cannot tell whether a new
     field's default equals the old behaviour; only the commit can.
   - A missing commit (non-git install) counts as a mismatch.
   - The drift table compares configuration only. An *empty* table on a
     commit mismatch does **not** mean the same science, because code changed.
     The message must say so.
3. **One manifest writer for every production driver** (fixes G3).
   - `run_omip.py`, `run_lmip.py`, `run_coupled.py` and `run_dino.py` call
     `write_run_manifest` (P0).
   - Later, `run_fesom_core2.py` and the matrix runners (once per case) do the
     same (P2).
   - A coupled run writes **one** manifest. Each component's resolved
     configuration is a named sub-block, so one `science_hash` covers the
     coupled system.
4. **A science fingerprint** (fixes G4).
   - Add `science_hash`: SHA-256 over the resolved configuration plus resolved
     physics, *excluding* a named, tested list of execution-only fields
     (output paths, device count, distribution mode, wallclock limits).
   - Precision, x64 and RNG seeds are science and are included.
   - Restart and diagnostic cadence need an explicit classification
     (decision below).
   - Keep `config_hash` unchanged, so old manifests still verify.
   - Equal `science_hash` means equal *configuration* only. The citable run
     identity is **(science_hash, legoESM commit, code-delta sha256, input
     sha256s)**, where the code delta is the dirty diff plus applied patches
     (empty for a clean, unpatched tree). All four will be in the manifest. This identifies the experiment, not a bitwise
     trajectory: execution choices excluded from the hash, such as restart
     cadence, can still change the numbers.
   - The execution-only exclusion list is under the same golden-file gate as
     item 8, so a science field cannot be moved into it quietly.
   - A new config field enters the hash at its default, so the hash changes
     across versions even when the science does not. The manifest therefore
     stores the list of hashed field names, and comparisons restrict to it,
     the same policy as `_restrict_to_stored_shape`.
5. **Environment block** (fixes G5).
   - Add the `uv.lock` sha256, the installed versions of the
     numerics-relevant distributions (`importlib.metadata`), the `XLA_FLAGS`,
     `JAX_*`, `NCCL_*` and `LEGOESM_*` environment variables, device kind and
     count, MPI size, hostname, and machine-profile name.
   - When the tree is dirty, save `git diff HEAD` and a copy of the untracked
     `.py` files under the package roots beside the manifest, and record
     their sha256.
6. **Input checksums** (fixes G6).
   - Populate `sha256` in the catalogs. The verifiers already check it.
   - Have the manifest record sha256 for each input, using the existing
     `dataset_provenance_entry(sha256=...)`.
   - A cache keyed on (path, size, mtime) avoids re-hashing large files, but
     can be fooled by an in-place rewrite that keeps size and mtime. Prefer
     the catalog's sha256 once `fetch_data` has verified the file.
   - Directory inputs (Zarr stores, mesh directories) need a manifest of
     per-file hashes.
   - Audit that every forcing, IC, orography and mesh read is routed through
     the provenance helper. Otherwise the record is silently partial.
7. **Close the record** (fixes G7).
   - Copy each YAML the driver read into `config_input/`.
   - Fold `run.yaml` (template, overrides, machine) into the manifest's `run`
     block.
   - Let the runners pass `runner_tag` and `patches` (name plus sha256);
     the manifest fields already exist.
8. **A default-snapshot gate** (prevents G1 from recurring).
   - A committed golden file lists the default of every field of every
     `*Config` NamedTuple, and a test fails when any default changes.
   - A default change then shows up as a one-line diff in review and needs
     a CHANGELOG "default changes" entry.
   - The golden file also covers `DEFAULT_CONFIG` in `src/legoesm/config.py`
     and the `ExperimentConfig` defaults. Module-level constants and tables
     stay uncovered; for those only the commit helps.
   - This is the mechanical gate that CLAUDE.md RULE 3 currently enforces by
     honour system.

9. **Provenance in outputs** (fixes G8).
   - Add `science_hash`, `legoesm_commit`, the manifest sha256 and the
     template id to the existing CMOR global attributes, and record the run's
     `tracking_id`s in the manifest, so a copied file still identifies its
     run.
10. **A nightly reproduce test** (keeps level 3 honest).
    - A CIME-ERP-style job reruns one short canonical case on fixed hardware
      and compares `state_digest`.
    - Without it, the bitwise claim is untested between releases.
11. **A release matrix with statistical bands** (a step toward level 4;
    modelled on JCM's `tools/release_validation/`,
    climate-analytics-lab/jax-gcm).
    - **Supported set.** One committed `matrix.yaml` names the configurations
      legoESM claims to support. Each member points at an existing template
      or production config, not a composition written for the test. The
      template `maturity` field records run and validation status, not a
      support claim.
    - **Release check.** Before a release tag, each member runs a long window.
      Its statistics are compared with a fixed reference climate, not only
      with the previous release, so that small changes cannot accumulate into
      drift across releases. This is regression detection between code
      versions. Separately they are scored against loose
      climatological health gates (plausibility) and NaN and conservation
      gates. Known failures are declared in the matrix with an owner and an
      issue, not hidden.
    - **Fast regression.** Each member also has a short run from a stored
      warm state. Area-weighted global means of a named variable set are
      compared with committed band files, stored as text so a change shows up
      as a reviewable diff.
    - **Band derivation.** The band width must exceed the run-to-run spread
      that roundoff and hardware differences cause in a chaotic model.
      Proposal: derive it from a small perturbed-initial-condition ensemble.
      That measures internal variability on one machine. It says nothing
      about how far runs on other hardware spread, so the bands are set per
      hardware type unless a cross-hardware ensemble shows they can be
      shared. Without an ensemble a band is a guess. The fast regression is
      the first arm of level 4. It does not answer "same climate" on its own,
      which needs the long window.
    - **Provenance.** The warm state is named by its content digest. Each band
      file records the state's own manifest (code, configuration, inputs) and
      the input-data revision of the band run. A mismatch is reported as
      "regenerate the bands", not as a physics regression.
    - **Limits.**
      - Global means miss regressions that keep the global mean, such as
        spatial patterns or polar processes.
      - A short window sees fast processes only, so ocean members need a
        longer or separate window.
      - Bands regenerated in the same PR as an intended change guard only
        later PRs, not that change itself.
      - Full level 4 also needs levels 1 and 2, i.e. P0 and P1.

How this relates to established practice:

- **CESM/CIME**: a case directory whose resolved XML is the record.
- **ESM-Tools**: writes out the fully merged ("finished") configuration of
  each run.
- **ESMValTool**: keeps the filled recipe plus a provenance record per
  output.
- **CMIP**: global attributes identifying source and experiment.

Items 1, 4, 7 and 9 are the legoESM equivalents of these. Machine description
stays outside the scientific configuration, as in ESM-Tools and CIME.

## 4. Responsibility split

| Concern | legoESM core | Experiment repos (legoESM-ocean-runners, ledgers) |
|---|---|---|
| Manifest schema, writer, `reproduce` | owns | consumes; links `experiment.tag` → `run_manifest.json` |
| Resolved config + physics, `science_hash` | owns | quotes it; drops its own template `config_hash` (or renames it `template_hash`) |
| Environment capture | owns (records what is running) | owns *rebuilding* it: pin, patches, `uv.lock` checkout |
| Default-change gate + CHANGELOG | owns | reads CHANGELOG at pin bumps |
| Data checksum *mechanism* | owns (`dataset_provenance_entry`, verifier) | owns catalog *content* for its experiments (URLs, sha256, staging per machine) |
| Templates, overrides, machine profiles, `run.sh` | owns generic ones (`config/templates`, `config/machines`) | owns experiment-specific ones; passes template id + overrides in |
| Pre-merge patches | — | owns; records name **and sha256** of each applied patch |
| Experiment ledger (what was run, why, results) | — | owns |

Rule of thumb: legoESM records **what actually ran**. Experiment repositories
record **what was intended, where, and how to rebuild the environment**. The
link between the two is a pair of fields (`runner_tag`, `patches`) that the
manifest already has.

## 5. Implementation plan

Each step is a separate small PR. Schema policy: new keys are **additive**
under `schema_version` 1, and readers ignore keys they do not know.
`validate_run_manifest` rejects every version other than 1 today. So if a
bump is ever needed, the reader must accept v1 and v2 *before* any writer
emits v2. Old manifests keep loading.

**P0: the record is complete and drift is loud**
1. `config.resolved_physics` is written, and `reproduce` replays it (item 1).
   Start with the atmosphere pipeline, then ocean.
   **Implemented for the atmosphere (`ModelDriver`).** Each lane records
   the configs its kernels actually run with: convection, microphysics,
   turbulence (with its surface layer), gravity-wave drag, radiation (the
   standalone lanes' `RadiationConfig` carries clouds, ozone and orbit; the
   pipeline lanes record their radiation scheme config plus the cloud knobs
   when a cloud scheme is on), the multilayer land config, the snow-albedo
   config when that feedback is on, and the solar, ozone, aerosol and GHG
   forcing configs. `turbulence_override` and `gravity_wave_drag_override` are
   recorded too. An entry the lane does not run is recorded as `null`, e.g.
   the idealized FV3-duo / simple tiled lanes record only the Kessler config
   they run, and the dry spectral path records its default gray radiation
   and orbit. The pipeline cloud entry is the `CloudConfig` the radiation path
   builds; its convective-cloud term is recorded on only where the lane passes
   the lagged convective precip (compiled and operator-split lanes, not the
   per-step lane).
   Pipeline lanes record at the start of `run()`; the MPAS / column,
   spectral, FV3-duo, lat-band and tiled-cube lanes record their own build.
   Process 0's resolution is recorded and checked; the outcome is broadcast
   to every process.
   `reproduce` refuses a manifest whose `resolved_config` today's code does
   not rebuild exactly, or that has no complete physics record, even with
   `--allow-version-drift`. It rebuilds the two overrides strictly from the
   record and runs; the rerun refuses before stepping if its resolved physics
   differs from the record (closure check). A resume whose resolved physics
   differs from the manifest's record is refused.
   Replay verifies rather than injects: apart from the two overrides, values
   come from today's resolvers, and a difference refuses the run.
   Not recorded: decisions made inside the physics step at trace time (e.g.
   Bechtold's diurnal CAPE correction is switched off when no surface-flux
   source exists); per-process differences (only process 0's resolution is
   recorded); array-valued overrides (stored as a digest, so they cannot be
   replayed).
   On pipeline lanes the record also holds the rollout (`compiled` or
   `per_step`); `reproduce` reruns in that rollout and reapplies the recorded
   convection / microphysics / turbulence / GWD configs after `setup()`, since
   `run_amip.py` may replace them there (`--params` routing, AIMIP trained
   configs, Sundqvist command-line overrides). This is the one place replay
   injects; the closure check still compares the whole record afterwards.
   **Component runners** (implemented for land `run_lmip.py`, ocean
   `run_omip.py` and DINO `run_dino.py`; coupled follows; CORE2 is validated
   only). Replay of `run_omip.py` always sets `--grid` to the manifest's grid
   (`--grid all` writes one manifest per grid); `run_dino.py --timing-jsonl`
   is pointed at the same file name in the rerun's output. `--diag-every`
   is not an OMIP continuation control (it also sets the JRA55 forcing block
   size), nor is `--quick` (it changes other defaults). The `run_omip.py` digest covers the
   final sea ice when the run carries it. A runner builds its configuration through script code,
   so `reproduce` reruns the recording script: the manifest records the exact
   command line (`run.argv`), the launch directory (`run.cwd`) and the process
   count (`reproducibility.n_processes`, taken when the manifest is written,
   so a runner writes it after its process setup). `reproduce` refuses on version drift
   (unless allowed), on a missing command line or launch directory, on a
   multi-process run (it prints the command, to be launched on the recorded
   process count with the original launcher, which is not recorded), on a
   manifest without a process count, on a run launched through anything but
   the runner script itself or whose recorded script content
   (`run.script_sha256`) is missing or matches neither the recorded script
   nor this checkout's copy (it reruns the recorded script when it is still
   there unchanged), when this checkout has no `scripts/run/` runner, on
   `--check` when the reference recorded no final digest, and on an
   output directory overlapping the reference run's or already holding a
   manifest.
   Runner manifests without replay support are validated only, as before. It reruns
   with only the output flag rewritten and with `LEGOESM_REPRODUCE_REFERENCE`
   / `LEGOESM_REPRODUCE_OUTPUT` set; the rerun's manifest writer then refuses
   before stepping unless the rerun resolves to the reference record
   (resolved configuration minus the output path, applied `--params`, resolved
   physics, and each input's path and size). `--check` compares final digests;
   without it, a rerun that records no final digest did not complete and
   fails. Without `--output` the rerun goes to a new temporary directory,
   which is kept (also after a MATCH) and named in the log. The interpreter, its flags and the environment (`JAX_*`,
   `XLA_FLAGS`, ...) are not recorded or compared yet (P1 environment block),
   so a MATCH is not evidence of environment independence. A run started
   through a wrapper that keeps `argv[0]`, or as `python -m`, is rerun as the
   plain script.
   A run into a directory that already holds a run's manifest is a
   continuation: besides the version-drift rule it may change only its
   continuation controls (resume/restart flags, run length, output and
   checkpoint frequency, wallclock budget, output path); any other difference
   is refused. Manifests written before `run.argv` existed get only the
   version-drift rule, and so do directories where neither run's continuation
   controls are classified (runners without replay support); a classified run
   on either side refuses a change of runner. An existing manifest that
   can be read but not parsed or validated counts as unknown code: it is
   refused, and
   with the drift override it is archived and replaced unchecked. Restart files carry no code
   identity of their own: the drift gate trusts the manifest found beside the
   restart (or one level above it, for `run_omip.py` and `run_dino.py`).
2. Commit check plus a drift diff table in `reproduce` and restart-chain
   continuation, and reporting from `config_hash_matches` (item 2). Until the
   code delta is captured (P1 step 6), a recorded or current `git_dirty` (a
   field the manifest already has) or any applied patch counts as drift.
   **Implemented** for `legoesm reproduce`, `ModelDriver` resume and
   `load_checkpoint`, and reruns through `write_run_manifest_on_root`. To
   proceed anyway, pass `legoesm reproduce --allow-version-drift` or set
   `LEGOESM_ALLOW_VERSION_DRIFT=1` (any driver). A checkpoint needs the
   writing run's `run_manifest.json` beside it. `run_lmip.py --restart-from`
   is gated the same way, before the run writes its own manifest;
   `run_omip.py --restart` and `run_dino.py --restart-from` look for the
   manifest beside the file or one level above (`restarts/`, `snapshots/`),
   before loading it. `run_omip_core2.py --restart-from` refuses a restart
   archive whose recorded source revision is unknown, different, dirty or
   mixed-tree (it was a warning before).
3. The manifest is written by `run_omip.py`, `run_lmip.py`, `run_coupled.py`
   and `run_dino.py` (item 3).
4. Docs: correct the README claim that every run is reproducible
   bit-for-bit (done alongside this page).

**P1: identity, inputs, link to experiment repos**
5. `science_hash`, with a tested list of execution-only fields (item 4).
6. Environment block plus the saved dirty diff (item 5).
7. Populate catalog sha256 and hash dataset provenance (item 6). Merge the
   two catalog schemas *first*, so the column is added once (G9).
8. Default-snapshot golden test plus the CHANGELOG convention (item 8).
9. `config_input/` copy, and `run.yaml` folded into the manifest (item 7).
10. Runners write `runner_tag` and patch sha256 into the upstream manifest,
    and reference the manifest from `experiment.tag`. This is a two-field
    change on each side.
11. Provenance attributes in CMOR outputs (item 9).

**P2: consolidation and hardening**
12. Merge the two `init_experiment.py` scripts (G9).
13. FESOM and the matrix runners move to the standard manifest.
14. Nightly reproduce test (item 10).
15. Release matrix with statistical bands (item 11). The direction was
    approved on 2026-10-12, to start once P0 is merged. Its parameters are
    still pending (listed below).
16. Optional: `legoesm manifest show` and `legoesm manifest diff A B`, which
    prints the science-field diff table between two runs.

## Decisions

These items change existing behaviour. Under the repository's
no-unasked-choices rule each needed an explicit yes or no. **All were
approved as proposed by the maintainer on 2026-10-12.** None is implemented
yet.

- **Manifest write failure.** Today a failed write in ocean CORE2 is a
  warning. Proposal: make it a hard error in every production driver.
- **`reproduce` on a different commit.** Today it reruns silently.
  Proposal: refuse unless `--allow-version-drift` is given.
- **`init_experiment` copying machine `precision` into the science config.**
  Proposal: keep it, but record it as a scientific field so it enters
  `science_hash`. The alternative is to require precision in the template.
- **Default-snapshot gate.** Proposal: a new CI test that fails on *any*
  `*Config` default change until the golden file is updated.
- **Hashing large inputs every run.** Proposal: on by default, trusting a
  catalog-verified sha256 where one exists. The alternative is opt-in only.
- **Restart-chain continuation on a different commit.** Today it is accepted
  when `config_hash_matches` passes. Proposal: refuse unless the drift flag
  is given, the same rule as `reproduce`.
- **Restart and diagnostic cadence in `science_hash`.** Restart cadence can
  perturb a trajectory through checkpoint round-trips. Diagnostic cadence
  changes which samples exist. Proposal: exclude both, and record them in
  the manifest's execution block.
- **Strict replay.** Proposal: `reproduce` fails on any field missing from or
  extra to the record, for both the top-level configuration and the physics
  records, instead of today's silent drop or fill.

### Pending decisions for the release matrix (item 11)

These are not yet decided:

- Which configurations are members, for example AMIP res6 MPAS, OMIP lat-lon
  KPP, Held-Suarez, and an SCM case.
- The banded variable set and the health-gate ranges.
- The release-check metric and tolerance against the reference climate, and
  how that reference is chosen and renewed.
- The warm-state generation recipe (spin-up length, initial state).
- Whether declared expected failures expire.
- The run lengths for each component (release check and fast regression).
- The band derivation: ensemble size, and whether bands are set per hardware
  type or shared.
- Whether a failed release check blocks the tag, or is reported only.
- The warm-state hosting location and how its revision is pinned.
- Whether the fast regression runs in CI on a GPU or only before releases.
- How ClimateEval is wired in: it is optional and AMIP-focused today.
