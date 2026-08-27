# DINO equatorial-undercurrent mechanism probe

Date: 2026-08-27  
Branch: `fidelity/dino-euc-mechanism-codex`  
State: shared NEMO day-180 restart; matched first step `kt=5761`

## Reported outcome

The offline matched-state comparison does **not** identify a material
vertical-mixing mismatch at the EUC core.  At the 20.6 and 31.4 m interfaces,
legoESM/NEMO closure-only momentum-viscosity ratios are respectively 0.9862
and 1.0142.  The thickness-weighted NRMS is 0.03185 over 20--40 m and 0.03238
over 26--40 m.  The large discrepancy is confined to the shallower 10.1 m
interface (ratio 2.0514).

The pre-registered 10--40 m aggregate is therefore labelled
`UNRESOLVED_CLOSURE_DIFFERENCE`: its NRMS is 0.68865, but only one, not two
consecutive, scored interfaces lies outside [0.75, 1.25].  This is deliberately
not relabelled after seeing the result.  Scientifically, the core-depth
diagnostic argues against the closure coefficient as the lead explanation for
the persistent 3.0--3.4 m EUC displacement, while leaving a testable surface-
mixing route.

The registered 10-day causal leg is `DESIGN_ONLY_NO_CUDA_IN_SANDBOX`.  With
`CUDA_VISIBLE_DEVICES=0`, JAX reported `CUDA_ERROR_NO_DEVICE`; running on CPU
would violate the registered execution condition.  No response verdict is
claimed.

## Why this was the discriminator

The regional audit found the following stable EUC signal at days 90, 180,
270, and 360:

| day | legoESM shear | NEMO shear | ratio | legoESM core/top (m) | NEMO core/top (m) |
|---:|---:|---:|---:|---:|---:|
| 90 | 0.01548 | 0.02178 | 0.711 | 34.9 | 32.0 |
| 180 | 0.01460 | 0.02147 | 0.680 | 34.5 | 31.1 |
| 270 | 0.01448 | 0.02029 | 0.714 | 33.1 | 30.1 |
| 360 | 0.01427 | 0.01983 | 0.719 | 31.6 | 28.4 |

Its prescribed matched-state test was to feed the shared state into the
shipped closure, compare the resulting coefficient profile with NEMO's dumped
coefficient, then replace only momentum viscosity in a short arm.  The audit's
wet-zonal-mean zero crossing and two-level shear definitions are retained for
the planned response readout.  Thickness weighting here uses the W-interface
control thickness `0.5*(e3t_0[k]+e3t_0[k+1])`.

## Pre-registration and controls

The bars were committed before the first coefficient score in
`f3a9f7ae26bee72dbe3656c890cb2db10bfe9377`:

- offline confirm: NRMS at least 0.25 plus two consecutive level ratios outside
  [0.75, 1.25] in the same direction;
- offline refute: NRMS at most 0.10 and every level ratio in [0.90, 1.10];
- otherwise unresolved;
- response confirm: at least half of the BASE-to-NEMO core-depth gap closed,
  in the correct direction, without worsening absolute shear error by more
  than 10%; response refute: at most 10% closed or motion in the wrong
  direction.

The controls were able to fail.  A one-interface mapping error raised closure-
`avm` NRMS from 0.68865 to 1.26861; a dry-cell coefficient plant was detected
in 86 cells; an empty wet mask raised; and disabling EVD changed the realized
tracer coefficient by 99.99897 m2/s.  The bridge independently printed exact
day-0 identity for T, S, eta, u, v, all before levels, and TKE.

## Mixing profiles

All values below are equatorial wet-zonal means in m2/s.  NEMO's closure dumps
are from `tke_dump_{avm,avt}_final.bin`; realized coefficients are from the
post-EVD `dump_{avm,avt}.bin`.

| depth (m) | lego `avm_k` | NEMO `avm_k` | ratio | lego `avt_k` | NEMO `avt_k` | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 10.139 | 1.47767e-3 | 7.20307e-4 | 2.0514 | 1.32626e-3 | 5.83845e-4 | 2.2716 |
| 20.593 | 3.00487e-4 | 3.04698e-4 | 0.9862 | 2.67725e-4 | 2.75115e-4 | 0.9731 |
| 31.429 | 4.47538e-4 | 4.41291e-4 | 1.0142 | 3.78498e-4 | 3.70527e-4 | 1.0215 |

At 31.4 m, 19.23% of equatorial columns receive 100 m2/s EVD.  Including those
cells, legoESM and NEMO realized profiles agree to thickness-weighted NRMS
`4.93e-7` over 26--40 m for momentum and `6.59e-7` for tracers.  Closure-only
profiles are the cleaner mechanism diagnostic because EVD otherwise dominates
the norm; both views agree at core depth.

For context, the campaign-wide offset control also selected the declared
alignment (`lego index 0 <-> NEMO jk=2`) uniquely: closure-only `avm` had
correlation 0.999632 and mean element ratio 1.000812 across all wet cells;
the best correlation among relative shifts -2..+2 was shift 0.

## Model paths verified

The DINO run is TKE, not the historical generic-card KPP path.  Its executed
namelist has `ln_zdftke=T`, `ln_zdfevd=T`, `nn_evdm=1`, `rn_evd=100`,
`rn_avm0=1.2e-4`, `rn_avt0=1.2e-5`, and `nn_etau=1`
(`RUN_VERDICT360_M0/namelist_cfg:384-401`); `ocean.output:731-782` confirms the
effective branch and defaults.

NEMO computes shear before dispatching TKE (`src/OCE/ZDF/zdfphy.F90:264-286`),
copies `avm_k/avt_k` to realized coefficients and then applies EVD
(`zdfphy.F90:311-323`).  The TKE path constructs the stress surface anchor,
buoyancy mixing length, top/down bounds, `avm`, and Prandtl-corrected `avt` at
`src/OCE/ZDF/zdftke.F90:575-724`.  EVD uses
`MIN(rn2,rn2b)<=-1e-12` and, for `nn_evdm=1`, sets momentum viscosity too
(`src/OCE/ZDF/zdfevd.F90:92-120`).

legoESM selects the NEMO TKE configuration in
`packages/ocean/legoesm/ocean/experiments/dino.py:2890-2941` and layers the
same enhanced-diffusion momentum option at `:3234-3259`.  The production call
and coefficient handoff to implicit mixing are in
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6347-6519`;
the closure entry is `physics/vertical_mixing/k_profiles.py:108`.  The import
receipt resolves that module inside this worktree.

## Costed short-run design

Run two 10-day, 320-step arms from the same exact bridge on GPU 0:

1. `BASE`: shipped `nemo_dino_kamm_mlf`.
2. `NEMO_AVM`: at every step, replace only the non-EVD momentum viscosity
   passed to the implicit vertical momentum solve with frozen NEMO day-180
   `tke_dump_avm_final` at jk=2..36.  Keep legoESM K_v, TKE evolution, forcing,
   all other physics, and cells where its native trigger sets K_m=100 m2/s.

Required controls before scoring are bit-identical initial arms; unchanged K_v
and TKE returned by the hook; dry/wet and vertical-shift failures; a doubled-
K_m first-step tendency plant; BASE reproduction of archived day-10 control;
and read-back/re-recording of both NPZ provenance stamps.

The archived one-GPU control reports 93 s through day 10 (including JIT) and
637 s through day 360.  Budget 93 s per separately compiled arm, about 186 s
for the pair, plus approximately 50 MiB total when retaining only day-0/day-10
3-D states and daily reducers.

## Provenance

The machine-readable receipt is `dino_euc_mechanism_artifact.json`.  Key input
SHA-256 values are:

- shared restart: `0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e`;
- NEMO seqdump executable: `545b383996e644591492e0fc7c82663a2d47272e2c7172ad8bbc58e5cd7d5239`;
- closure `avm`: `1240ccb86e674edd1309831e2292a244aaabc3bca2f245b530fd3b88b59c275f`;
- realized `avm`: `ad8104d352838ff36c1c09cb3b21231eacabeb00ce7bd2b5477cc40ead4faed0`;
- mesh mask: `3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622`.

The scorer hash-pins and reuses the committed bridge/dump reducers.  It writes
the receipt, reads it back, and rejects a provenance mismatch.

Repository note: no `AGENTS.md` exists in this checkout, its tracked tree,
`.agents`, `.codex`, `/home/dbalwada/legoESM`, `/home/dbalwada`, or `/tmp`.
That missing instruction source is recorded rather than silently substituted;
the explicit campaign instructions in the task were followed as binding.
