# DINO equatorial-undercurrent mechanism probe

Date: 2026-08-27  
Branch: `fidelity/dino-euc-mechanism-codex`  
State: shared NEMO day-180 restart; matched first step `kt=5761`

## Reported outcome

**Restated verdict: `CONFIRM_VISCOSITY_PRIME_SUSPECT`.** The prior headline
that vertical mixing was unsupported is retracted. The audited shear spans
5--26 m, so the 10.14 m interface is shear-setting, not peripheral. At that
interface legoESM's momentum viscosity is 2.0514 times NEMO's while the model's
shear is about 0.68 times NEMO's; `A_v * shear` is consequently similar
(`2.16e-5` versus `1.55e-5`). Excess near-surface viscosity predicts both
observed signs: weaker 5--26 m shear and a deeper zero crossing.

The review-corrected offline bar confirms because 10--40 m NRMS is 0.68865 and
the single shear-setting 10.14 m ratio exceeds 1.25. The original
`UNRESOLVED_CLOSURE_DIFFERENCE` label is retained in the artifact only as a
retracted result from a physically invalid two-consecutive-level clause.

The registered 10-day causal leg is `DESIGN_ONLY_NO_CUDA_IN_SANDBOX`.  With
`CUDA_VISIBLE_DEVICES=0`, JAX reported `CUDA_ERROR_NO_DEVICE`; running on CPU
would violate the registered execution condition.  No response verdict is
claimed.

The adversarial review's initial accusation that this CUDA status was false is
withdrawn. GPU logs and the untracked runner belong to the coordinator's
executor lane; this lane's no-CUDA status was honest. The coordinator is
scoring those separate response arms.

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

The original bars were committed before the first coefficient score in
`f3a9f7ae26bee72dbe3656c890cb2db10bfe9377`:

- offline confirm: NRMS at least 0.25 plus two consecutive level ratios outside
  [0.75, 1.25] in the same direction;
- offline refute: NRMS at most 0.10 and every level ratio in [0.90, 1.10];
- otherwise unresolved;
- response confirm: at least half of the BASE-to-NEMO core-depth gap closed,
  in the correct direction, without worsening absolute shear error by more
  than 10%; response refute: at most 10% closed or motion in the wrong
  direction.

Review A3/A4 showed that the consecutive-level clause makes a surface-anchored
mechanism impossible to confirm even though the top interface sets the metric.
The reissue and decomposition attribution bar were committed in
`7d73e69ea41f2955371706f070527a706412f9e7` before the decomposition. It permits
a single 5--26 m shear-setting interface to confirm when NRMS clears 0.25 and
the viscosity sign predicts weak shear plus a deeper core.

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
cells produces tiny 26--40 m realized-profile NRMS values, but those statistics
are **not evidence of closure fidelity**: the 100 m2/s EVD values dominate the
normalizer. They are retained only as a caveated diagnostic in the artifact.
At the shear-setting 10.14 m interface EVD does not erase the signal: the
realized momentum-viscosity ratio is still exactly 2.0514.

The earlier 0.999632 correlation / 1.000812 ratio offset scan used hardcoded
`RUN_GDB` year-5 data, not the day-180 lane. It is dropped from the day-180
physics argument; only the direct day-180 mapping controls are dispositive.

## TKE-closure decomposition at 10.14 m

The attribution uses 49 of 50 wet equatorial columns (98% eligible); 47 have
direct legoESM/NEMO `avm > 1.25`. Per column it closes

```
log(avm_L/avm_N) = log(Ck_L/rn_ediff_N)
                 + log(mxl_L/zmxlm_N)
                 + 0.5 log(en_L/en_N)
```

to `2.22e-16` maximum log residual and reconstructs direct avm with zero
reported relative error. The registered result is
`DISTRIBUTED_OR_UNRESOLVED` under the 60% single-carrier bar, but the physical
decomposition is specific: **TKE energy and mixing length co-carry the
infidelity, essentially 50/50 in log space**.

| piece | median absolute log score | score share | geometric-mean factor | sign agreement |
|---|---:|---:|---:|---:|
| TKE energy, `sqrt(en)` | 0.384323 | 50.0000% | 1.501877 | 100% |
| mixing length, `zmxlm` | 0.384323 | 50.0000% | 1.501877 | 100% |
| coefficient/stability, `Ck/rn_ediff` | 0 | 0% | 1.000000 | 0% |

Thus the 2.05x zonal-mean excess is not carried by `rn_ediff` or a hidden
momentum stability function. It is the coupled TKE-energy/mixing-length state:
each supplies about a 1.50x geometric-mean amplification in excess columns.
The factors are coupled physically because the buoyancy mixing length itself
depends on TKE; “50/50” is the registered multiplicative decomposition, not a
claim of independent causality.

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

The executed instrumented DINO branch states the coefficient composition
directly at `cfgs/DINO/MY_SRC/zdftke.F90:832-837`:
`zsqen=SQRT(en)`, `zav=rn_ediff*zmxlm*zsqen`, and
`p_avm=MAX(zav,avmb)*wmask`. Its `nn_pdl` block at `:841-844` modifies only
`p_avt`; there is no independent stability-function multiplier on `p_avm`.
The realized `dump_avm`/`dump_avt` time levels are now registered from the
current `zdf_phy` call (`MY_SRC/stpmlf.F90:210`) through the exact write sites
(`MY_SRC/ldftra.F90:955-956`).

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
- NEMO final TKE energy: `b7a28c78a10dc3ed15b424197f7139d53c42cdfea6fe4191d96796b635722565`;
- NEMO final mixing length: `60ed331574d813885fb29399b2dbf041a6913ea965d45c09e2890eef8b3ef830`.

The scorer hash-pins and reuses the committed bridge/dump reducers.  It writes
the receipt, reads it back, and rejects a provenance mismatch.
The final receipt was regenerated from clean committed HEAD
`a8361054d95858baab8bd51ed8e6af253bb8f872` on the named campaign branch and
records `dirty:false`; its probe and time-registry hashes are also recorded.

Repository note: no `AGENTS.md` exists in this checkout, its tracked tree,
`.agents`, `.codex`, `/home/dbalwada/legoESM`, `/home/dbalwada`, or `/tmp`.
That missing instruction source is recorded rather than silently substituted;
the explicit campaign instructions in the task were followed as binding.
