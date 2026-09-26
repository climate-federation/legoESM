# NEMO testcase lane 1: preregistered oracle dossier

Status: **PREREGISTERED — no receipt run had been started when this file was
committed.**  This is phase 1 only: certify NEMO 5.0.2 oracle geometry,
configuration, execution, and step-entry state.  It makes no legoESM match or
fidelity claim.

## Immutable provenance and controls

| item | pinned value |
|---|---|
| NEMO tree | `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2` |
| NEMO commit | `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796` |
| architecture | `arch-conda.fcm`, SHA256 `64cf1b90f611936bbb92a7514800f8a8e9c8365be7b3cf9a9963f1615c5c836a` |
| arithmetic | `REAL(wp)` compiled with `-fdefault-real-8`; the trajectory header must additionally report `STORAGE_SIZE(1._wp)=64` |
| DINO binary immutability control | SHA256 `00bc1bf78167b6c46c470533955ac39f6e285acb7730dc54a0ef76c770c5373a` |
| DINO `MY_SRC` aggregate control | SHA256 `8b78ad0f12726f689e95e46f7241af000c4eca5f2f8a5f112a2c570255bc2930` |
| execution | one CPU process, no GPU, no `mpirun`; run roots below `/data/abyssal/dbalwada/nemo-testcases-l1/` |

The NEMO source checkout was already dirty with DINO and toolchain artifacts.
This lane modifies only new configuration names beginning `OVERFLOW_OMIP_L1`
or `LOCK_EXCHANGE_OMIP_L1`.  The certified `cfgs/DINO` tree and executable are
read-only controls and their hashes are reconciled at exit.

## Source dossier

### OVERFLOW

The domain is a closed, nonrotating, three-row channel, 200 km long and 2000 m
deep.  With the receipt resolution `rn_dx=1000 m`, `rn_dz=20 m`, the source
sets `jpi=202`, `jpj=3`, `jpk=101`.  Bathymetry rises from 2000 m in the east
to a 500 m western shelf through a tanh slope centred at 40 km with 7 km width
(`tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:90-103`).  The model begins at rest,
with uniform salinity 35 and a 10 C cold reservoir west of 20 km against 20 C
water to the east; surface fluxes are identically zero.  The documented
phenomenology is a dense gravity current descending the slope, with numerical
mixing sensitive to tracer advection, vertical coordinate, and bottom
parameterisation (`tests/README.rst:127-143`).  The shipped S-EOS density
contrast is bring-up documentation only; the receipt uses TEOS-10 and measures
the resulting density/BN2 fields rather than carrying that number across EOSes.

The source implements zco (`nn_COORD=0`), zps (`1`), and sco (`2`) branches
(`usrdef_zgr.F90:71-85`).  This lane runs **zps and sco**.  zps is the primary
receipt: the bottom U/V/F thicknesses equal T thickness because thickness
increases monotonically in i and is invariant in j
(`usrdef_zgr.F90:177-187`).  Thus it exercises the exact local specialization
of the neighboring-cell face minimum.  sco is the controlled vertical-coordinate
variant, with every non-coordinate setting held equal.  zco is not run because
it does not exercise partial cells and adds no gate-1 coverage beyond the two
registered variants.

Pinned source hashes are: `usrdef_nam f21a82df...`, `usrdef_hgr 9a8d7208...`,
`usrdef_zgr 91160159...`, `usrdef_sbc 9b4ba4c...`, and `usrdef_istate
eeaaa70e...` (full values are emitted by the receipt collector).

The BBL overlay is copied from the actual ORCA1-OMIP target, not inferred:
`/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_cfg:284-291`
(SHA256 `7cfe2d47d78a00553cb28fe72c7e2be8655f96f0ea22920f0b8f17f5b2a47a0b`)
sets `ln_trabbl=.true.`, `nn_bbl_ldf=0`, `nn_bbl_adv=2`,
`rn_ahtbbl=1000.`, and `rn_gambbl=20.` exactly.

### LOCK_EXCHANGE

The domain is a closed, nonrotating, flat-bottom, three-row tank 64 km long and
20 m deep.  At `rn_dx=500 m`, `rn_dz=1 m`, the source sets `jpi=130`, `jpj=3`,
`jpk=21`.  It begins at rest with uniform salinity 35, temperature 5 C to the
left of 32 km and 30 C to the right (`tests/LOCK_EXCHANGE/MY_SRC/usrdef_istate.F90:65-75`);
surface fluxes are zero.  Its expected behaviour is the classical exchange
flow, whose front propagation and spurious interior mixing diagnose advection
and closure choices (`tests/README.rst:109-125`).  The only supported coordinate
is the shipped one-dimensional z coordinate (`key_vco_1d`); that is the sole
LOCK_EXCHANGE variant.

Pinned source hashes are: `usrdef_nam 31315b6d...`, `usrdef_hgr d8a923a0...`,
`usrdef_zgr 019f01af...`, `usrdef_sbc 13ed0f0d...`, and `usrdef_istate
4313abe5...`.

## Exact receipt resolution

All builds retain the shipped `key_qco` and `key_RK3` and remove only
`key_xios`, which the DINO conda toolchain does not provide.  A missing key, a
compile failure, or a run failure is a finding and forbids fallback.  All runs
retain the shipped testcase dynamics (flux-form UBS momentum, enstrophy
vorticity, split-explicit free surface, no explicit tracer diffusion, no drag)
so the documented testcase remains identifiable.  “OMIP-style” gates the
features in scope, not unrelated dynamics.

| receipt | base namelist | required resolved overlay |
|---|---|---|
| `OVERFLOW_OMIP_L1_ZPS` | `namelist_zps_FCT2_flux_ubs_cfg` SHA256 `120c4e9d...` | `nn_COORD=1`; 6120 x 10 s; TEOS-10 only; FCT h2/v2/implicit=1; `ln_zad_Aimp=T`; advective BBL option 2, diffusive BBL off, `rn_ahtbbl=1000`, `rn_gambbl=20`; mesh and restart written |
| `OVERFLOW_OMIP_L1_SCO` | `namelist_sco_FCT2_flux_ubs_cfg` SHA256 `1e4252d7...` | identical to ZPS except `nn_COORD=2` |
| `LOCK_EXCHANGE_OMIP_L1_ZCO` | `namelist_cfg` SHA256 `c647e318...` | 61200 x 1 s; TEOS-10 only; FCT h2/v2/implicit=1; `ln_zad_Aimp=T`; BBL disabled (flat-bottom waiver); mesh and restart written |

The resolved `output.namelist.dyn`, compiled `cpp.history`, executable, mesh,
restart, trajectory, and stdout are SHA256-pinned in the final receipt.  The
active RK3 source calls stages 1/2/3 from `src/OCE/stprk3.F90:195-207`; FCT is
selected at `src/OCE/TRA/traadv.F90:358-365`; the adaptive-implicit selector is
read and rejected only for UBS tracer advection at `traadv.F90:421-488`; BBL is
applied in RK stage 3 at `src/OCE/stprk3_stg.F90:587-599`.

## Time-level registry and trajectory contract

Every trajectory record is captured on entry to `stp_RK3`, before boundary
updates and before the three stage calls.  Its state is therefore `Nbb`
(before): `ts(:,:,:,:,Nbb)`, `uu(:,:,:,Nbb)`, `vv(:,:,:,Nbb)`, and
`ssh(:,:,Nbb)`.  The record carries magic, format version, experiment, step,
`Nbb`, dimensions, 64-bit storage declaration, and those arrays.  No field may
be relabelled “now” or “after”.  The gate rejects unknown labels, absent fields,
dimension mismatches, or storage other than 64 bits.

Full entry records are preregistered at steps 1, midpoint, and last step:
`1,3060,6120` for each OVERFLOW run and `1,30600,61200` for LOCK_EXCHANGE.
These are structural trajectory receipts, not a claimed scientific averaging
window.  The instrument is write-only and must not mutate model arrays.

## Geometry and coverage gate (must pass before trajectory interpretation)

The gate inventories every variable in `mesh_mask*.nc`, every variable in the
final restart, and every assignment in the resolved namelist.  A manifest must
give each item exactly one `VERIFIED` or `WAIVED` disposition and a nonempty
reason.  Missing or extra inventory entries, duplicate dispositions, unknown
statuses, non-finite wet state, wrong dtype, wrong dimensions, wrong coordinate
branch, an unresolved required selector, or any hash mismatch hard-fails.

Geometry verification is analytic: dimensions/resolution, closed boundary
masks, uniform horizontal metrics, zero Coriolis, bathymetry formula, zps 10%
minimum and bottom-cell reconstruction, zps face-min specialization, sco
terrain-following thickness, flat LOCK bottom, positive wet thickness, and
metric/depth identities.  Metadata and unused optional restart fields may be
waived only by explicit name and reason.  A planted unaccounted variable and a
planted nonzero wet-cell geometry perturbation must each make the gate fail.

## Preregistered sanity predicates

These predicates are decided before receipt runs:

1. **CONFIRM completion** iff NEMO exits zero, reaches the exact registered
   final step, writes the final restart, and stdout contains no `ctl_stop`,
   floating exception, NaN, or infinite model value.  Otherwise **REFUTE**.
2. **CONFIRM build mode** iff resolved keys contain `key_qco`, `key_RK3`, and
   the registered vertical key, omit `key_xios`, and trajectory storage is 64
   bits.  Otherwise **REFUTE**, with no fallback.
3. **CONFIRM geometry/coverage** iff the fail-closed gate and both planted
   controls behave exactly as specified above.  Otherwise **REFUTE** and do not
   interpret trajectories.
4. **CONFIRM FCT monotonicity** iff all wet-cell temperature and salinity values
   in every registered entry record remain within their initial closed ranges
   (OVERFLOW T 10..20 C; LOCK T 5..30 C; S exactly 35).  The corrected
   numerical bar is relative and tracer-specific: closed-range excess divided
   by `max(abs(initial endpoints),1)` must not exceed
   `sqrt(N_steps) * eps(fp64)`.  Temperature remains **UNMEASURED** pending
   attribution of its 640--796-epsilon relative excess; salinity is classified
   against this floor and is never called oracle debt merely for missing it.
5. **CONFIRM documented behaviour** iff the dense/cold water centre of mass
   advances in the expected direction and the final wet velocity norm is
   nonzero, while zero-flux global tracer inventory closure stays within the
   gate-coded `64 * eps * max(initial absolute inventory,1)` envelope.  Values
   are reported, never silently reclassified.
6. **CONFIRM zps face-min rule** iff every active U/V/F bottom thickness equals
   the local neighboring minimum to fp64 comparison tolerance.  The separate
   2 m minimum/re-indexing arm at `usrdef_zgr.F90:202-214` is registered but
   must be classified from the compiled vertical-coordinate key before use.
   Otherwise **REFUTE**.

TEOS-10 density, `rab`, and BN2; adaptive vertical advection; and BBL activity
are coverage receipts: the exact active namelist/source arms and finite output
arrays must be present.  Their numerical oracle values are emitted without a
legoESM comparison in phase 1.

## Preregistered round-1 BBL attribution control

Before this control is run, the only registered arm is
`OVERFLOW_OMIP_L1_SCO` with `ln_trabbl=.false.`.  Its committed namelist is
semantically identical to the failed sco receipt except for that one logical;
the executable, `key_qco + key_RK3`, coordinate, EOS, FCT, timestep, end step,
and trajectory instrument remain fixed.  Run root:
`/data/abyssal/dbalwada/nemo-testcases-l1/overflow_sco_no_bbl_control`.

* If it reaches step 6120 and writes the final restart, the advective BBL owns
  the original step-4772 blow-up.  This is mechanistically consistent with the
  sco downslope mask being nonzero in every interior column.
* If it still stops, BBL is refuted as sole owner and TEOS-10 sigma-coordinate
  pressure-gradient truncation is the next suspect.

No second attribution arm is authorized in this round.

## Post-run Rule-0 erratum (does not alter the registered bar)

The preregistration initially associated the source's 10% minimum-thickness
arm with the zps receipt.  Compiled-arm reconciliation shows that statement was
wrong: the pinned OVERFLOW build has `key_vco_3d`, so the active zps code is
`usrdef_zgr.F90:157-188`; the 10% arm at lines 202-214 is guarded by
`lk_vco_1d3d` and is dead.  The gate therefore certifies the active
neighboring-face minimum specialization and explicitly waives the dead 10%
arm.  It never prints a 10%-minimum certification for this build.
