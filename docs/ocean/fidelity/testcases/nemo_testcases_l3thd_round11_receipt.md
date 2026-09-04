# Lane 3b round-11 receipt — scalar libm closure and rung-3.6 preflight stop

Date: 2026-09-04  
Tracker: `climate-federation/legoESM#1699`  
Starting commit: `b834e734bb82df187c2c9843fa6ad97767c84e23`  
Recovery git: `/tmp/codex-si3thd-localgit`

## Outcome

Rung 3.5b is **CONFIRMED AT-BIT** under the registered CPU/fp64 stack.  The
shared scalar-libm policy moves all 18 previously exp-owned rows to exact
agreement: **271,560/271,560 bit-identical**, zero non-bit rows, and zero rows
over the pointwise normalized `1e-15` bar.

Rung 3.6 is **BLOCKED AT INPUT PREFLIGHT and remains UNMEASURED**.  Its reviewed
oracle requires the real ORCA1 RGB chlorophyll source and weight file.  Neither
exists under `/data/abyssal/dbalwada`; the only local chlorophyll input is the
explicitly forbidden `CHLA_BATS.nc`.  The published input archive expired on
2026-04-01 and the sandbox cannot resolve its host.  In accordance with the
reviewed preregistration, no rung-3.6 NEMO config, derived input, writer, run,
legoESM exchange implementation, numerical claim, or plant was created.

## Exact GYRE import and composition

The shared implementation was fetched from
`origin/fidelity/nemo-testcases-l2-gyre-codex2` and imported exactly from source
commit `61180a6776c`.  Import commit `e2ad2c30629` contains only the requested
files and records that source commit in its message.

| imported file | SHA-256 in this branch and `61180a6776c` |
|---|---|
| `.claude/skills/oracle-fidelity/SKILL.md` | `7eeadbdab53a486fb305553bb8c10bd1ec645591596de9eb9ac8942db78aef28` |
| `core/precision.py` | `c2eab4ae531621dce8f15ab5e8c80ff7989831feb1471ae69dbdea8a7964272a` |
| `core/transcendentals.py` | `bb324d5245160c4115094d198fd8d3ae2f58694c6cd29c2f67c94049dc572724` |
| `tests/unit/test_precision.py` | `c6b200c0558427c267aacdd1f6f8d53865ec62d66dfddc12409170312d8cc4a2` |
| `tests/unit/test_transcendentals.py` | `524bd716f96feb65a05eb2a345aa65334d49d2d8c523b55df53a3316d3604f9f` |

The exact GYRE `precision.py` predates lane 3b's public
`nemo_source_round` symbol.  Pre-implementation search found no surviving
owner after the checkout.  The already certified Round-10 helper was therefore
promoted to the single `core/source_rounding.py` owner and all three existing
callers were redirected there.  The imported policy/transcendental modules
were not modified.  This composition fact is **DEBT TO REPORT TO THE GYRE
LANE**; it is not hidden as a local change to their module.

## Executing policy and source path

Search found exactly two active `jnp.exp` calls on the certified SI3 bulk path,
both the dry/melt snow-decay evaluations in no-pond `ice_alb`.  They now call
the shared policy `exp`; Goff saturation keeps native `log10` and power, and no
executing certified-path `tanh` exists.  Native sin/cos/log/pow were not
changed.  The C1D card explicitly carries
`PrecisionPolicy.fp64(transcendentals="libm")`, which the gate installs before
JAX traces the operator.  No production default changed and no second math or
bulk implementation was added.

| boundary | NEMO source | legoESM owner | result |
|---|---|---|---|
| dry/melt snow exponential | `icealb.F90:167-169` | `ice/sea_ice.py`, shared `core/transcendentals.exp` | 18 non-bit rows to 0 |
| all source-associated bulk statements | `sbcblk.F90:1085-1346`; `icesbc.F90:310-437` | existing bulk/ice paths plus shared `nemo_source_round` | 271,542 rows remain bit-identical |

The accepted bit claim is valid only under Python 3.13.0, JAX/jaxlib 0.10.0,
NumPy 2.4.4, CPU, JAX x64, and the explicit libm policy.  The gate stamps those
values and withholds bit claims on another stack unless
`--require-bit-identity` is requested, in which case it fails closed.

## Measurement and controls

Accepted external artifact:
`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_round11_libm/bulk_gate.json`

| metric | measured value |
|---|---:|
| steps | 8,760 |
| comparisons | 271,560 |
| bit-identical | 271,560 |
| non-bit | 0 |
| normalized rows over `1e-15` | 0 |
| maximum absolute/normalized/relative/row-scale-ULP error | 0 / 0 / 0 / 0 |
| JSON SHA-256 | `9c6040ef1824c5d1c1a9a261afd273358e55dbe07455bf17b31739e87d4e5712` |
| JSON bytes | 39,286 |

The private `libm_return` plant changes only the selected policy exponential by
`1e-6`.  Its CLI run exits **1**, places 13,468 scored rows over the bar (13,902
rows are non-bit-identical), and first fires on `POST_BLK_ICE_2.albedo` at step
1.  The affected outputs are albedo, ice shortwave, and total shortwave,
demonstrating propagation into the score.
All earlier bulk plants also exit nonzero in the focused gate test.

| plant evidence | SHA-256 |
|---|---|
| `plant_libm_return.exit` (contains `1`) | `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` |
| `plant_libm_return.stderr` | `1293c678c178290fdba0744fd4277b3a42d54bfab10f0095abd3a7b416b0e6c7` |

Test receipts:

- imported policy/transcendental plus bulk gate: `85 passed in 33.12s`;
- constants ratchet selected for every touched package file: `7 passed in
  0.71s`;
- bulk/thermodynamic regression selection: `21 passed in 18.53s`.
- complete retained SI3 fidelity selection: `63 passed in 92.30s`.

A repository-wide constants-ratchet probe was also attempted fail-fast.  It
stopped after 45 passes and one skip on the untouched pre-existing
`core/fv3_native_physics_coupling.py` radius literal.  No touched file failed;
the selected seven-file run above is the applicable green claim.

## Rung-3.6 hard input blocker

The ORCA1 deck selects data-driven RGB light at
`/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_cfg:160-170`, specifically
`ln_qsr_rgb=.true.`, `nn_chldta=1`, and this source/weight pair:

| role | exact expected filename | local result | documented checksum |
|---|---|---|---|
| RGB chlorophyll source | `merged_ESACCI_BIOMER4V1R1_CHL_REG05.nc` | absent | none published/found |
| ORCA1 interpolation weights | `weights_reg05_bilinear.nc` | absent | none published/found |

The NEMO@IGE input documentation identifies the first file as the Mercator
ESACCI/BIOMER 1998--2011 climatology on a regular 0.5-degree grid, but gives no
download URL or checksum for it and no ORCA1 weight-file checksum.  A NEMO
Community post names both files and published this exact archive URL:

- input documentation:
  `https://pmathiot.github.io/NEMOCFG/docs/build/html/input_DATA.html`;
- NEMO Community input post:
  `https://nemo-ocean.discourse.group/t/agrif-setup-with-eorca-1-configuration/996`;
- archive named by that post:

`https://filesender.renater.fr/?s=download&token=d60510df-bcff-49d2-8e5f-9107314fdc7e`

The post explicitly says the link expired on 2026-04-01.  From this sandbox,
`curl -IL --max-time 20 <URL>` returned exactly:
`curl: (6) Could not resolve host: filesender.renater.fr`.

The existing official C1D forcing remains present and was reverified:

| input | digest |
|---|---|
| `C1D_v5.0.0.tar.gz` | MD5 `9456e6a0a84d40630ad1804fd4061caf` |
| `C1D_v5.0.0.tar.gz` | SHA-256 `54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382` |
| `ERA5_NorthGreenland_surface_84N_-36E_1h_y2018.nc` | SHA-256 `e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe` |

That is insufficient for the reviewed oracle because replacing RGB data with
constant chlorophyll, `CHLA_BATS.nc`, or generated/unverified weights would
change the rung.  The precise unblock is receipt of the two files above with
their provenance and hashes, or a live documented archive containing them.

## Commits

- `e2ad2c30629` — exact scalar-libm policy/test/skill import from
  `61180a6776c`;
- `5b34cb0d7e2` — round-11 preregistration before measurement;
- `86a8eb21d18` — policy routing, explicit card selection, shared source-round
  compatibility owner, bit gate, and poison control.

Source hashes at `86a8eb21d18`:

| file | SHA-256 |
|---|---|
| `core/source_rounding.py` | `b9c201d69384cdf4296e654e80b4e364d8ba290394815688aa7502de1e9c98e5` |
| `core/bulk_flux.py` | `bd559b7590b327714888af1a9d6a927d93e40b4f07463418938daa083d6611e6` |
| `thermo.py` | `218ba1536439a0176e5e706f136a9d042ad44c264e04fca6cf4a3d9efc34ae0a` |
| `ice/c1d_omip_l3.py` | `25ca35d12a6b95d0b041729876e4cbbd030ab499c8684d2c9d1ff712f6bf3210` |
| `ice/sea_ice.py` | `9249b40ee2beaaca70773e64e05ad6b40dab4fad84c8465ea37760861e665da0` |
| bulk gate | `b637494505326fcaf5c6fa8a49517373f1acba72a42460f184cc852acea5bf48` |
| bulk gate tests | `da7db28900c09f6bddf5bbf4c050ff1d4fab6985cc2c480707ea417b6e98bfe7` |

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| import the exact GYRE scalar-libm files | ASKED | complete; hashes above, modules unchanged |
| select libm explicitly on the C1D card | ASKED | complete |
| score 271,560 rows bitwise | ASKED | 271,560/271,560 bit-identical |
| poison the libm return and require propagation | ASKED | exits 1; 13,468 rows change |
| implement the independently reviewed rung 3.6 | ASKED | stopped at its mandatory real-input preflight |
| use BATS, constant chlorophyll, generated weights, or another substitute | UNASKED | forbidden; not done |
| create a second coupler/bulk/math implementation | UNASKED | not done |
| create config/writer/code before the reviewed preflight passes | UNASKED | not done |
| modify/delete shipped NEMO or retained data | UNASKED | not done |
| describe codex-internal work as independent review | UNASKED | not done |
| push | UNASKED | not done |

## Codex-internal review record

Two codex-internal adversarial reviews of `b135e32ac73` returned **SHIP**.  The
source review independently reran the full bit gate and found one low-severity
wording ambiguity: 13,468 plant rows are over-bar while 13,902 are non-bit.
That wording is corrected above.  The scope review reproduced the input
absence, local hashes, fetch failure, test selections, and correct preflight
stop.  Neither review is described as an independent external review.  Their
identity/commit/verdict records are committed beside this receipt.

## Flagged for future deletion

Nothing was deleted.  `bulk_gate.stdout`, `plant_libm_return.stdout`, and
`plant_libm_return.stderr` in the Round-11 external run directory are retained
intermediate logs and are **FLAGGED FOR FUTURE DELETION**.  The accepted JSON
and plant exit marker remain the hash-bound evidence.  All V1/V2 oracle roots
and prior-round artifacts remain retained.
