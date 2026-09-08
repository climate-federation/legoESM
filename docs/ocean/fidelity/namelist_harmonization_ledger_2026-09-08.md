# ORCA1 namelist vs our production tripole card — the rows that still differ

Written 2026-09-08 during the cluster maintenance window, when nothing can be
measured. Every row below is READ OFF THE CODE AND THE ORACLE'S NAMELIST, not
measured; each is a candidate one-variable arm once the machine returns.
Provenance: `cfgs/ORCA1/EXP00/namelist_cfg` (and `namelist_ref` where the cfg
does not override), against `scripts/cluster/omip_nemo/_trp_base2_d30.sbatch`
plus what `orca1_zdftke_config` / `NEMOMatchTripoleRecipeConfig` construct.

## Rows that DIFFER

| what | the oracle | ours | note |
|---|---|---|---|
| eddy-induced velocity coefficient | `ln_ldfeiv=T`, `nn_aei_ijk_t=21` (Treguier 1997), `rn_Ue=0.018`, `rn_Le=100e3`, i.e. aei0 = Ue·Le/2 = **900 m2/s**, spatially varying | constant `kappa_GM=600` | `--gm-treguier` EXISTS and the branch is named for it, but the production card does not pass it |
| lateral tracer diffusion | `nn_aht_ijk_t=21`, same Treguier form, `rn_Ud=0.018`, `rn_Ld=100e3` | constant `kappa_Redi=600` | same shape of gap as the row above |
| tracer advection | `ln_traadv_fct=T`, `nn_fct_h=2`, `nn_fct_v=2` (FCT2) | card passes `superbee` | our driver's own help calls `ppm_fct` "closest to NEMO's FCT2"; an earlier note records ppm_fct destabilising the Gulf Stream, so this row has a REASON, not just a gap |
| kinetic-energy gradient | `nn_dynkeg=1` (Hollingsworth) | `centered` | our recipe documents why: the Hollingsworth form is not fold-aware on the tripole seam. A real constraint, not an oversight |

### ★ The eddy row has the WRONG SIGN at the equator, and that matters here

Read off the oracle's own formula (`compute_treguier_kappa_gm`, transcribed
from `ldf_eiv`): the Treguier coefficient carries an explicit TROPICAL TAPER,
`min(1, |f/f_20|)` with `f_20 = 2*Omega*sin(20 deg)`. At the equator the
Coriolis parameter goes to zero, so the ORACLE'S EDDY COEFFICIENT GOES TO ZERO
THERE. Our production card uses a CONSTANT 600 m2/s at every latitude, and the
same is true of our Redi coefficient while the oracle tapers that one too.

So in the equatorial waveguide we are applying roughly 600 m2/s of eddy
transport and isopycnal mixing where the oracle applies essentially none. The
sign is the one that matters for every bias this campaign is chasing: spurious
eddy flattening of the equatorial thermocline weakens the zonal density
gradient, which weakens the undercurrent, which removes the shear that feeds
turbulence, which is the 20-60 m mixing collapse and the too-high Prandtl
number. Each link is individually plausible and the first two are measured.

This REVERSES the reason for running the eddy arm. It is not that we are
missing a flow-dependent coefficient somewhere useful; it is that we are
applying a large constant one exactly where the oracle applies none. Status:
the taper and our constant are CONFIRMED by reading both codes; the chain from
there to the biases is PLAUSIBLE and is what the arm tests.

(An independent reviewer ranked this row as irrelevant at the equator on the
grounds that eddy closures taper off in the waveguide. That is right about the
ORACLE and wrong about US, which is the whole point.)

### What `--gm-treguier` does and does not close (checked 2026-09-08)

The flag makes ONLY the eddy-induced coefficient flow-dependent. Our
implementation's own constant is `gm_aei0 = 900`, derived in the code as
half of the oracle's `rn_Ue * rn_Le` with the namelist provenance written next
to it, so row 1 above is exactly what the flag closes -- including the value.

There is NO equivalent path for the Redi/lateral-tracer coefficient: the module
exposes a Treguier kappa for GM only, and `kappa_Redi` stays the constant 600.
So row 2 survives the flag and would need its own work. Do not expect one arm
to close both.

## Rows that AGREE (checked, so they are not re-opened)

| what | value on both sides |
|---|---|
| momentum advection | vector-invariant (`ln_dynadv_vec=T` / `momentum_advection="vector_invariant"`) |
| hydrostatic pressure gradient | s-coordinate form (`ln_hpg_sco=T` / `--pgf-scheme smc03`) |
| penetrative solar | RGB with chlorophyll data (`ln_qsr_rgb=T`, `nn_chldta=1` / `--sw-rgb-chl`) |
| internal-wave mixing | on, and it forces the molecular backgrounds on BOTH sides (`ln_zdfiwm=T` / `--iwm`) |
| TKE closure | on, with Langmuir, sub-mixed-layer penetration at `rn_efr=0.08` and the latitude-dependent depth (`nn_etau=1`, `nn_htau=1`) |
| Prandtl number | Richardson-dependent, `nn_pdl=1`, same critical-Richardson slope |
| enhanced vertical diffusion | present on both, and separately shown not to matter here |
| tracer damping | off (`ln_tradmp=F`) |

## Reading

The first two rows are the same defect twice: the oracle makes both its eddy
transport and its lateral tracer mixing follow a Treguier coefficient that
varies with the flow, and we use a constant that is a third smaller at the
reference value. That bears directly on the equatorial thermocline, and unlike
the shear row it is a coefficient rather than a discretisation, so it is cheap
to test.

The last two rows have documented reasons and should NOT be flipped casually:
one destabilised the Gulf Stream when tried, the other is unsafe at the tripole
fold. They are recorded so nobody re-derives them.
