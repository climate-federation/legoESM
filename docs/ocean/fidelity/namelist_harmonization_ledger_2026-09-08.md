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
