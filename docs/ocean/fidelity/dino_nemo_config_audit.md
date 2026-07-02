# DINO (Kamm et al. 2025) — authoritative NEMO config audit

Source of truth: the NEMO 4.2.1 DINO test configuration
[`vopikamm/DINO@DINO_4.2.1`](https://github.com/vopikamm/DINO) — its
`EXPREF/namelist_cfg` (active overrides) + `MY_SRC/usrdef_*.F90` (grid /
forcing / IC generators) + the Zenodo release `10.5281/zenodo.15016824`.
This is the ground truth for the legoESM `dino` experiment and the planned
NEMO-DINO oracle. Every flag below was read from the actual namelist and
confirmed against `namelist_ref` comments (the meaning of each `nn_*`/`ln_*`).

**This audit corrects two long-standing wrong notes** in our DINO encodings
(see "Corrections" at the end) and — most importantly — settles which
barotropic solver the paper actually uses.

## The barotropic solver — the headline

| | barotropic solver | in paper? | legoESM analog |
|---|---|---|---|
| **NEMO DINO R1** | **`ln_dynspg_ts=.true.`** — split-explicit time-splitting free surface; `nn_e=30` sub-steps (auto, `rn_bt_cmax=0.8`), backward (`ln_bt_fw=.false.`), boxcar time-average (`ln_bt_av=.true.`, `nn_bt_flt=2`), `rn_bt_alpha=0`, nonlinear z* (`ln_linssh=.false.`, `key_qco`) | **YES** | **`explicit_substep`** (split-explicit) |
| `pr-265-dino` (Dhruv) | `implicit_cn` | no | — |
| `omip-faithful-nemo-comparison` (Pierre) | `rigid_lid` | no | — |

**Neither of the two legoESM DINO efforts uses the paper's barotropic
solver.** The paper uses split-explicit time-splitting — legoESM's
`explicit_substep` path, not `implicit_cn` (Dhruv, stable-but-ACC-limited)
and not `rigid_lid` (Pierre, physically-correct-ACC-but-runs-away).

The ACC in NEMO DINO is limited **physically by bottom form stress over the
Drake sill** (`ln_drake_sill=.true.`, depth 2500 m, width 4°, channel
65°S–45°S) acting through the split-explicit barotropic dynamics. That is
exactly the "sill form stress / bottom-pressure-torque" Pierre's rigid-lid
diagnosis (`19c04e2ca`) says "is not engaging." So the faithful DINO ACC is a
property of `explicit_substep` + the sill, and the NEMO oracle's first job is
to confirm legoESM reproduces NEMO's ACC transport and sill form stress on
that path.

## Full wiring table (NEMO DINO R1 → legoESM)

| Aspect | NEMO DINO (namelist_cfg) | legoESM `DINOConfig` (main) | Match? |
|---|---|---|---|
| Grid | 1° (`rn_e1_deg=1`), lon 0–50°, lat −70–70°, 36 levels | Mercator/MPAS 1°, 36 lvl | ✓ setup |
| Time step | `rn_Dt=2700 s` | 2700 s | ✓ |
| Free surface | nonlinear z* (`ln_linssh=.false.`, `key_qco`) | `z_star_nonlinear` | ✓ |
| **Barotropic** | **split-explicit ts, 30 substeps, boxcar avg** | `implicit_cn` | ✗ → `explicit_substep` |
| EOS | **S-EOS** (`ln_seos=.true.`, `ln_non_lin=.true.`): a0=0.165, b0=0.76554, λ1=0.06, λ2=0, μ1=1.497e-4, μ2=0, ν=0 → nonlinear in T (cabbeling+thermobaric), **linear in S, no T·S** | Wright full nonlinear | ✗ |
| Vorticity / Coriolis | **EEN** (`ln_dynvor_een=.true.`, `nn_e3f_typ=1`) | `explicit_ab2` face-f avg | ✗ |
| KE gradient | **Hollingsworth** (`nn_dynkeg=1`) | `hollingsworth` | ✓ (Dhruv's #264 fix — paper confirms it's required) |
| Momentum advection | vector-invariant (`ln_dynadv_vec=.true.`) | vector-invariant | ✓ |
| PGF | s-coord standard Jacobian (`ln_hpg_sco=.true.`) | `adcroft` (closest) | ✓ |
| Momentum viscosity | Laplacian, **iso-level** (`ln_dynldf_lev=.true.`), Uv=0.27 m/s, `nn_ahm_ijk_t=20` | iso-level Laplacian | ✓ |
| Tracer LDF | Laplacian **isopycnal** (`ln_traldf_iso`) + **MSC** (`ln_traldf_msc`, slpmax=0.01); Ud=0.027 m/s, Ld=100 km, `nn_aht_ijk_t=20` | GM/Redi (Visbeck) | ✗ partial (need iso+MSC, Ud·Ld coeff) |
| GM / eddy-induced vel | `ln_ldfeiv=.true.`, **Ue=0.03 m/s × Le=100 km** (velocity×length), `nn_aei_ijk_t=21` | Visbeck (1997) | ✗ (formulation differs) |
| Tracer advection | **FCT** 2nd-order h&v (`ln_traadv_fct`, nn_fct_h=2, nn_fct_v=2) | (verify) | ? |
| Vertical mixing | **TKE** (`ln_zdftke`) + NIW penetration (`nn_etau=1`) + **EVD** (`ln_zdfevd`, `nn_evdm=1` T&U, rn_evd=100); background avm0=1.2e-4, avt0=1.2e-5 m²/s | KPP default (TKE option exists, unstable >~day40) | ✗ (paper = TKE+EVD) |
| Bathymetry | H=4000 m deep, 2000 m coast; **Drake sill ON** 2500 m × 4°; no mid-ridge | matches | ✓ |
| Channel | re-entrant 65°S–45°S (`ln_Iperio`), wall slope 1.5 | matches | ✓ |
| Wind stress | annual cycle, τ0=0.2 N/m² (`nn_forcingtype=4`) | matches | ✓ |
| T* restoring | −40 W/m²/K; T*_s=−0.5, T*_n=5, T*_eq=27 °C | A_theta=40, matches | ✓ |
| S* restoring | −3.858e-3; S*_s=35, S*_n=35.1, S*_eq=37.25 | matches | ✓ |
| Solar | `ln_qsr`, 2-band (`ln_qsr_2bd`) | Q_sr split + Jerlov | ✓ |
| Spin-up | 3000 yr R1 + 400 yr prod, last-50-yr diagnostics | local ≤20 yr (code-correctness only) | n/a |

## Corrections to existing legoESM encodings

1. **EOS is S-EOS, not TEOS-10/EOS-80.** `veros_configs/dino.py` says the paper
   "uses TEOS-10 / EOS-80" — **wrong**. The namelist sets `ln_seos=.true.`,
   `ln_teos10=.false.`, `ln_eos80=.false.`: a simplified Roquet S-EOS with the
   exact coefficients above (nonlinear in T only). A faithful oracle/recipe
   needs this S-EOS as a selectable linear-plus-cabbeling block, not Wright and
   not full TEOS-10.
2. **Hollingsworth is paper-required, confirmed.** `nn_dynkeg=1` ⇒ Dhruv's
   Finding 2 ("lat-lon C-grid needs Hollingsworth") matches the paper exactly;
   this is not a legoESM-only stability hack.

## NEMO-specific build facts (for the local oracle)

- NEMO 4.2.1 + XIOS, compile keys `key_xios key_qco` (`cpp_DINO.fcm`).
- DINO config = NEMO `tests/` overlay: `MY_SRC/usrdef_{nam,hgr,zgr,sbc,istate}.F90`
  generate geometry / vertical grid / surface forcing / initial state in
  Fortran (the authoritative IC/forcing source — read these for the bridge).
- 1° R1 ≈ 2 hCPU/yr on 36+1 cores (Jean Zay). Local short tendency + days–months
  spin-up reference is tractable; multi-century equilibrium is not local work.
- Reference restarts / `emp` forcing are on the Zenodo release (too big for git).
