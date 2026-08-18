"""SHRINK-ONLY dispositions for NEMO's shared mutable state (see
``tests/ocean/test_nemo_shared_state_coverage.py`` for the enumeration and the
gate).

FOUR DISPOSITIONS, NO FIFTH
---------------------------
``IN_STATE``   represented in ``LatLonCGridOceanState`` -- name the field.  A
               reason containing ``MISSING`` marks a NEMO carry legoESM does
               NOT reproduce: it is a state gap, deliberately kept visible
               here rather than waived away.
``THREADED``   passed explicitly to the consumer, or recomputed inside the
               step from arguments -- name where.
``INERT``      provably dead for THIS config.  Requires a machine EVIDENCE
               tuple, re-checked against the oracle by
               ``test_inert_evidence_holds``:
                 ``("switch", "ln_x", False)``   -- resolved namelist logical
                 ``("value",  "nn_x", 0)``       -- resolved namelist number
                 ``("cpp",    "key_x", False)``  -- cpp key in cpp_DINO.fcm
                 ``("text",   "<printed line>")``-- exact line in ocean.output
``WAIVED``     written reason, no machine evidence required.

Entries may only SHRINK: a baseline key that no longer matches an enumerated
symbol/module is RED (``test_baseline_only_shrinks``), so the table cannot rot
into slack that hides a new gap.

RETRACTION (recorded per oracle-fidelity Rule 11).  The seed list handed to
this task named ``sbc_hc_b`` and ``sbc_sc_b`` as missing carries.  They are
NOT Fortran symbols: ``trasbc.F90:118-119`` reads the RESTART VARIABLES
``'sbc_hc_b'``/``'sbc_sc_b'`` into ``sbc_tsc_b(:,:,jp_tem)`` and
``sbc_tsc_b(:,:,jp_sal)``.  There is one array, ``sbc_tsc_b``, and it is the
entry below.
"""
from __future__ import annotations

IN_STATE = "IN_STATE"
THREADED = "THREADED"
INERT = "INERT"
WAIVED = "WAIVED"

DISPOSITIONS = frozenset({IN_STATE, THREADED, INERT, WAIVED})

# The standard reason for the bulk waiver.  A symbol reaches the enumeration
# through a bare ``USE mod`` (which imports every PUBLIC symbol) but its name
# appears in NO live-chain source file, so no live routine can read or write
# it.  Absence of a textual reference is the only sound direction for a
# waiver; the gate proves the reference join is real
# (``test_unreferenced_waiver_is_evidence_backed``).
UNREFERENCED_REASON = (
    "reached only through a whole-module `USE` with no ONLY clause; the "
    "identifier appears in no live-chain source file, so no live routine "
    "references it")


# ---------------------------------------------------------------------------
# SCOPE CARVE-OUT.  Stated, not silent (oracle-fidelity Rule 1: an honest
# "N excluded, here is what covers them" beats a clean table with guesses).
# Each name must be a module the enumeration WOULD otherwise produce
# (``test_excluded_modules_are_real_and_declared``).
# ---------------------------------------------------------------------------
EXCLUDED_MODULES: dict[str, str] = {
    "dom_oce": (
        "STATIC GRID/DECOMPOSITION state (metrics, masks, depth ladders, "
        "wet-level indices, MPI tile bounds, calendar lengths, and the "
        "key_qco thickness ratios r3t/r3u/r3v/r3f derived from ssh).  Covered "
        "by a DIFFERENT, array-by-array coverage gate that compares against "
        "the oracle's own mesh_mask.nc: "
        "scripts/validate/ocean_fidelity/dino_1226/nemo_geometry_gate.py "
        "(45 arrays, 0 unaccounted).  Duplicating it here would re-guess what "
        "that gate measures.  124 symbols."),
}


# ---------------------------------------------------------------------------
# MODULE-WIDE RULES.  Applied to a REFERENCED symbol with no per-symbol entry.
# One line per module keeps a uniform verdict from being restated 22 times;
# anything non-uniform gets a per-symbol entry below instead.
# ---------------------------------------------------------------------------
MODULE_DISPOSITION: dict[str, tuple] = {
    # --- whole subsystems switched off for DINO -------------------------
    "abl": (INERT, "atmospheric boundary-layer model off", ("switch", "ln_abl", False)),
    "agrif_oce": (INERT, "no nesting; key_agrif is not in cpp_DINO.fcm",
                  ("cpp", "key_agrif", False)),
    "asminc": (INERT, "assimilation increments compiled out (trasbc.F90:28 "
                      "`#if defined key_asminc`)", ("cpp", "key_asminc", False)),
    "ice": (INERT, "no sea ice in the surface BC", ("value", "nn_ice", 0)),
    "sbc_ice": (INERT, "no sea ice in the surface BC", ("value", "nn_ice", 0)),
    "icb_oce": (INERT, "icebergs off (ocean.output icb_nam block)",
                ("text", "No icebergs used")),
    "icbdia": (INERT, "iceberg budget diagnostics; icebergs off",
               ("text", "No icebergs used")),
    "isf_oce": (INERT, "no ice-shelf cavities", ("switch", "ln_isfcav", False)),
    "sbcapr": (INERT, "no atmospheric-pressure forcing", ("switch", "ln_apr_dyn", False)),
    "sbcrnf": (INERT, "no river runoff", ("switch", "ln_rnf", False)),
    "sbcssr": (INERT, "no SST/SSS restoring via sbcssr (DINO restores inside "
                      "usrdef_sbc instead)", ("switch", "ln_ssr", False)),
    "sbcwave": (INERT, "no wave forcing/coupling", ("switch", "ln_wave", False)),
    "sbc_phy": (INERT, "bulk-formula constants; no bulk formulation",
                ("switch", "ln_blk", False)),
    "tide_mod": (INERT, "no tidal potential", ("switch", "ln_tide", False)),
    "trc": (INERT, "passive tracers not compiled", ("cpp", "key_top", False)),
    "trc_oce": (INERT, "passive-tracer shared variables; TOP not compiled",
                ("cpp", "key_top", False)),
    "zdfmfc": (INERT, "mass-flux convection off", ("switch", "ln_zdfmfc", False)),
    "zdfosm": (INERT, "OSMOSIS BL closure off", ("switch", "ln_zdfosm", False)),

    # --- I/O, diagnostics, RNG and harness plumbing ---------------------
    "diaar5": (WAIVED, "AR5/CMIP diagnostic accumulators; output only, no "
                       "tendency reads them"),
    "diadct": (WAIVED, "transport-section diagnostics; output only"),
    "iom": (WAIVED, "XIOS/netCDF I/O plumbing"),
    "iom_def": (WAIVED, "XIOS/netCDF file-handle table"),
    "lib_mpp": (WAIVED, "MPI delayed-reduction bookkeeping; legoESM's DINO "
                        "harness runs single-domain (nn_hls=0)"),
    "stpctl": (WAIVED, "run-control (blow-up detection) bookkeeping"),
    "storng": (WAIVED, "stochastic-parameterisation RNG state; referenced only "
                       "inside storng.F90, which no live-chain routine calls. "
                       "WAIVED not INERT: ln_sto_eos is not printed in "
                       "ocean.output, so its value is not mechanically proven"),
    "stopts": (WAIVED, "stochastic T/S perturbation array, read under "
                       "`IF( ln_sto_eos )` (eosbn2.F90:438,746). WAIVED not "
                       "INERT: ln_sto_eos is not printed in ocean.output"),
    "traadv_fct": (WAIVED, "allocate-once FCT scratch (tridiagonal workspace); "
                           "SAVEd for allocation cost only, carries no meaning "
                           "between calls"),
    "dynzdf": (WAIVED, "#1226 debug dump buffers added to the ORACLE by this "
                       "campaign; instrumentation, not NEMO physics"),
}


# ---------------------------------------------------------------------------
# PER-SYMBOL DISPOSITIONS.
# ---------------------------------------------------------------------------
SYMBOL_DISPOSITION: dict[tuple[str, str], tuple] = {

    # =====================================================================
    # oce -- the prognostic core and the barotropic sub-step workspace
    # =====================================================================
    ("oce", "uu"): (IN_STATE, "state.u (Nnn) + state.u_before (Nbb)"),
    ("oce", "vv"): (IN_STATE, "state.v (Nnn) + state.v_before (Nbb)"),
    ("oce", "ts"): (IN_STATE, "state.T/state.S (Nnn) + state.T_before/"
                              "state.S_before (Nbb)"),
    ("oce", "ssh"): (IN_STATE, "state.eta (Nnn) + state.eta_before (Nbb)"),
    ("oce", "ww"): (IN_STATE, "state.w"),
    ("oce", "uu_b"): (THREADED, "barotropic depth mean; recomputed from u/h "
                                "inside the barotropic solve each step"),
    ("oce", "vv_b"): (THREADED, "as uu_b, for v"),
    ("oce", "hdiv"): (THREADED, "horizontal divergence recomputed from u/v "
                                "each step (div_hor) and consumed in-step by "
                                "ssh_nxt/wzv"),
    ("oce", "rhd"): (THREADED, "in-situ density anomaly recomputed by eos each "
                               "step; legoESM's compute_ocean_rho output flows "
                               "as a local through the HPG/PGF chain"),
    ("oce", "rhop"): (THREADED, "potential density recomputed by eos each step"),
    ("oce", "rab_n"): (THREADED, "alpha/beta recomputed by eos_rab at Nnn each "
                                 "step (stpmlf.F90:185)"),
    ("oce", "rab_b"): (THREADED, "alpha/beta recomputed by eos_rab at Nbb each "
                                 "step (stpmlf.F90:184); the BEFORE time level "
                                 "is pinned in ocean/fidelity/time_levels.py"),
    ("oce", "rn2"): (THREADED, "N^2 recomputed by bn2 each step"),
    ("oce", "rn2b"): (THREADED, "N^2 at the BEFORE tracer level, recomputed "
                                "in-step and passed explicitly as "
                                "`n2_tracers_before` "
                                "(ocean/dynamics/ocean_model_latlon_cgrid.py"
                                ":4571,4580) -- the EVD-trigger defect this "
                                "gate's class produced"),
    ("oce", "ssh_frc"): (THREADED, "barotropic ssh forcing assembled in-step "
                                   "from the surface fluxes"),
    # NEMO's persistent barotropic sub-step histories: written to the restart
    # and read back at the START of the next window.
    ("oce", "ub_e"): (IN_STATE, "state.bt_hist (deviation form) when "
                                "barotropic_time_filter='nemo_ab3am4'"),
    ("oce", "ubb_e"): (IN_STATE, "state.bt_hist (deviation form)"),
    ("oce", "vb_e"): (IN_STATE, "state.bt_hist (deviation form)"),
    ("oce", "vbb_e"): (IN_STATE, "state.bt_hist (deviation form)"),
    ("oce", "sshb_e"): (IN_STATE, "state.bt_hist (deviation form)"),
    ("oce", "sshbb_e"): (IN_STATE, "state.bt_hist (deviation form)"),
    # ...and the within-window scratch, which never crosses a window boundary.
    ("oce", "un_e"): (THREADED, "within-window barotropic substep value; local "
                                "to the substep loop"),
    ("oce", "ua_e"): (THREADED, "within-window barotropic substep value"),
    ("oce", "vn_e"): (THREADED, "within-window barotropic substep value"),
    ("oce", "va_e"): (THREADED, "within-window barotropic substep value"),
    ("oce", "sshn_e"): (THREADED, "within-window barotropic substep value"),
    ("oce", "ssha_e"): (THREADED, "within-window barotropic substep value"),
    ("oce", "hu_e"): (THREADED, "substep u-column depth, rebuilt from eta"),
    ("oce", "hv_e"): (THREADED, "substep v-column depth, rebuilt from eta"),
    ("oce", "hur_e"): (THREADED, "reciprocal of hu_e"),
    ("oce", "hvr_e"): (THREADED, "reciprocal of hv_e"),
    ("oce", "un_adv"): (THREADED, "window-averaged barotropic transport handed "
                                  "to tra_adv within the SAME step"),
    ("oce", "vn_adv"): (THREADED, "as un_adv, for v"),
    ("oce", "ub2_b"): (INERT, "half-step flux carry used only on the FORWARD "
                              "barotropic branch (dynspg_ts.F90:914 "
                              "`IF( ln_bt_fw )`); DINO integrates centred",
                       ("switch", "ln_bt_fw", False)),
    ("oce", "vb2_b"): (INERT, "as ub2_b (dynspg_ts.F90:914)",
                       ("switch", "ln_bt_fw", False)),
    ("oce", "un_bf"): (INERT, "Asselin-filtered half-step flux, forward branch "
                              "only (dynspg_ts.F90:920-932)",
                       ("switch", "ln_bt_fw", False)),
    ("oce", "vn_bf"): (INERT, "as un_bf (dynspg_ts.F90:920-932)",
                       ("switch", "ln_bt_fw", False)),
    ("oce", "ub2_i_b"): (INERT, "AGRIF time-integrated flux; dynspg_ts.F90:1022 "
                                "sits under `#if defined key_agrif`",
                         ("cpp", "key_agrif", False)),
    ("oce", "vb2_i_b"): (INERT, "as ub2_i_b (dynspg_ts.F90:1022)",
                         ("cpp", "key_agrif", False)),
    ("oce", "wi"): (INERT, "adaptive-implicit vertical advection off",
                    ("switch", "ln_zad_Aimp", False)),
    ("oce", "Cu_adv"): (INERT, "vertical Courant number for adaptive-implicit "
                               "advection", ("switch", "ln_zad_Aimp", False)),
    ("oce", "riceload"): (INERT, "sea-ice load on the free surface",
                          ("value", "nn_ice", 0)),
    ("oce", "fraqsr_1lev"): (WAIVED, "fraction of qsr absorbed in level 1: "
                                     "written by tra_qsr (traqsr.F90:197-215) "
                                     "and restart-saved (:243); its only "
                                     "consumers in NEMO are the passive-tracer "
                                     "(TOP) surface flux and the restart, and "
                                     "key_top is not compiled"),

    # =====================================================================
    # sbc_oce -- the surface-flux deposit box.  This is the module the whole
    # architectural argument is about (utau at sbc_oce.F90:107).
    # =====================================================================
    ("sbc_oce", "utau"): (THREADED, "OceanSurfaceForcing.tau_x, an explicit "
                                    "argument of step()"),
    ("sbc_oce", "vtau"): (THREADED, "OceanSurfaceForcing.tau_y"),
    ("sbc_oce", "utauU"): (THREADED, "U-point interpolation of utau, done "
                                     "in-step from the same forcing argument"),
    ("sbc_oce", "vtauV"): (THREADED, "V-point interpolation of vtau"),
    ("sbc_oce", "utau_b"): (IN_STATE, "state.tau_x_prev -- NEMO's MLF centred "
                                      "surface stress 0.5*(utau_b+utauU) "
                                      "(dynzdf.F90:360, dynspg_ts.F90:392-421); "
                                      "gated on barotropic_forcing_centred"),
    ("sbc_oce", "vtau_b"): (IN_STATE, "state.tau_y_prev (see utau_b)"),
    ("sbc_oce", "taum"): (THREADED, "wind-stress modulus; DINO's usrdef_sbc "
                                    "sets it every call and zdf_tke reads it "
                                    "for the surface TKE BC in the same step"),
    ("sbc_oce", "qns"): (THREADED, "non-solar heat flux; SurfaceTracerForcing"
                                   ".dT_dt surface entry"),
    ("sbc_oce", "qsr"): (THREADED, "solar heat flux; enters the penetrating-"
                                   "shortwave column in-step"),
    ("sbc_oce", "emp"): (THREADED, "freshwater flux; FreshwaterForcing arg"),
    ("sbc_oce", "sfx"): (THREADED, "salt flux (DINO's S-restoring, "
                                   "usrdef_sbc.F90:373); virtual-salt-flux arg"),
    ("sbc_oce", "sbc_tsc"): (THREADED, "surface tracer-content trend assembled "
                                       "from qns/qsr/emp/sfx (trasbc.F90) and "
                                       "consumed in the same step"),
    ("sbc_oce", "sbc_tsc_b"): (
        IN_STATE,
        "MISSING: NEMO averages 0.5*(sbc_tsc_b + sbc_tsc) at trasbc.F90:152, "
        "i.e. the tracer surface deposit is CENTRED in time. legoESM carries "
        "no previous-step tracer deposit -- state.py documents that only the "
        "eta/barotropic channel is centred and 'the separate virtual-salt-flux "
        "tracer deposit (NEMO tra_sbc) is untouched and stays at NOW'. Cited, "
        "not re-derived: the sbc carry is bounded offline at 2.9e-5 K/step "
        "against 0.152, so this is a REAL GAP that is not the current driver."),
    ("sbc_oce", "qsr_hc"): (THREADED, "penetrating-shortwave column heat "
                                      "content, recomputed by tra_qsr in-step"),
    ("sbc_oce", "qsr_hc_b"): (
        IN_STATE,
        "MISSING: traqsr.F90:207 applies 0.5*(qsr_hc_b + qsr_hc) and "
        "traatf_qco.F90:294/306 uses (qsr_hc - qsr_hc_b) in the Asselin "
        "correction, so the penetrating-solar column is CENTRED in time and "
        "feeds the filter. legoESM has no qsr_hc field at all (no match for "
        "'qsr_hc' anywhere under packages/ocean)."),
    ("sbc_oce", "qns_b"): (WAIVED, "written by sbcmod.F90:384 and restart-saved "
                                   "at :595, but NO routine in the OCE MLF "
                                   "build reads it -- trasbc builds sbc_tsc "
                                   "from the NOW fluxes and centres via "
                                   "sbc_tsc_b instead. Not a carry legoESM can "
                                   "be missing."),
    ("sbc_oce", "sfx_b"): (WAIVED, "as qns_b: written (sbcmod.F90:386) and "
                                   "restart-saved (:599), never read"),
    ("sbc_oce", "emp_b"): (
        INERT,
        "read by ssh_nxt (sshwzv.F90:125), the ssh Asselin filter (:427) and "
        "ssh_frc (dynspg_ts.F90:417), but DINO's emp is IDENTICALLY ZERO: "
        "usrdef_sbc.F90:395 sets emp = rn_emp_prop * (...) and "
        "rn_emp_prop = 0.0",
        ("value", "rn_emp_prop", 0.0)),
    ("sbc_oce", "rnf"): (INERT, "river runoff", ("switch", "ln_rnf", False)),
    ("sbc_oce", "rnf_b"): (INERT, "before-runoff", ("switch", "ln_rnf", False)),
    ("sbc_oce", "fr_i"): (INERT, "ice fraction", ("value", "nn_ice", 0)),
    ("sbc_oce", "fwfice"): (INERT, "ice-ocean freshwater budget",
                            ("value", "nn_ice", 0)),
    ("sbc_oce", "utau_icb"): (INERT, "iceberg stress",
                              ("text", "No icebergs used")),
    ("sbc_oce", "vtau_icb"): (INERT, "iceberg stress",
                              ("text", "No icebergs used")),
    ("sbc_oce", "e3t_abl"): (INERT, "ABL vertical scale factor",
                             ("switch", "ln_abl", False)),
    ("sbc_oce", "e3w_abl"): (INERT, "ABL vertical scale factor",
                             ("switch", "ln_abl", False)),
    ("sbc_oce", "ght_abl"): (INERT, "ABL geopotential height",
                             ("switch", "ln_abl", False)),
    ("sbc_oce", "ghw_abl"): (INERT, "ABL geopotential height",
                             ("switch", "ln_abl", False)),
    ("sbc_oce", "cloud_fra"): (WAIVED, "cloud cover: set only by the bulk/ABL "
                                       "formulations and re-emitted by "
                                       "sbcmod's iom_put; diagnostic here"),
    ("sbc_oce", "wndm"): (WAIVED, "10 m wind-speed modulus; referenced only by "
                                  "sbcmod's iom_put and diawri -- output only "
                                  "(DINO's usrdef_sbc zeroes it in the "
                                  "no-forcing branches)"),
    # The nn_fsbc running means exist to feed the sea-ice model, the coupler
    # and sbcssr.  All three are off, so nothing consumes them.
    ("sbc_oce", "sst_m"): (WAIVED, "nn_fsbc running mean for the sea-ice model "
                                   "(nn_ice=0), the coupler (ln_cpl=F) and "
                                   "sbcssr (ln_ssr=F) -- no live consumer"),
    ("sbc_oce", "sss_m"): (WAIVED, "as sst_m"),
    ("sbc_oce", "ssh_m"): (WAIVED, "as sst_m"),
    ("sbc_oce", "e3t_m"): (WAIVED, "as sst_m"),
    ("sbc_oce", "frq_m"): (WAIVED, "as sst_m (qsr fraction for the ice model)"),

    # =====================================================================
    # zdf_oce -- the vertical-mixing coefficient box
    # =====================================================================
    ("zdf_oce", "avm"): (THREADED, "total vertical viscosity, assembled by "
                                   "zdf_phy each step and passed to the "
                                   "implicit momentum solve"),
    ("zdf_oce", "avt"): (THREADED, "total vertical diffusivity for T"),
    ("zdf_oce", "avs"): (THREADED, "total vertical diffusivity for S"),
    ("zdf_oce", "avm_k"): (
        IN_STATE,
        "MISSING: the turbulent-closure-only Kz. NEMO computes it in zdf_tke "
        "and CARRIES it across the timestep (it is a restart field); the next "
        "step's zdf_phy starts from it. legoESM carries only the TKE itself "
        "(state.tke) and rebuilds Kz from scratch. Cited, not re-derived: "
        "injecting NEMO's own avt_k/avm_k leaves 98.9% of the per-step "
        "divergence, so this is a REAL GAP that is not the current driver."),
    ("zdf_oce", "avt_k"): (IN_STATE, "MISSING: see avm_k -- the tracer-side "
                                     "turbulent-closure Kz carry"),
    ("zdf_oce", "en"): (IN_STATE, "state.tke -- the prognostic TKE carried "
                                  "across steps (vertical_mixing.tke."
                                  "prognostic=True on the DINO cards)"),
    ("zdf_oce", "avmb"): (THREADED, "background viscosity PROFILE, derived "
                                    "from config (rn_avm0) and constant in "
                                    "time"),
    ("zdf_oce", "avtb"): (THREADED, "background diffusivity profile (rn_avt0)"),
    ("zdf_oce", "avtb_2d"): (THREADED, "horizontal shape of the background Kz "
                                       "profile (nn_havtb), constant in time"),

    # =====================================================================
    # Lateral-physics coefficient and slope boxes: all recomputed per step
    # from the state + geometry, and consumed inside the same step.
    # =====================================================================
    ("ldfslp", "uslp"): (THREADED, "isoneutral i-slope at U, recomputed by "
                                   "ldf_slp each step and passed to GM/Redi"),
    ("ldfslp", "vslp"): (THREADED, "isoneutral j-slope at V"),
    ("ldfslp", "wslpi"): (THREADED, "isoneutral i-slope at W"),
    ("ldfslp", "wslpj"): (THREADED, "isoneutral j-slope at W"),
    ("ldfslp", "wslp2"): (THREADED, "slope^2 at W, derived from wslpi/wslpj"),
    ("ldfslp", "ah_wslp2"): (THREADED, "aht*slope^2 at W, handed to tra_zdf's "
                                       "implicit solve in the same step"),
    ("ldfslp", "akz"): (THREADED, "stabilising vertical diffusivity from the "
                                  "Method of Stabilising Correction"),
    ("ldftra", "ahtu"): (THREADED, "lateral tracer diffusivity at U, derived "
                                   "from config (rn_Ud/rn_Ld/nn_aht_ijk_t)"),
    ("ldftra", "ahtv"): (THREADED, "lateral tracer diffusivity at V"),
    ("ldftra", "aeiu"): (THREADED, "eddy-induced-velocity coefficient at U"),
    ("ldftra", "aeiv"): (THREADED, "eddy-induced-velocity coefficient at V"),
    ("ldfdyn", "ahmt"): (THREADED, "lateral viscosity at T, config-derived"),
    ("ldfdyn", "ahmf"): (THREADED, "lateral viscosity at F, config-derived"),
    ("dynldf_iso", "akzu"): (INERT, "vertical component of the ROTATED lateral "
                                    "viscosity; DINO's lateral momentum "
                                    "diffusion is iso-LEVEL",
                             ("switch", "ln_dynldf_iso", False)),
    ("dynldf_iso", "akzv"): (INERT, "as akzu", ("switch", "ln_dynldf_iso", False)),

    # =====================================================================
    # Drag / mixed layer / shear -- all recomputed per step
    # =====================================================================
    ("zdfdrg", "rCd0_bot"): (THREADED, "precomputed bottom-drag coefficient "
                                       "(config: rn_Cd0/rn_z0), constant"),
    ("zdfdrg", "rCd0_top"): (THREADED, "precomputed top-drag coefficient"),
    ("zdfdrg", "rCdU_bot"): (THREADED, "-Cd*|U| at the bottom, recomputed each "
                                       "step from the state and consumed by "
                                       "the implicit momentum solve"),
    ("zdfdrg", "rCdU_top"): (THREADED, "-Cd*|U| at the top (ice drag); "
                                       "recomputed each step"),
    ("zdfmxl", "nmln"): (THREADED, "mixed-layer level index, recomputed by "
                                   "zdf_mxl each step"),
    ("zdfmxl", "hmld"): (THREADED, "turbocline depth, recomputed by zdf_mxl"),
    ("zdfmxl", "hmlp"): (THREADED, "density mixed-layer depth, recomputed by "
                                   "zdf_mxl and read by ldf_slp/ldf_eiv in the "
                                   "same step"),
    ("zdfphy", "sh2"): (THREADED, "shear production, computed by zdf_sh2 and "
                                  "passed as an explicit argument to zdf_tke "
                                  "(zdfphy.F90 `CALL zdf_tke(..., sh2, ...)`)"),

    # =====================================================================
    # zdftke -- the TKE closure's own module state
    # =====================================================================
    ("zdftke", "dissl"): (
        IN_STATE,
        "MISSING: the dissipative mixing length. zdf_tke calls tke_tke THEN "
        "tke_avn (zdftke.F90:184), and tke_tke reads dissl at :414 and :419 -- "
        "so the value it reads was left by the PREVIOUS timestep's tke_avn "
        "(:735). It is a module-PRIVATE SAVE array, i.e. even more hidden than "
        "utau. legoESM recomputes it within the step (physics/vertical_mixing/"
        "tke.py:1207), so the one-step lag is not reproduced."),
    ("zdftke", "htau"): (THREADED, "latitude-dependent TKE penetration depth "
                                   "profile (nn_htau=1); config-derived from "
                                   "gphit, constant in time"),
    ("zdftke", "zmxld_dump"): (WAIVED, "#1226 debug dump buffer added to the "
                                       "ORACLE by this campaign"),
    ("zdftke", "zmxlm_dump"): (WAIVED, "#1226 debug dump buffer"),
    ("zdftke", "l_1226_tke_mxl_dump_done"): (WAIVED, "#1226 dump-once flag"),

    # =====================================================================
    # dynvor / dynspg_ts -- precomputed metric combinations and the
    # barotropic solver's own scratch
    # =====================================================================
    ("dynvor", "e3f_0vor"): (THREADED, "F-point reference thickness for EEN; "
                                       "built from the geometry, which is "
                                       "passed as an explicit `geom`/`z_coord` "
                                       "argument (een_e3f_scheme selects the "
                                       "rule)"),
    ("dynvor", "di_e2u_2"): (THREADED, "precomputed metric combination for the "
                                       "ENS/ENE/EEN vorticity operators"),
    ("dynvor", "dj_e1v_2"): (THREADED, "precomputed metric combination"),
    ("dynvor", "di_e2v_2e1e2f"): (THREADED, "precomputed metric combination"),
    ("dynvor", "dj_e1u_2e1e2f"): (THREADED, "precomputed metric combination"),
    ("dynspg_ts", "wgtbtp1"): (THREADED, "barotropic substep filter weights, "
                                         "derived from nn_bt_flt/nn_e"),
    ("dynspg_ts", "wgtbtp2"): (THREADED, "secondary substep filter weights"),
    ("dynspg_ts", "ffu_nw"): (THREADED, "precomputed EEN Coriolis triad metric"),
    ("dynspg_ts", "ffu_ne"): (THREADED, "precomputed EEN Coriolis triad metric"),
    ("dynspg_ts", "ffu_sw"): (THREADED, "precomputed EEN Coriolis triad metric"),
    ("dynspg_ts", "ffu_se"): (THREADED, "precomputed EEN Coriolis triad metric"),
    ("dynspg_ts", "ffv_nw"): (THREADED, "precomputed EEN Coriolis triad metric"),
    ("dynspg_ts", "ffv_ne"): (THREADED, "precomputed EEN Coriolis triad metric"),
    ("dynspg_ts", "ffv_sw"): (THREADED, "precomputed EEN Coriolis triad metric"),
    ("dynspg_ts", "ffv_se"): (THREADED, "precomputed EEN Coriolis triad metric"),
    ("dynspg_ts", "sshe_rhs"): (THREADED, "barotropic ssh RHS assembled in-step"),
    ("dynspg_ts", "Ue_rhs"): (THREADED, "barotropic u RHS assembled in-step"),
    ("dynspg_ts", "Ve_rhs"): (THREADED, "barotropic v RHS assembled in-step"),
    ("dynspg_ts", "CdU_u"): (THREADED, "drag at U for the barotropic solve, "
                                       "recomputed in-step from rCdU_bot"),
    ("dynspg_ts", "CdU_v"): (THREADED, "drag at V for the barotropic solve"),

    # =====================================================================
    # ldfeke
    #
    # `tradmp` USED to have `tclim`/`sclim` entries here.  They were never
    # legitimately enumerated: `tra_dmp` is DEAD (ln_tradmp = F, ocean.output)
    # and `diaobs.F90` -- the ONLY live-chain candidate that does `USE tradmp`
    # -- was itself a phantom of the by-line coverage join this gate used to
    # do (dia_obs's guard is UNRESOLVED, so it is never LIVE).  With the join
    # done by (name, ordinal) both vanish from the enumeration, so the entries
    # are stale and were removed.  Do not re-add them without first showing
    # `tradmp` back in `Enumeration.symbols`.
    # =====================================================================
    ("ldfeke", "eke_keS"): (WAIVED, "GEOMETRIC total-EKE source term, read only "
                                    "at ldfeke.F90:461 under l_ldfeke "
                                    "(ldftra.F90:96, default .FALSE., set true "
                                    "only by ln_eke_equ at :635). WAIVED not "
                                    "INERT: ln_eke_equ is not printed in "
                                    "ocean.output, so its value is not "
                                    "mechanically proven here"),

    # =====================================================================
    # traqsr
    # =====================================================================
    ("traqsr", "rktab"): (THREADED, "published RGB attenuation lookup table "
                                    "(Morel-Berthon); a constant table, not "
                                    "run state"),
    ("traqsr", "sf_chl"): (WAIVED, "fldread file-handle structure for the "
                                   "chlorophyll input field; I/O plumbing"),
}
