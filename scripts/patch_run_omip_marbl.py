"""Wire --biogeo marbl into scripts/run_omip.py (all-or-nothing; backup *.bak_marbl2).

Edits:
1. ``--biogeo`` accepts "marbl"; new options --marbl-ic, --marbl-forcing,
   --marbl-every, --marbl-batch-size, --marbl-diag-days, --marbl-write-grid.
2. Model setup: "marbl" does not build the old BiogeoConfig, so the old BGC
   init / step paths are never used for MARBL tracers.
3. main: before the old biogeo template init, call omip_coupling.setup_marbl
   (loads the MARBL initial state, builds the jitted step; with
   --marbl-write-grid writes the grid and exits).
4. _run_omip_loop: MARBL runs use the single-step JRA55 path.
5. _jra55_step: MARBL sources and sinks after the step and closure operations.

Usage (repo root):  python scripts/patch_run_omip_marbl.py
"""
import sys
from pathlib import Path

OMIP = Path("scripts/run_omip.py")

CHOICES_OLD = 'choices=["none", "abiotic", "npzd", "npzd_v2", "marbl_cesm"],'
CHOICES_NEW = 'choices=["none", "abiotic", "npzd", "npzd_v2", "marbl_cesm", "marbl"],'

ARGS_MARKER = '                   help="BGC scheme (default: none)")\n'
ARGS_INSERT = '''    p.add_argument("--marbl-ic", type=str, default=None,
                   help="MARBL initial state on the model grid (scripts/build_marbl_ic.py)")
    p.add_argument("--marbl-forcing", type=str, default=None,
                   help="Monthly dust/iron/NOx/NHy deposition and sea-ice fraction on the "
                        "model grid; without it: no deposition, ice where SST < -1 C")
    p.add_argument("--marbl-every", type=int, default=1,
                   help="Apply MARBL sources/sinks every N ocean steps (dt_bgc = N*dt)")
    p.add_argument("--marbl-batch-size", type=int, default=4096,
                   help="Water columns per MARBL batch (memory vs speed)")
    p.add_argument("--marbl-diag-days", type=float, default=30.0,
                   help="Averaging period of the MARBL 2-D diagnostics [days]")
    p.add_argument("--marbl-write-grid", type=str, default=None,
                   help="Write the model grid to this .npz (for build_marbl_ic.py) and exit")
'''

SETUP_OLD = '        if biogeo != "none":\n            from legoesm.ocean.biogeochemistry import BiogeoConfig\n'
SETUP_NEW = '        if biogeo not in ("none", "marbl"):\n            from legoesm.ocean.biogeochemistry import BiogeoConfig\n'

MAIN_MARKER = "    # Initialise biogeo on template state BEFORE loading restart,\n"
MAIN_INSERT = '''    if args.biogeo == "marbl":
        from legoesm.ocean.biogeochemistry.marbl import omip_coupling as _marbl_omip
        state = _marbl_omip.setup_marbl(args, state, grid, z_coord, model)

'''

SCAN_MARKER = "    if use_scan_blocks:\n"
SCAN_INSERT = '''    if use_scan_blocks:
        from legoesm.ocean.biogeochemistry.marbl import omip_coupling as _marbl_omip
        if _marbl_omip.active():
            print("  MARBL: using the single-step JRA55 path (scan blocks not wired for MARBL)")
            use_scan_blocks = False
'''

STEP_MARKER = '''    if jra55_state.get("enable_freeze_cap", False):
        state = _apply_freeze_cap(state, jra55_state)
'''
STEP_INSERT = '''
    # MARBL sources and sinks (transport happens inside model.step).
    from legoesm.ocean.biogeochemistry.marbl import omip_coupling as _marbl_omip
    if _marbl_omip.active():
        state = _marbl_omip.marbl_jra55_step(
            state, step_idx, dt, model, jra55_state, atm, tile_resp,
        )
'''


def main():
    if not OMIP.exists():
        sys.exit(f"{OMIP} not found -- run from the repo root; nothing changed")
    s = OMIP.read_text()
    if "omip_coupling" in s:
        sys.exit("run_omip.py already references omip_coupling (patched before?); nothing changed")
    checks = [
        (CHOICES_OLD, 1), (ARGS_MARKER, 1), (MAIN_MARKER, 1), (SCAN_MARKER, 1), (STEP_MARKER, 1),
    ]
    bad = [f"{pat.strip()[:70]!r} found {s.count(pat)}x (expected {n})"
           for pat, n in checks if s.count(pat) != n]
    n_setup = s.count(SETUP_OLD)
    if n_setup < 1:
        bad.append("model-setup 'if biogeo != \"none\":' block not found")
    if bad:
        sys.exit("not patched, nothing changed:\n  " + "\n  ".join(bad))

    s = s.replace(CHOICES_OLD, CHOICES_NEW)
    s = s.replace(ARGS_MARKER, ARGS_MARKER + ARGS_INSERT)
    s = s.replace(SETUP_OLD, SETUP_NEW)
    s = s.replace(MAIN_MARKER, MAIN_INSERT + MAIN_MARKER)
    s = s.replace(SCAN_MARKER, SCAN_INSERT + SCAN_MARKER)
    s = s.replace(STEP_MARKER, STEP_MARKER + STEP_INSERT)

    OMIP.with_name(OMIP.name + ".bak_marbl2").write_text(OMIP.read_text())
    OMIP.write_text(s)
    print(f"patched {OMIP} (backup run_omip.py.bak_marbl2; model-setup blocks changed: {n_setup})")


if __name__ == "__main__":
    main()
