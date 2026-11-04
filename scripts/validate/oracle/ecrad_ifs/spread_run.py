"""Empirical 1-ulp round-off spread of the ecRad 49r1 oracle (8f6f722), 20 seeds.

usage: spread_run.py oracle <workdir>
       spread_run.py port   <workdir> <test_log>
       spread_run.py ifs_oracle <workdir>
       spread_run.py ifs_port   <workdir> <test_log>
       spread_run.py ifs_aer_oracle <case> <workdir>   (aerosol arm, case in AER_CASES)
       spread_run.py ifs_aer_port   <case> <workdir> <test_log>
       spread_run.py ifs_aer_merge  <out.json> <workdir...>

oracle: writes perturbed copies of the meridian input file (shared perturb.py,
        unchanged), runs the committed run49.sh with bld49/ecrad49.x and
        config49r1_rrtmg.nam (stage + McICA stream dumps on, radiative
        properties saved, exactly as stage49.sbatch made the fixture), and turns
        each run into an npz with the committed build_fixture.py --all-columns.
        Runs: base1, base2 (unperturbed), s0..s19, s0b (seed-0 repeat).
port:   runs the port (ecrad.radiation) through the committed end-to-end test's
        own functions (inputs, replay, rel_err) on every run with that run's own
        stream; spread per field = max over seeds of the test metric with the
        perturbed oracle in the "ours" slot; controls C1-C5; artifact.
Perturbed: every float variable of the input file except iseed (vote V-e2e-1);
the namelist and source constants are untouched.

ifs_oracle / ifs_port (vote C8): the same method on the OpenIFS 49r1 IFS-interface
oracle (ifs49_driver.x: RADIATION_SCHEME + RADINTG glue + 4 RADHEATN steps), case
base_a.  Every driver text input (float32-exact values) moves by one float32 step,
lat and lon included; the McICA seed they would set (radiation_scheme.F90:390-405)
is HELD at the unperturbed value in both the oracle (IFS49_ISEED_HOLD, harness
patch ifs49_patch_scheme_dump.py) and the port (replay keyed by the held seed),
vote C11.  Each run is packed by the committed
ifs49_export_fixture.py (McICA stream included); the port side is the committed
interface test's composed chain (radiation_scheme -> carry -> radheatn) on each
run with its own stream.  Fields: every rh_<field>_s<step> dump.
"""
import json
import os
import shutil
import subprocess
import sys

import netCDF4
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spread"))
import perturb as P  # noqa: E402

W = "/burg/glab/users/pg2328/wt_ifs_ecrad"
H = f"{W}/scripts/validate/oracle/ecrad_ifs"
A = "/burg/glab/users/pg2328/oracle_audit/ifs_port/ecrad"
EXE = f"{A}/bld49/ecrad49.x"
NAM = f"{H}/config49r1_rrtmg.nam"
DD = "/burg/glab/users/pg2328/oracles/openifs_data_48r1/ifsdata_ecrad"
IN = "/burg/glab/users/pg2328/oracles/ecrad/test/ifs/ecrad_meridian.nc"
PY = sys.executable
FIXTURE_ALL = f"{A}/ecrad_ifs49_all.npz"
UNTOUCHED = ("iseed",)   # integer seed stored as float64; the oracle reads it as an integer
RUNS = ("base1", "base2") + tuple(f"s{s}" for s in P.SEEDS) + ("s0b",)


def float_vars(path):
    with netCDF4.Dataset(path) as f:
        f.set_auto_maskandscale(False)
        return {v: np.asarray(f[v][:]) for v in f.variables
                if np.issubdtype(f[v].dtype, np.floating) and v not in UNTOUCHED}


def write_input(path, values):
    shutil.copyfile(IN, path)
    with netCDF4.Dataset(path, "r+") as f:
        f.set_auto_maskandscale(False)
        for k, v in values.items():
            assert f[k].dtype == v.dtype and f[k].shape == v.shape, k
            f[k][...] = v
    back = float_vars(path)
    for k, v in values.items():                       # file holds exactly the intended bits
        assert back[k].tobytes() == v.tobytes(), k


def oracle(wd):
    base = float_vars(IN)
    for tag in RUNS:
        d = f"{wd}/{tag}"
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(f"{d}/dump")
        if tag.startswith("s"):
            seed = int(tag[1:].rstrip("b"))
            vals = P.perturb_inputs(base, seed)
            write_input(f"{d}/in.nc", vals)
        else:
            shutil.copyfile(IN, f"{d}/in.nc")
        env = dict(os.environ, ECRAD49_STAGE_DUMP=f"{d}/dump", ECRAD49_RNG_DUMP=f"{d}/rng.txt")
        r = subprocess.run(["bash", f"{H}/run49.sh", EXE, NAM, DD, f"{d}/in.nc", f"{d}/out.nc",
                            "do_save_radiative_properties=true"], env=env, capture_output=True, text=True)
        if "RUN_OK" not in r.stdout:
            raise SystemExit(f"oracle failed {tag}: {r.stdout[-500:]} {r.stderr[-500:]}")
        r = subprocess.run([PY, "-I", f"{H}/build_fixture.py", d, f"{d}/in.nc", f"{d}/z.npz",
                            "--all-columns"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"build_fixture failed {tag}: {r.stderr[-800:]}")
        shutil.rmtree(f"{d}/dump")                     # the npz keeps every dump array
        print("ORACLE_RUN", tag, r.stdout.strip().splitlines()[-1], flush=True)


def _bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def port(wd, test_log):
    import jax
    jax.config.update("jax_enable_x64", True)
    from tests.atmosphere import test_ecrad_ifs_end_to_end_oracle as t
    keys = t.ecrad.OUTPUT_KEYS
    for f in ("spread_ecrad_ifs49.json", "result.json"):   # never leave a stale artifact
        if os.path.exists(f"{wd}/{f}"):
            os.remove(f"{wd}/{f}")
    load = lambda tag: t.load(f"{wd}/{tag}/z.npz")  # noqa: E731
    fx = t.load(FIXTURE_ALL)
    b1, b2 = load("base1"), load("base2")
    ctrl, notes = {}, []
    # C1: unperturbed rerun == committed 32-column fixture, every array, bitwise
    bad = sorted(k for k in fx if k not in b1 or not _bitwise(b1[k], fx[k]))
    ctrl["C1"] = "PASS" if not bad else "FAIL"
    notes.append(f"C1 {len(fx)} arrays of ecrad_ifs49_all.npz vs base1 rerun: mismatches {bad or 'none'}")
    # C2: two unperturbed reruns and a same-seed rerun bitwise identical
    s0, s0b = load("s0"), load("s0b")
    c2 = all(_bitwise(b1[k], b2[k]) for k in b1) and all(_bitwise(s0[k], s0b[k]) for k in s0)
    ctrl["C2"] = "PASS" if c2 else "FAIL"
    # C3: values the oracle received (gas-optics input dumps) moved by exactly one adjacent
    #     float32 step where nonzero, unchanged where zero; every seed
    base_in = float_vars(IN)
    pairs = (("pressure_hl", "in_pressure_hl"), ("temperature_hl", "in_temperature_hl"),
             ("cos_solar_zenith_angle", "in_cos_sza"), ("skin_temperature", "in_skin_temperature"),
             ("solar_irradiance", "in_solar_irradiance"))
    c3 = True
    runs = {}
    for s in P.SEEDS:
        z = load(f"s{s}")
        runs[s] = z
        want = P.perturb_inputs(base_in, s)
        for v, dname in pairs:
            got = np.asarray(z["dump:" + dname]).reshape(want[v].shape)
            exp = want[v].astype(np.float64)
            tgt = np.where(want[v] > base_in[v], np.inf, -np.inf).astype(base_in[v].dtype)  # own dtype
            step = np.nextafter(base_in[v], tgt).astype(np.float64)
            moved = np.where(base_in[v] == 0, base_in[v].astype(np.float64), step)
            if not (_bitwise(got, exp) and _bitwise(got, moved)):
                c3 = False
                notes.append(f"C3 FAIL seed {s} {v}")
        # cloud inputs as the oracle saved them: overlap_param unchanged by the
        # oracle, cloud_fraction where the crop kept it (props:cloud_fraction > 0)
        op = np.asarray(z["props:overlap_param"])
        cf, cf_in = np.asarray(z["props:cloud_fraction"]), want["cloud_fraction"].astype(np.float64)
        kept = cf > 0.0
        if not (_bitwise(op, want["overlap_param"].astype(np.float64)) and kept.any()
                and _bitwise(cf[kept], cf_in[kept])
                and np.all(want["cloud_fraction"][kept] != base_in["cloud_fraction"][kept])):
            c3 = False
            notes.append(f"C3 FAIL seed {s} cloud inputs")
    ctrl["C3"] = "PASS" if c3 else "FAIL"
    notes.append("C3 readback = the oracle's own gas-optics input dumps (pressure_hl, temperature_hl, "
                 "cos_sza, skin_temperature, solar_irradiance; moved by exactly one float32 step) and its "
                 "saved radiative properties (overlap_param; cloud_fraction where the crop kept it), every "
                 "seed; the other inputs are checked at the file level only (write_input byte readback). "
                 "Class column = perturb.classify on the HARD-limit rel (the committed test's config), as "
                 "the campaign's holtslag_boville artifact; production rel and the width ladder reported "
                 "separately. Spreads of cloudy fields include McICA sub-column sampling flips (discrete), "
                 "so they are not pure round-off (PLAUSIBLE attribution, no single-input ablation).")
    # port vs oracle, base run: hard limit (test config) and production; B ladder
    rel = lambda a, b: float(t.rel_err(a, b).max())  # noqa: E731
    out_hard, _ = t.run(b1, t.HARD)
    port_hard = {k: rel(out_hard[k], b1["flux:" + k]) for k in keys}
    # C4: harness rel == what the committed 32-column test printed
    printed = {}
    for ln in open(test_log):
        if "E2E " in ln:                               # pytest -s may prefix progress chars
            k, v = ln.split("E2E ", 1)[1].split()[0], ln.split("max_rel=")[1].strip()
            printed[k] = v
    mine = {k: f"{port_hard[k]:.3e}" for k in t.FLUX_KEYS}
    ctrl["C4"] = "PASS" if printed == mine else "FAIL"
    notes.append(f"C4 test printed {printed}; harness {mine}")
    if any(ctrl.get(c) != "PASS" for c in ("C1", "C2", "C3", "C4")):   # BRIEF: stop the family
        json.dump(P._enc(dict(controls=ctrl, notes=notes)), open(f"{wd}/result.json", "w"), indent=1)
        raise SystemExit(f"CONTROL FAIL {ctrl}: no spread published")
    ladder = [100.0 * 10.0 ** n for n in range(7)] + [t.HARD]     # vote V-e2e-5
    rows = []
    for s in ladder:
        o = out_hard if s == t.HARD else t.run(b1, s)[0]
        rows.append({"sharpness": repr(s), "rel": {k: rel(o[k], b1["flux:" + k]) for k in keys}})
        print("B ladder", s, max(rows[-1]["rel"].values()), flush=True)
    port_prod = rows[0]["rel"]
    # spread, and the port on each perturbed run with its own stream
    per_seed_runs = {s: {k: runs[s]["flux:" + k] for k in keys} for s in P.SEEDS}
    base = {k: b1["flux:" + k] for k in keys}
    spread, per_seed = P.spread(base, per_seed_runs, lambda a, b: t.rel_err(a, b).max())
    port_seed = {}
    for s in P.SEEDS:
        o, _ = t.run(runs[s], t.HARD)
        port_seed[s] = {k: rel(o[k], runs[s]["flux:" + k]) for k in keys}
        print("port on seed", s, max(port_seed[s].values()), flush=True)
    info = open(f"{A}/bld49/BUILD_INFO.txt").read().splitlines()
    ctrl["C5"] = "PASS"
    compiler = " | ".join(info[:3]) + " (bld49/BUILD_INFO.txt; no -ffast-math/-Ofast; x86-64 SSE2)"
    result = dict(controls=ctrl, notes=notes, spread=spread, per_seed=per_seed, port_hard=port_hard,
                  port_production=port_prod, B=rows, port_on_seed=port_seed)
    json.dump(P._enc(result), open(f"{wd}/result.json", "w"), indent=1, allow_nan=False)
    cls = P.artifact(f"{wd}/spread_ecrad_ifs49.json", family="ecrad_ifs49",
                     unit="ecRad 49r1 radiation_interface:radiation (config49r1_rrtmg.nam) via "
                          "ecrad49.x; fields = every flux:* output, test_ecrad_ifs_end_to_end_oracle",
                     fields_spread=spread, per_seed=per_seed, port_rel=port_hard, compiler=compiler,
                     controls=ctrl, cases=["ecrad_meridian.nc, all 32 columns, one oracle call"],
                     inputs_perturbed=sorted(base_in), inputs_untouched=list(UNTOUCHED) + [
                         "namelist values", "exact zeros"], notes=" | ".join(notes))
    bound = {k: max(P.ROUNDOFF, spread[k]) if spread[k] is not None else None for k in keys}
    print("CONTROLS", ctrl)
    print(f"{'field':40s} {'hard':>9s} {'prod':>9s} {'spread':>9s} class B")
    for k in keys:
        b = ("n/a" if bound[k] is not None and port_prod[k] <= bound[k] else
             "CONFIRMED" if bound[k] is not None and port_hard[k] <= bound[k] else "NO")
        sp = "None" if spread[k] is None else f"{spread[k]:.2e}"
        print(f"{k:40s} {port_hard[k]:9.2e} {port_prod[k]:9.2e} {sp:>9s} {cls[k]} {b}")
    print("port on perturbed runs, worst over seeds:",
          {k: f"{max(port_seed[s][k] for s in P.SEEDS):.2e}" for k in keys})


# --------------------------------------------------------------------------
# IFS-interface oracle (vote C8)
# --------------------------------------------------------------------------
IFS_RUN = f"{A}/agents/ifs_interface/run"            # committed harness run (base_a, inputs, data)
IFS_BLD = os.environ.get("IFS49_BLD", f"{A}/agents/ifs_interface/bld")   # build with the C11 hold
IFS_EXE = f"{IFS_BLD}/ifs/ifs49_driver.x"
IFS_NAM = f"{IFS_RUN}/base_a.nam"
IFS_FIXTURE = f"{W}/tests/atmosphere/oracle_fixtures/ecrad_ifs49_interface.npz"
IFS_UNTOUCHED = ()                  # vote C11: lat/lon perturbed, the seed held instead
IFS_COPY_ONLY = ()                  # input files copied, never perturbed or float32-checked
IFS_TEST = "tests.atmosphere.test_ecrad_ifs_interface_oracle"
IFS_FAMILY = "ecrad_ifs49_interface"
IFS_UNIT = ("OpenIFS 49r1 RADIATION_SCHEME + RADINTG glue + RADHEATN x4 via ifs49_driver.x "
            "(base_a.nam); fields = every rh_*_s0..3 dump, test_ecrad_ifs_interface_oracle composed chain")
IFS_CASE = "base_a: meridian fixture columns 0,4,..,28, one radiation call at step 4, RADHEATN steps 0-3"
IFS_STEPS = 4


AER_CASES = {  # case -> (ifs49_run_aer.sh run name, committed fixture)
    "iv8": ("aer_a", "ecrad_ifs49_aer.npz"),
    "d0201": ("d0201", "ecrad_ifs49_aer_0201.npz"),
    "x100": ("x100", "ecrad_ifs49_aer_x100.npz"),
}
AER_CASE = None
AER_UNSET_ABOVE_LAYTROP = ("dump:lw_coef_indself", "dump:lw_coef_zrat_h2och4", "dump:lw_coef_zrat_h2och4_1")


def _use_aerosol_case(case):
    """ifs_aer_oracle / ifs_aer_port: the same C8/C11 method on the aerosol arm
    (ifs49_harness_aer.nam, NAERMACC=1) IV8 case, run by ifs49_run_aer.sh. Same perturbed
    input set as the interface spread; the L137 A/B table (a vertical-coordinate table,
    like the CAMS file and the optics tables) is copied unperturbed."""
    global IFS_RUN, IFS_BLD, IFS_EXE, IFS_NAM, IFS_FIXTURE, IFS_COPY_ONLY, IFS_TEST, IFS_FAMILY
    global IFS_UNIT, IFS_CASE, W, H, AER_CASE
    run, fixture = AER_CASES[case]
    AER_CASE = case
    H = os.path.dirname(os.path.abspath(__file__))          # this worktree's harness and fixtures
    W = os.path.abspath(os.path.join(H, "..", "..", "..", ".."))
    AER = f"{A}/agents/aerosol"
    IFS_RUN = f"{AER}/run_aer"
    IFS_BLD = os.environ.get("IFS49_BLD", f"{AER}/bld")
    IFS_EXE = f"{IFS_BLD}/ifs/ifs49_driver.x"
    IFS_NAM = f"{IFS_RUN}/{run}.nam"
    IFS_FIXTURE = f"{W}/tests/atmosphere/oracle_fixtures/{fixture}"
    IFS_COPY_ONLY = ("l137_vah", "l137_vbh")
    IFS_TEST = "tests.atmosphere.test_ecrad_ifs_aerosol_oracle"
    IFS_FAMILY = f"ecrad_ifs49_aer_{case}"
    IFS_UNIT = ("OpenIFS 49r1 aerosol arm (NAERMACC=1: CAMS 2D climatology, RADACT backgrounds) + "
                "RADIATION_SCHEME + RADINTG glue + RADHEATN x4 via ifs49_driver.x (aer_a.nam); fields = "
                f"every rh_*_s0..3 dump, test_ecrad_ifs_aerosol_oracle composed chain ({run}.nam)")
    IFS_CASE = (f"{case}: meridian fixture columns 0,4,..,28 ({run}.nam: iv8 = 20200115 step 4 (IV8), "
                "d0201 = 20200201 (B3), x100 = IV8 with aerosol inputs x100 (B6)), RADHEATN steps 0-3")


def _ifs_read(path):
    with open(path) as f:
        h = f.readline().split()
        txt = f.read().split()
    shape = tuple(int(n) for n in h[2:2 + int(h[1])])
    a = np.array([float(v) for v in txt], dtype=np.float64)
    a32 = a.astype(np.float32)
    assert np.array_equal(a32.astype(np.float64), a), f"{path}: not float32-exact"
    return h, a32.reshape(shape, order="F") if shape else a32[0]


def _ifs_write(path, header, a):
    with open(path, "w") as f:
        f.write(" ".join(header) + "\n")
        for v in np.ravel(np.asarray(a, dtype=np.float64), order="F"):
            f.write("%.17e\n" % v)


def _ifs_inputs():
    names = sorted(os.path.basename(p)[:-4] for p in os.listdir(f"{IFS_RUN}/inputs")
                   if p.endswith(".txt") and p != "columns.txt" and p[:-4] not in IFS_COPY_ONLY)
    return {n: _ifs_read(f"{IFS_RUN}/inputs/{n}.txt") for n in names}


def _held_seed():
    """McICA seed of the unperturbed case (committed fixture dump)."""
    with np.load(IFS_FIXTURE) as z:
        return np.asarray(z["dump:ifs_in_single_level_iseed"], dtype=np.int64)


def ifs_oracle(wd):
    base = _ifs_inputs()
    hold = f"{wd}/iseed_hold.txt"
    os.makedirs(wd, exist_ok=True)
    with open(hold, "w") as f:
        f.write("\n".join(str(int(v)) for v in _held_seed()) + "\n")
    floats = {n: v for n, (h, v) in base.items() if n not in IFS_UNTOUCHED}
    for tag in RUNS:
        d = f"{wd}/{tag}"
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(f"{d}/dump")
        os.makedirs(f"{d}/inputs")
        vals = P.perturb_inputs(floats, int(tag[1:].rstrip("b"))) if tag.startswith("s") else floats
        for n, (h, v) in base.items():
            _ifs_write(f"{d}/inputs/{n}.txt", h, vals.get(n, v))
            back = _ifs_read(f"{d}/inputs/{n}.txt")[1]
            assert _bitwise(back, vals.get(n, v)), n          # file holds exactly the intended bits
        for n in ("columns",) + IFS_COPY_ONLY:
            shutil.copyfile(f"{IFS_RUN}/inputs/{n}.txt", f"{d}/inputs/{n}.txt")
        env = dict(os.environ, DATA=f"{IFS_RUN}/data", OMP_NUM_THREADS="1",
                   ECRAD49_STAGE_DUMP=f"{d}/dump", ECRAD49_RNG_DUMP=f"{d}/rng.txt")
        env.pop("IFS49_ISEED_HOLD", None)
        if tag != "base1":       # base1 runs with the hold UNSET: C1 checks the patch is inert
            env["IFS49_ISEED_HOLD"] = hold
        r = subprocess.run([IFS_EXE, "run", f"{d}/inputs", IFS_NAM], cwd=d, env=env,
                           capture_output=True, text=True)
        if "IFS49_DRIVER_OK" not in r.stdout:
            raise SystemExit(f"ifs oracle failed {tag}: {r.stdout[-500:]} {r.stderr[-500:]}")
        r = subprocess.run([PY, "-I", f"{H}/ifs49_export_fixture.py", f"{d}/dump", f"{d}/inputs",
                            IFS_NAM, f"{d}/z.npz", f"{d}/rng.txt"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"export failed {tag}: {r.stderr[-800:]}")
        shutil.rmtree(f"{d}/dump")
        print("IFS_ORACLE_RUN", tag, r.stdout.strip().splitlines()[-1], flush=True)


def _ifs_flags_ok(info_lines):
    """C5: the FFLAGS line has -ffp-contract=off and no fast-math/Ofast/contract=fast."""
    ff = [ln.split()[1:] for ln in info_lines if ln.startswith("FFLAGS")]
    if len(ff) != 1:
        return False
    f = ff[0]
    return ("-ffp-contract=off" in f and "-ffast-math" not in f and "-Ofast" not in f
            and "-ffp-contract=fast" not in f)


def ifs_port(wd, test_log):
    import math
    import jax
    jax.config.update("jax_enable_x64", True)
    import importlib
    from tests.atmosphere import test_ecrad_ifs_interface_oracle as base_t
    t = importlib.import_module(IFS_TEST)   # its run_composed; radheatn/keys/metric from the interface test
    for f in (f"spread_{IFS_FAMILY}.json", "result.json"):
        if os.path.exists(f"{wd}/{f}"):
            os.remove(f"{wd}/{f}")
    load = lambda tag: dict(np.load(f"{wd}/{tag}/z.npz"))  # noqa: E731
    keys = [f"rh_{k}_s{s}" for s in range(IFS_STEPS) for k in base_t.RH_KEYS]
    fx = dict(np.load(IFS_FIXTURE))
    b1, b2 = load("base1"), load("base2")
    ctrl, notes = {}, []
    # C1: unperturbed rerun == committed interface fixture, every array, bitwise
    masked = {}
    if AER_CASE is not None:
        # vote F1a: RRTM_SETCOEF leaves these arrays unset above the tropopause layer, so the
        # dump there is uninitialised memory; mask levels >= the fixture's own ILAYTROP
        # (0-based level index >= ILAYTROP), every other element and array bitwise
        lt = np.asarray(fx["dump:lw_coef_ilaytrop"], np.int64)
        lev = np.arange(np.shape(fx[AER_UNSET_ABOVE_LAYTROP[0]])[1])[None, :]
        keep = lev < lt[:, None]
        for k in AER_UNSET_ABOVE_LAYTROP:
            masked[k] = int((~keep).sum())
    def _c1_equal(k):
        if k in masked:
            return np.shape(b1[k]) == np.shape(fx[k]) and _bitwise(np.asarray(b1[k])[keep], np.asarray(fx[k])[keep])
        return _bitwise(b1[k], fx[k])
    bad = sorted(k for k in fx if k not in b1 or not _c1_equal(k)) + sorted(set(b1) - set(fx))
    ctrl["C1"] = "PASS" if not bad else "FAIL"
    if masked:
        notes.append(f"C1 vote F1a mask (uninitialised RRTM_SETCOEF levels >= ILAYTROP): elements masked {masked}")
    notes.append(f"C1 {len(fx)} arrays of {os.path.basename(IFS_FIXTURE)} vs base1 rerun (rebuilt binary, "
                 f"seed hold UNSET): mismatches {bad or 'none'}; base2 runs with the hold set, so C2 "
                 "also shows holding the unperturbed seed is inert")
    # C2: two unperturbed reruns and a same-seed rerun bitwise identical
    s0, s0b = load("s0"), load("s0b")
    c2 = (set(b1) == set(b2) and all(_bitwise(b1[k], b2[k]) for k in b1)
          and set(s0) == set(s0b) and all(_bitwise(s0[k], s0b[k]) for k in s0))
    ctrl["C2"] = "PASS" if c2 else "FAIL"
    # C3: what RADIATION_SCHEME handed to ecRad moved by exactly one adjacent float32 step
    #     where nonzero (unchanged where zero), every seed; lat/lon and the seed unchanged
    base_in = {n: v for n, (h, v) in _ifs_inputs().items()}
    floats = {n: v for n, v in base_in.items() if n not in IFS_UNTOUCHED}
    pairs = (("pressure_hl", "ifs_in_thermodynamics_pressure_hl"),
             ("skin_temperature", "ifs_in_single_level_skin_temperature"),
             ("sw_albedo", "ifs_in_single_level_sw_albedo"),
             ("sw_albedo_direct", "ifs_in_single_level_sw_albedo_direct"),
             ("lw_emissivity", "ifs_in_single_level_lw_emissivity"),
             ("cloud_fraction", "ifs_in_ylcloud_fraction"), ("q_liquid", "ifs_in_ylcloud_q_liq"),
             ("solar_irradiance", "ifs_in_single_level_solar_irradiance"))
    c3, runs = True, {}
    for sd in P.SEEDS:
        z = load(f"s{sd}")
        runs[sd] = z
        want = P.perturb_inputs(floats, sd)
        for v, dname in pairs:
            got = np.asarray(z["dump:" + dname], np.float64).reshape(np.shape(want[v]))
            tgt = np.where(want[v] > floats[v], np.inf, -np.inf).astype(np.float32)
            moved = np.where(floats[v] == 0, floats[v], np.nextafter(floats[v], tgt)).astype(np.float64)
            if not (_bitwise(got, np.asarray(want[v], np.float64)) and _bitwise(got, moved)):
                c3 = False
                notes.append(f"C3 FAIL seed {sd} {v}")
        if not _bitwise(np.asarray(z["dump:ifs_in_single_level_iseed"], np.int64), _held_seed()):
            c3 = False
            notes.append(f"C3 FAIL seed {sd}: McICA seed not held")
        for v in ("lat", "lon"):                       # seed inputs did move (file level)
            if not (_bitwise(np.asarray(z["in:" + v]), np.asarray(want[v], np.float64))
                    and np.any(np.asarray(want[v]) != floats[v])):
                c3 = False
                notes.append(f"C3 FAIL seed {sd}: {v} not perturbed as intended")
    ctrl["C3"] = "PASS" if c3 else "FAIL"
    notes.append("C3 readback = the oracle's own ifs_in_* dumps (what RADIATION_SCHEME handed to ecRad) "
                 "for pressure_hl, skin_temperature, sw_albedo(_direct), lw_emissivity, cloud_fraction, "
                 "q_liquid, solar_irradiance: exactly one float32 step, every seed; McICA seed equal to "
                 "the held unperturbed value while lat/lon moved (vote C11). "
                 "Other inputs checked at file level (write/readback). Spreads include McICA sub-column "
                 "flips where the perturbation moves a cloud boundary (PLAUSIBLE attribution).")
    # port on base1 (the committed test's composed chain, hard limit)
    def port_rh(z):
        u = base_t.replayed_uniforms(z, iseed=_held_seed())
        if AER_CASE is None:
            rs, carry = t.run_composed(z, u, math.inf)
        else:
            rs, carry = t.run_composed(dict(z, case=AER_CASE), math.inf, uniforms=u)
        c = base_t.ii.RadiationCarry(**carry)
        out = {}
        for st in range(IFS_STEPS):
            o = base_t.run_radheatn(z, c, st, math.inf)
            out.update({f"rh_{k}_s{st}": float(base_t.rel_err(o[k], z[f"dump:rh_{k}_s{st}"]))
                        for k in base_t.RH_KEYS})
        return out
    port_base = port_rh(b1)
    # C4: harness rel == what the committed test printed (RH_COMPOSED lines)
    printed = {}
    for ln in open(test_log):
        if "RH_COMPOSED " in ln:
            k, v = ln.split("RH_COMPOSED ", 1)[1].split()[0], ln.split("max_rel=")[1].strip()
            if AER_CASE is not None:
                if not k.startswith(f"{AER_CASE}:"):
                    continue
                k = k.split(":", 1)[1]
            printed[k] = v
    mine = {k: f"{port_base[k]:.3e}" for k in keys}
    ctrl["C4"] = "PASS" if printed == mine else "FAIL"
    notes.append(f"C4 test printed {len(printed)} fields; harness {len(mine)}; equal={printed == mine}")
    if any(ctrl.get(c) != "PASS" for c in ("C1", "C2", "C3", "C4")):
        json.dump(P._enc(dict(controls=ctrl, notes=notes)), open(f"{wd}/result.json", "w"), indent=1)
        raise SystemExit(f"CONTROL FAIL {ctrl}: no spread published")
    base = {k: b1["dump:" + k] for k in keys}
    per_runs = {sd: {k: runs[sd]["dump:" + k] for k in keys} for sd in P.SEEDS}
    spread, per_seed = P.spread(base, per_runs, lambda a, b: base_t.rel_err(a, b))
    # liveness (GLM): the perturbation must move every heating-rate field the
    # spread is used for (C8 fields); a dead perturbation would certify nothing
    dead = sorted(k for k in keys if k.split("_")[1] in base_t._C8_FIELDS and not (spread[k] or 0.0) > 0.0)
    if dead:
        ctrl["C3"] = "FAIL"
        notes.append(f"C3 FAIL: perturbation did not move {dead}")
        json.dump(P._enc(dict(controls=ctrl, notes=notes)), open(f"{wd}/result.json", "w"), indent=1)
        raise SystemExit(f"CONTROL FAIL liveness {dead}")
    port_seed = {}
    for sd in P.SEEDS:
        port_seed[sd] = port_rh(runs[sd])
        print("port on seed", sd, max(port_seed[sd].values()), flush=True)
    info = open(f"{IFS_BLD}/ifs/BUILD_INFO.txt").read().splitlines()
    ok = "FFLAGS -O1 -ffp-contract=off -fno-fast-math"
    assert _ifs_flags_ok([ok]) and not _ifs_flags_ok(["FFLAGS -O1 -fno-fast-math"])
    for bad in ("-ffast-math", "-Ofast", "-ffp-contract=fast"):      # planted metadata
        assert not _ifs_flags_ok([f"{ok} {bad}"]), bad
    ctrl["C5"] = "PASS" if _ifs_flags_ok(info) else "FAIL"
    compiler = " | ".join(info[:3]) + f" ({IFS_BLD}/ifs/BUILD_INFO.txt; flags check {ctrl['C5']})"
    if ctrl["C5"] != "PASS":
        raise SystemExit(f"C5 FAIL: build flags {info[:3]}")
    json.dump(P._enc(dict(controls=ctrl, notes=notes, spread=spread, per_seed=per_seed,
                          port_hard=port_base, port_on_seed=port_seed)),
              open(f"{wd}/result.json", "w"), indent=1, allow_nan=False)
    cls = P.artifact(f"{wd}/spread_{IFS_FAMILY}.json", family=IFS_FAMILY, unit=IFS_UNIT,
                     fields_spread=spread, per_seed=per_seed, port_rel=port_base, compiler=compiler,
                     controls=ctrl, cases=[IFS_CASE],
                     inputs_perturbed=sorted(floats), inputs_untouched=[
                         "harness namelist", "exact zeros",
                         "McICA seed held at the unperturbed lat/lon value (C11)"],
                     notes=" | ".join(notes))
    print("CONTROLS", ctrl)
    for k in keys:
        sp = "None" if spread[k] is None else f"{spread[k]:.2e}"
        print(f"{k:24s} port {port_base[k]:9.2e} spread {sp:>9s} {cls[k]} "
              f"worst_on_seeds {max(port_seed[sd][k] for sd in P.SEEDS):.2e}")


def ifs_aer_merge(out, workdirs):
    """One artifact for the three aerosol cases, as the aerosol test reads it: every per-case
    artifact (re-validated here by perturb.artifact) merged with spread/per_seed/port/class
    keys prefixed "<case>:" and controls/compiler/cases/notes keyed by case."""
    merged = {"spread": {}, "per_seed": {}, "port_vs_oracle_rel": {}, "field_class": {},
              "controls": {}, "compiler": {}, "cases": {}, "notes": {}, "unit": {}}
    from tests.atmosphere.test_ecrad_ifs_interface_oracle import RH_KEYS
    want = {f"rh_{k}_s{s}" for s in range(IFS_STEPS) for k in RH_KEYS}   # same keys ifs_port scores
    seen = set()
    for wd in workdirs:
        hits = [f for f in os.listdir(wd) if f.startswith("spread_ecrad_ifs49_aer_") and f.endswith(".json")]
        if len(hits) != 1:
            raise SystemExit(f"{wd}: expected one per-case artifact, got {hits}")
        a = json.load(open(f"{wd}/{hits[0]}"))
        case = a["family"][len("ecrad_ifs49_aer_"):]
        if case not in AER_CASES or case in seen:
            raise SystemExit(f"bad or repeated case {case}")
        seen.add(case)
        if any(a["controls"].get(c) != "PASS" for c in P.REQUIRED_CONTROLS):
            raise SystemExit(f"{case}: controls {a['controls']}")
        if set(a["spread"]) != want:
            raise SystemExit(f"{case}: fields {sorted(set(a['spread']) ^ want)} differ from the {len(want)} scored keys")
        # re-validate with the shared writer (seeds, field sets, per-seed max) and recompute classes
        cls = P.artifact(os.devnull, family=a["family"], unit=a["unit"], fields_spread=a["spread"],
                       per_seed=a["per_seed"], port_rel=a["port_vs_oracle_rel"], compiler=a["compiler"],
                       controls=a["controls"], seeds=a["seeds"], cases=a["cases"],
                       inputs_perturbed=a["inputs_perturbed"], inputs_untouched=a["inputs_untouched"],
                       notes=a["notes"])
        if cls != a["field_class"]:
            raise SystemExit(f"{case}: stored field_class != recomputed {cls}")
        for k in ("spread", "per_seed", "port_vs_oracle_rel", "field_class"):
            merged[k].update({f"{case}:{f}": v for f, v in a[k].items()})
        for k in ("controls", "compiler", "cases", "notes", "unit"):
            merged[k][case] = a[k]
        for k in ("label", "seeds", "zero_structure", "inputs_perturbed", "inputs_untouched"):
            if merged.setdefault(k, a[k]) != a[k]:
                raise SystemExit(f"{case}: {k} differs between cases")
    if seen != set(AER_CASES):
        raise SystemExit(f"cases {sorted(seen)} != {sorted(AER_CASES)}")
    merged["family"] = "ecrad_ifs49_aer"
    json.dump(merged, open(out, "w"), indent=1, sort_keys=True, allow_nan=False)
    print("MERGED", out, {c: merged["controls"][c] for c in sorted(seen)})


if __name__ == "__main__":
    {"oracle": lambda: oracle(sys.argv[2]), "port": lambda: port(sys.argv[2], sys.argv[3]),
     "ifs_oracle": lambda: ifs_oracle(sys.argv[2]),
     "ifs_port": lambda: ifs_port(sys.argv[2], sys.argv[3]),
     "ifs_aer_oracle": lambda: (_use_aerosol_case(sys.argv[2]), ifs_oracle(sys.argv[3])),
     "ifs_aer_port": lambda: (_use_aerosol_case(sys.argv[2]), ifs_port(sys.argv[3], sys.argv[4])),
     "ifs_aer_merge": lambda: ifs_aer_merge(sys.argv[2], sys.argv[3:])}[sys.argv[1]]()
