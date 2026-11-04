"""Controls and RAD_CONFIG diff for the 49r1 IFS-interface oracle runs.

usage: python -I ifs49_compare.py <run dir>
  <run dir>/base_a, base_b      baseline twice (determinism: every dump bitwise equal)
  <run dir>/nan_unused, big_unused, nan_thf   controls: every OUTPUT dump bitwise equal
                                to base_a (inputs claimed unused by ecRad: aerosol arrays,
                                NO2; saturation humidity, aerosol hydration only)
  <run dir>/base_a.log, refconfig.log   RAD_CONFIG prints (SU path vs ecRad namelist)
Outputs = dumps named rs_*, sub_*, ri_*, rh_*, ifs_out_*, su_*. Prints one
verdict line per check and a RAD_CONFIG diff (every differing line); exits 1
if a check fails. NaN in any output dump of base_a is reported per array (a
NaN there means an output the oracle never wrote, or a poisoned read).
"""
import difflib
import os
import sys

OUT_PREFIX = ("rs_", "sub_", "ri_", "rh_", "ifs_out_", "su_")


def files(d):
    return {f[:-4]: open(os.path.join(d, f)).read() for f in os.listdir(d) if f.endswith(".txt")}


def config_block(log, tag="RAD_CONFIG"):
    lines = open(log).read().splitlines()
    i, j = lines.index(f"=== {tag} BEGIN"), lines.index(f"=== {tag} END")
    # gfortran's STOP note on stderr can land inside the block in a 2>&1 log
    return [l.rstrip() for l in lines[i + 1:j] if not l.startswith("Note: The following floating-point")]


def show_diff(title, a, b, la, lb):
    d = [l for l in difflib.unified_diff(a, b, la, lb, n=0, lineterm="") if not l.startswith("@@")][2:]
    print(f"{title} {len(d)} lines (- {la}, + {lb}):")
    for l in d:
        print("  " + l)


def main(rd):
    ok = True
    a, b = files(os.path.join(rd, "base_a")), files(os.path.join(rd, "base_b"))
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    print(f"DETERMINISM {'PASS' if not diff else 'FAIL'}: {len(a)} dumps, {len(diff)} differ {diff[:5]}")
    ok &= not diff and len(a) > 0
    outs = sorted(k for k in a if k.startswith(OUT_PREFIX))
    # the expected inventory (committed list, every output name): exact set equality,
    # so a run that wrote fewer, more or differently named outputs fails
    need = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ifs49_expected_outputs.txt")).read().split()
    def inventory_ok(dumps):
        got = {k for k in dumps if k.startswith(OUT_PREFIX)}
        return got == set(need), sorted(set(need) - got), sorted(got - set(need))
    def finite_ok(dumps):
        return [k for k in dumps if k.startswith(OUT_PREFIX) and ("NaN" in dumps[k] or "Infinity" in dumps[k])]
    good, miss, extra = inventory_ok(a)
    print(f"INVENTORY {'PASS' if good else 'FAIL'}: {len(outs)} outputs vs {len(need)} expected, missing {miss[:8]} extra {extra[:8]}")
    ok &= good
    nan = finite_ok(a)
    print(f"FINITE_OUTPUTS {'PASS' if not nan else 'FAIL'}: non-finite in {nan}")
    ok &= not nan
    # planted controls for the two checks above (must FAIL on a damaged copy)
    k0 = need[0]
    pl_miss = {k: v for k, v in a.items() if k != k0}
    pl_nan = dict(a); pl_nan[k0] = a[k0].split("\n", 1)[0] + "\n NaN\n"
    planted = (not inventory_ok(pl_miss)[0]) and bool(finite_ok(pl_nan))
    print(f"PLANTED inventory/finite controls {'rejected (good)' if planted else 'ACCEPTED (vacuous)'}")
    ok &= planted
    for c in ("nan_unused", "big_unused", "nan_thf"):
        d = os.path.join(rd, c)
        if not os.path.isdir(d):
            print(f"CONTROL {c} MISSING"); ok = False; continue
        x = files(d)
        bad = [k for k in outs if x.get(k) != a[k]]
        print(f"CONTROL {c} {'PASS' if not bad else 'FAIL'}: {len(outs)} outputs, {len(bad)} differ {bad[:8]}")
        ok &= not bad
        # non-vacuity: the control must have changed what it claims to perturb
        probe = {"nan_unused": "ifs_in_aerosol_mixing_ratio", "big_unused": "ifs_in_aerosol_mixing_ratio", "nan_thf": "ifs_in_thermodynamics_h2o_sat_liq"}[c]
        if probe:
            changed = probe in x and probe in a and x[probe] != a[probe]
            print(f"CONTROL {c} perturbed {probe}: {'yes' if changed else 'NO (vacuous)'}")
            ok &= changed
    # cross-check (information, not a gate): bld49 namelist config in place of SU's
    d = os.path.join(rd, "refcfg")
    if not os.path.isdir(d):
        print("REFCFG MISSING"); ok = False
    else:
        x = files(d)
        bad = [k for k in outs if x.get(k) != a[k]]
        print(f"REFCFG_CROSSCHECK {'BITWISE' if not bad else 'DIFFERS'}: {len(outs)} outputs, {len(bad)} differ {bad[:8]}")
        # non-vacuity: the configuration RADIATION_SCHEME used must really be the namelist one
        ca = config_block(os.path.join(rd, "base_a.log"), "RAD_CONFIG_DUMP_AFTER")
        cr = config_block(os.path.join(rd, "refcfg.log"), "RAD_CONFIG_DUMP_AFTER")
        swapped = [l for l in cr if l.startswith(("do_setup_ifsrrtm", "use_general_aerosol_optics"))]
        print(f"REFCFG swap applied: {'yes' if ca != cr else 'NO (vacuous)'} {swapped}")
        ok &= ca != cr and sorted(swapped) == ["do_setup_ifsrrtm = T", "use_general_aerosol_optics = T"]
    # inventory and finiteness on EVERY run, not only the baseline
    for c in ("base_b", "nan_unused", "big_unused", "nan_thf", "refcfg"):
        d = os.path.join(rd, c)
        x = files(d) if os.path.isdir(d) else {}
        g, nanc = inventory_ok(x)[0], finite_ok(x)
        if not g or nanc:
            print(f"RUN {c}: inventory {'ok' if g else 'FAIL'}, non-finite {nanc[:5]}")
        ok &= g and not nanc
    # non-triviality: outputs are not all zero, and the per-step RADHEATN inputs
    # (mu0, skin T) really change its outputs (approximate SW and LW updates exercised)
    def vals(k):
        return [float(v) for v in a[k].split("\n")[1:] if v.strip()]
    live = {k: max(abs(v) for v in vals(k)) for k in ("rs_flux_sw", "rs_flux_lw", "rh_hrsw_s0", "rh_hrlw_s0")}
    moved = {v: a[f"rh_{v}_s0"] != a[f"rh_{v}_s3"] for v in ("frso", "frth", "frsod", "frthd", "hrlw")}
    nontriv = all(x > 0 for x in live.values()) and all(moved.values())
    print(f"NONTRIVIAL {'PASS' if nontriv else 'FAIL'}: max|x| {live}; step 0 vs 3 differ {moved}")
    ok &= nontriv
    u = a.get("chk_unset_rdecorr_cf", "")
    print(f"NAN_PREFILL {'PASS' if 'NaN' in u else 'FAIL'}: unset model member reads {u.splitlines()[1:2]}")
    ok &= "NaN" in u
    la, lr = os.path.join(rd, "base_a.log"), os.path.join(rd, "refconfig.log")
    show_diff("RAD_CONFIG_PRINT_DIFF", config_block(lr), config_block(la), "config49r1_rrtmg.nam", "su_radiation_scheme")
    da, dr = config_block(la, "RAD_CONFIG_DUMP"), config_block(lr, "RAD_CONFIG_DUMP")
    print(f"RAD_CONFIG_DUMP components: {len(da)} (SU) {len(dr)} (namelist)")
    ok &= len(da) == len(dr) and len(da) > 50
    show_diff("RAD_CONFIG_DUMP_DIFF", dr, da, "config49r1_rrtmg.nam", "su_radiation_scheme")
    show_diff("RAD_CONFIG_PER_CALL_DIFF", da, config_block(la, "RAD_CONFIG_DUMP_AFTER"), "after SU", "after RADIATION_SCHEME")
    print("COMPARE_OK" if ok else "COMPARE_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1])
