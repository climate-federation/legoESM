#!/usr/bin/env python
"""#1455 Decision 1, Part A: empirical NEMO-vs-NEMO envelope.

Rebuild NEMO at -O0 (arch-condadbgnb, a no-fcheck variant of the port's
existing arch-condadbg.fcm -- see that file's header for why fcheck=bounds
was dropped: it trips an unrelated array-bound false positive in
usrdef_sbc.f90 under -np 1 that is orthogonal to this task) into a SEPARATE
config ``DINO_DBG`` (own ``cfgs/DINO_DBG/BLD``, never touching the
production ``-m conda`` build under ``cfgs/DINO/BLD``), re-run the SAME
restart+namelist as ``RUN_GDB`` in a sibling dir ``RUN_GDB_O0``, and diff the
per-row ``.bin`` dumps against the existing -O3 (production) RUN_GDB dumps.

Per Dhruv's 2026-08-04 Decision 1 (docs/ocean/fidelity/dino_1226_state.md,
"DECISION 1 SETTLED"): our residual <= NEMO(-O3)-vs-NEMO(-O0) is matched in
the strongest sense that exists. This script reports the envelope only --
NO classification changes happen here (that is fidelity_bar_gate.py's
CEILING_ROWS, populated separately using these numbers as one input).

Convention matches the gate's own per-row probes (ldf_slp_per_element.py /
bn2_alpha_compare.py::_load_haloed, coverage_rows_measure.py): raw ``<f8``
stream, reshape (nlev_or_1, jpj, jpi), NO halo stripped (these MY_SRC WRITE
dumps are interior-only despite nn_hls=2 in the run -- confirmed by dump
byte count == jpi*jpj*nlev exactly, no ``+2*hls`` padding). err_norm =
|O3-O0| / RMS(O3), matching the gate's own err_norm = |lego-nemo| / RMS(nemo)
definition (see fidelity_bar_gate.py PER_ELEMENT docstring, ~line 156).
ratio/corr use the same masked-Pearson / mean-ratio convention as
bn2_alpha_compare.py::_report.

Run::

    cd /home/dbalwada/legoESM && .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/nemo_o0_o3_envelope.py
"""
from __future__ import annotations

import os

import numpy as np

O3_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB"
O0_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB_O0"
# Second, independent envelope source (only used because the O0-vs-O3
# compiler-flag envelope came back degenerate/bit-identical for 7/9 rows --
# see module docstring "Degenerate-envelope caution" in the task spec).
# SAME production -m conda nemo.exe, namelist-only change: -np 4 instead of
# -np 1 (jpni/jpnj auto-resolve from mppsize, namelist_ref:1516-1517) changes
# halo-exchange/reduction order, giving a second, non-compiler-flag source of
# NEMO's own floating-point non-associativity.
NP4_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB_NP4"

JPI, JPJ = 56, 203

# name -> dump basename (present in both O3 RUN_GDB and O0 RUN_GDB_O0)
DUMPS = {
    "ldf_slp wslpi": "eiv_dump_wslpi.bin",
    "ldf_slp wslpj": "eiv_dump_wslpj.bin",
    "ldf_slp uslp": "eiv_dump_uslp.bin",
    "ldf_slp vslp": "eiv_dump_vslp.bin",
    "ldf_eiv kappa (aeiu)": "eiv_dump_aeiu.bin",
    "ssh_nxt / div_hor (hdiv)": "sshnxt_dump_hdiv.bin",
    "ssh_nxt / div_hor (ssh_after)": "sshnxt_dump_ssh_after.bin",
    "ssh_atf (ssh_before)": "atf_dump_ssh_before.bin",
    "ssh_atf (ssh_after)": "atf_dump_ssh_after.bin",
}


def _load(path: str) -> np.ndarray:
    """Raw ``<f8`` stream -> (jpj, jpi, nlev) float64, no halo strip (see
    module docstring: these dumps are byte-exact jpi*jpj*nlev, no padding)."""
    a = np.fromfile(path, dtype="<f8")
    nlev = a.size // (JPI * JPJ)
    assert nlev * JPI * JPJ == a.size, (
        f"{path}: size {a.size} not divisible by jpi*jpj={JPI * JPJ}")
    a = a.reshape(nlev, JPJ, JPI)
    return np.moveaxis(a, 0, -1)


def envelope_row(name: str, dump: str, dir_a: str = O3_DIR,
                  dir_b: str = O0_DIR) -> dict | None:
    p3 = os.path.join(dir_a, dump)
    p0 = os.path.join(dir_b, dump)
    if not (os.path.isfile(p3) and os.path.isfile(p0)):
        print(f"  {name:<40s} MISSING dump ({dump})")
        return None
    o3 = _load(p3)
    o0 = _load(p0)
    if o3.shape != o0.shape:
        print(f"  {name:<40s} SHAPE MISMATCH o3={o3.shape} o0={o0.shape}")
        return None
    m = np.isfinite(o3) & np.isfinite(o0)
    diff = np.abs(o3[m] - o0[m])
    rms3 = float(np.sqrt(np.mean(o3[m] ** 2)))
    err_norm = diff / rms3 if rms3 > 0 else diff
    nz = np.abs(o3[m]) > 0
    ratio = np.ones_like(diff)
    ratio[nz] = o0[m][nz] / o3[m][nz]
    corr = float(np.corrcoef(o0[m], o3[m])[0, 1]) if m.sum() > 1 else float("nan")
    result = dict(
        n=int(m.sum()),
        median=float(np.median(err_norm)),
        p99=float(np.percentile(err_norm, 99)),
        max=float(np.max(err_norm)),
        max_abs_diff=float(np.max(diff)),
        rms_o3=rms3,
        corr=corr,
        ratio_mean=float(ratio.mean()),
        degenerate=bool(np.max(diff) == 0.0),
    )
    tag = "DEGENERATE (bit-identical, zero envelope, uninformative)" if result["degenerate"] else ""
    print(f"  {name:<40s} median={result['median']:.3e} p99={result['p99']:.3e} "
          f"max={result['max']:.3e} corr={corr:.6f} ratio={result['ratio_mean']:.6f} "
          f"n={result['n']} {tag}")
    return result


def main() -> dict:
    print(f"O3 (production, -m conda) dir: {O3_DIR}")
    print(f"O0 (debug, -m condadbgnb) dir:  {O0_DIR}")
    print()
    print("=== Envelope A: -O0 vs -O3 (compiler-flag control, -np 1 both) ===")
    out = {}
    for name, dump in DUMPS.items():
        r = envelope_row(name, dump, O3_DIR, O0_DIR)
        if r is not None:
            out[name] = r
    n_degenerate = sum(1 for r in out.values() if r["degenerate"])
    print()
    print(f"{len(out)}/{len(DUMPS)} rows measured, {n_degenerate} degenerate "
          "(bit-identical O0 vs O3, no usable envelope).")

    if os.path.isdir(NP4_DIR):
        print()
        print("=== Envelope B: -np 4 vs -np 1 (decomposition control, both -O3/production) ===")
        print("ATTEMPTED, NOT USABLE: the MY_SRC WRITE dumps under -np 4 are")
        print("PER-RANK LOCAL-DOMAIN arrays (1x4 jpni/jpnj tiling confirmed in")
        print("ocean.output), not a global-gathered field -- e.g. eiv_dump_wslpi.bin")
        print("under NP4 is 13230 floats (236.25 x 56, a non-integer jj-slab size,")
        print("i.e. literally the last rank's local write, not the whole domain).")
        print("Reassembling per-rank subdomains into a global field to diff against")
        print("the -np1 dump would need its own gather/stitch harness (out of scope")
        print("for this task's tool budget) -- reported as attempted per the task's")
        print("explicit fallback instruction, not silently skipped.")
    return out


if __name__ == "__main__":
    main()
