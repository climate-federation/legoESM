"""F4 spread instrument, generator side: run a family's COMMITTED oracle generator
unchanged, intercepting only the subprocess call(s) to the oracle executable(s).

mode "base": the oracle receives the generator's own bytes (no rewrite).
mode "seed N": every float token of the oracle's input text (file argument or stdin)
moves one ulp via the shared perturb.perturb_inputs (zeros/NaN/inf held); integer
tokens (no '.', 'e', 'E') are untouched.  Perturbed values are written with repr()
(shortest round-trip, exact under list-directed read).

C3 per oracle call: the exact text the oracle reads is saved and read back by
readback.x (list-directed, real(8)); the readback must equal the intended values
bitwise.  The intended float sequence is saved so analyze.py can check
"moved = one adjacent step, rest identical" against the base run.

usage: gen_intercept.py FAMILY MODE OUTDIR      (MODE = base | base2 | s<N> | s<N>r)
"""
from __future__ import annotations

import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/burg/glab/users/pg2328/oracle_audit/spread")
import perturb  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from families import FAMILIES  # noqa: E402

NUM = re.compile(r"(?<![\w.+-])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?(?![\w.])")
READBACK = os.environ.get("F4_READBACK", "")


def _is_float_tok(t: str) -> bool:
    return any(c in t for c in ".eEdD")


def perturb_text(text: str, seed: int | None, call: int = 0):
    toks = list(NUM.finditer(text))
    fl = [m for m in toks if _is_float_tok(m.group())]
    vals = np.array([float(m.group().replace("d", "e").replace("D", "e")) for m in fl], float)
    if seed is None:
        return text, vals
    # VOTE V2 (b): per-call seed, call 0 keeps the seed itself
    new = perturb.perturb_inputs({"tokens": vals}, seed + 20 * call)["tokens"]
    out, last = [], 0
    for m, v0, v in zip(fl, vals, new):
        out.append(text[last:m.start()])
        out.append(m.group() if v == v0 else repr(float(v)))
        last = m.end()
    out.append(text[last:])
    return "".join(out), new


def main(family: str, mode: str, outdir: str):
    fam = FAMILIES[family]
    seed = None if mode.startswith("base") else int(mode[1:].rstrip("r"))
    od = Path(outdir) / mode
    od.mkdir(parents=True, exist_ok=True)
    real_run = subprocess.run
    calls = []
    first_call = {}   # base float values -> index of the first call that received them

    def fake(args, *a, **kw):
        if (isinstance(args, (list, tuple)) and len(args) > 1 and str(args[0]) == "bash"
                and str(args[1]).endswith("build.sh")):
            # the generator's own oracle build: trace it so C5 sees the executed binary's flags
            r = real_run(["bash", "-x", *map(str, args[1:])], *a, **{**kw, "capture_output": True,
                                                                       "text": True})
            with open(od / "genbuild.log", "a") as fh:
                fh.write(r.stdout + r.stderr)
            return r
        if not (isinstance(args, (list, tuple)) and Path(str(args[0])).name in fam["exes"]):
            return real_run(args, *a, **kw)
        i = len(calls)
        cwd = Path(kw.get("cwd") or ".")
        rec = dict(exe=Path(str(args[0])).name, sources=[])
        texts = []
        if kw.get("input") is not None:
            _txt = kw["input"]
            t, v = perturb_text(_txt, seed, first_call.setdefault(perturb_text(_txt, None)[1].tobytes(), i))
            kw["input"] = t
            texts.append(("stdin", t, v))
        elif kw.get("stdin") is not None and hasattr(kw["stdin"], "name"):
            src = Path(kw["stdin"].name)
            _txt = src.read_text()
            t, v = perturb_text(_txt, seed, first_call.setdefault(perturb_text(_txt, None)[1].tobytes(), i))
            kw["stdin"].close()
            pf = od / f"call{i}_stdin.txt"
            pf.write_text(t)
            kw["stdin"] = open(pf)
            texts.append(("stdin-file", t, v))
        for arg in list(args[1:2]):  # every file-I/O driver here: argv[1] = input, argv[2] = output
            p = Path(str(arg))
            p = p if p.is_absolute() else cwd / p
            if p.is_file():
                _txt = p.read_text()
                t, v = perturb_text(_txt, seed, first_call.setdefault(perturb_text(_txt, None)[1].tobytes(), i))
                if seed is not None:
                    p.write_text(t)
                texts.append((f"file:{arg}", t, v))
        for j, (src, t, v) in enumerate(texts):
            fin, fo = od / f"call{i}_{j}_in.txt", od / f"call{i}_{j}_rb.txt"
            fin.write_text(t)
            real_run([READBACK, str(fin), str(fo)], check=True)
            got = np.array([float(x) for x in fo.read_text().split()], float)
            want = np.array([float(m.group().replace("d", "e").replace("D", "e"))
                             for m in NUM.finditer(t)], float)
            np.save(od / f"call{i}_{j}_floats.npy", v)
            rec["sources"].append(dict(src=src, n_tokens=int(want.size), n_float=int(v.size),
                                       readback_bitwise=bool(got.size == want.size and
                                                             got.tobytes() == want.tobytes())))
            fin.unlink()
            fo.unlink()
        calls.append(rec)
        return real_run(args, *a, **kw)

    subprocess.run = fake
    out_npz = od / "fixture.npz"
    argv = [a.replace("{OUT}", str(out_npz)).replace("{MODE}", mode) for a in fam["argv"]]
    sys.argv = [fam["generator"]] + argv
    try:
        runpy.run_path(fam["generator"], run_name="__main__")
    finally:
        subprocess.run = real_run
    json.dump(dict(family=family, mode=mode, seed=seed, calls=calls),
              open(od / "calls.json", "w"), indent=1)
    print("GEN_DONE", family, mode, len(calls), "oracle calls",
          all(s["readback_bitwise"] for c in calls for s in c["sources"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
