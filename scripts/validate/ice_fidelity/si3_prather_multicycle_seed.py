"""One 1-ulp seed for a PM multi-cycle arm (tier-4 seed_dyn.py semantics, shared perturb.py): perturb every recorded
float input of rhg/adv/rdg/cor of the arm's base record (zeros held), reload them at each routine boundary in a fresh
copy of the tier-4 run template whose restart delayed CFL is set to the arm's value; run to the cor record and stop;
C3 readback.  usage: si3_prather_multicycle_seed.py <arm cyc2|cyc3> <seed>   (one 32-rank NEMO run; sbatch --account=glab)"""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, "/burg/glab/users/pg2328/oracle_audit/spread"); import perturb  # noqa: E402
sys.path.insert(0, "/burg/glab/users/pg2328/oracle_audit/tier4/dyn"); import t4lib as L, t3rec  # noqa: E401,E402
P = "/burg/glab/users/pg2328/oracle_audit/si3dyn_fix/PM"
arm, seed = sys.argv[1], int(sys.argv[2]); CFL = {"cyc2": "1.0", "cyc3": "2.0"}[arm]; CH = ("rhg", "adv", "rdg", "cor")
base = {c: L.load(f"{P}/{arm}", c) for c in CH}
flat = {f"{c}:{r}:{k}": base[c][r][2][k] for c in CH for r in base[c] for k in base[c][r][3]}
pert = perturb.perturb_inputs(flat, seed)
reld = f"{P}/seeds/reload_{arm}_{seed:02d}"; os.makedirs(reld, exist_ok=True)
for c in CH:
    for r in base[c]:
        order = base[c][r][3]
        t3rec.write_reload(f"{reld}/t4_{c}_reload_{r}.bin", {k: pert[f"{c}:{r}:{k}"] for k in order}, order)
rd = f"{P}/seeds/run_{arm}_{seed:02d}"
if not (os.path.isdir(rd) and len([f for f in os.listdir(rd) if f.startswith("t4_cor_record_")]) == 32):
    subprocess.run(["bash", str(__import__("pathlib").Path(__file__).resolve().with_name("si3_prather_multicycle_seed_run.sh")), rd, reld, CFL], check=False)   # stop mode ends in MPI_ABORT
outs, c3bad = {}, []
for c in CH:
    got = L.load(rd, c)
    for r in base[c]:
        h2, mb2, inp, order2, out = got[r]
        assert h2["reloaded"] == 1 and order2 == base[c][r][3]
        for k in order2:
            a, b, p = base[c][r][2][k], inp[k], pert[f"{c}:{r}:{k}"]
            ok = (b.tobytes() == p.tobytes() and np.all((b == a) | (b == np.nextafter(a, np.inf)) | (b == np.nextafter(a, -np.inf)))
                  and np.all(b[a == 0] == 0))
            if not ok:
                c3bad.append(f"{c}:{r}:{k}")
        for k, v in out.items():
            outs[f"{c}:{r}:{k}"] = v
np.savez_compressed(f"{P}/seeds/out_{arm}_{seed:02d}.npz", **outs, c3_bad=np.array(c3bad, dtype="U60"))
print(arm, "seed", seed, "C3 (received exactly the perturbed values, zeros held):", not c3bad, c3bad[:5])
subprocess.run(["rm", "-rf", rd, reld])
