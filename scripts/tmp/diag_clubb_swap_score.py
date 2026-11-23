import glob, sys, numpy as np
L = "/burg-archive/glab/users/pg2328/legoESM/results/les_ref/bomex/profiles/prof_*.npz"
acc = {}
for f in sorted(glob.glob(L)):
    d = np.load(f); th = float(d["t_hours"])
    if 4.0 - 1e-6 <= th <= 6.0 + 1e-6:
        for k in ("z", "wqt", "wtheta"):
            acc.setdefault(k, []).append(np.asarray(d[k], float))
les = {k: np.mean(v, 0) for k, v in acc.items()}
print("LES frames", len(acc["z"]), "wqt sfc-ish", les["wqt"][:2], "wtheta", les["wtheta"][:2])
F = np.load("F.npz"); zm = F["zm"].ravel()
arms = {"F": F}
try: arms["J"] = np.load("J.npz")
except Exception: pass
NS = 72; w = range(NS * 4 // 6 + 1, NS + 1)
for a, d in arms.items():
    for k, lk in (("wprtp", "wqt"), ("wpthlp", "wtheta")):
        m = np.mean([d[f"{a}{i}/{k}"].ravel() for i in w], 0)
        for z in (300, 500, 700, 900, 1100, 1300, 1500):
            r = np.interp(z, zm, m) / np.interp(z, les["z"], les[lk])
            print(a, k, z, f"{np.interp(z, zm, m):.3e}", f"ratio {r:.2f}")
if "J" in arms:
    for k in ("wprtp", "wpthlp", "rtm", "thlm", "rcm"):
        for i in (1, 2, 12, 72):
            f, j = F[f"F{i}/{k}"], arms["J"][f"J{i}/{k}"]
            print("F-J", k, i, np.max(np.abs(f - j)) / max(np.max(np.abs(f)), 1e-300))
