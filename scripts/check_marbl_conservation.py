"""Tracer inventories between two run_omip restart files (conservation check).
Usage: python check_marbl_conservation.py RESTART_A.npz RESTART_B.npz GRID.npz"""
import sys
import numpy as np

a_path, b_path, g_path = sys.argv[1:4]
A, B, G = np.load(a_path), np.load(b_path), np.load(g_path)
print("restart keys:", ", ".join(f"{k}{list(A[k].shape)}" for k in A.files if A[k].ndim >= 2)[:900])

lat, lon, mask = G["lat"], G["lon"], G["mask"] > 0.5
dz = np.asarray(G["dz"], float)
dlat, dlon = np.abs(np.gradient(lat)), np.abs(np.gradient(lon))
area = 6.371e6 ** 2 * (np.sin(np.minimum(lat + dlat / 2, np.pi / 2))
                       - np.sin(np.maximum(lat - dlat / 2, -np.pi / 2)))[:, None] * dlon[None, :]
vol = area[..., None] * dz[None, None, :] * mask[..., None]


def find(npz, name):
    for k in npz.files:
        kk = k.split("/")[-1].split(".")[-1]
        for pre in ("", "biogeo_", "bgc_", "biogeo.", "marbl_"):
            if kk == pre + name or k == pre + name:
                return k
    return None


def inv(npz, terms):
    tot, used = 0.0, []
    for name, fac in terms:
        k = find(npz, name)
        if k is None:
            continue
        a = np.asarray(npz[k], float)
        if a.shape != vol.shape:
            continue
        tot += fac * np.nansum(a * vol)
        used.append(name)
    return tot, used


BUDGETS = {
    "salt": [("S", 1.0)],
    "total P": [("PO4", 1), ("DOP", 1), ("DOPr", 1), ("spP", 1), ("diatP", 1), ("diazP", 1), ("zooC", 1 / 117.0)],
    "total N": [("NO3", 1), ("NH4", 1), ("DON", 1), ("DONr", 1), ("spC", 16 / 117.0), ("diatC", 16 / 117.0),
                ("diazC", 16 / 117.0), ("zooC", 16 / 117.0)],
    "PO4 only": [("PO4", 1)],
    "DIC": [("DIC", 1)],
    "ALK": [("ALK", 1)],
    "SiO3": [("SiO3", 1), ("diatSi", 1)],
}
for name, terms in BUDGETS.items():
    ia, used = inv(A, terms)
    ib, _ = inv(B, terms)
    if not used:
        print(f"{name:9s}: tracers not found in restart")
        continue
    print(f"{name:9s}: {ia:.6e} -> {ib:.6e}   relative change {(ib - ia) / ia:+.2e}   ({'+'.join(used)})")

for key in ("eta", "ssh", "SSH"):
    k = find(A, key)
    if k is not None:
        ea, eb = np.asarray(A[k], float), np.asarray(B[k], float)
        dv = np.nansum((eb - ea) * area * mask)
        print(f"free-surface volume change {dv:+.3e} m3 = {dv / np.nansum(vol):+.2e} of ocean volume")
        break
