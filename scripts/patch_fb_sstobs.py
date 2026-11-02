"""Add --sst-obs (observed SST, RYF 1990/91 months) to flux_budget_offline.py, plus the run's SST bias table."""
import shutil
import sys

p = sys.argv[1]
s = open(p).read()
if "--sst-obs" in s:
    sys.exit("already patched")
R = [
('''    ap.add_argument("--sst-run", default=None)''',
'''    ap.add_argument("--sst-obs", default=None)
    ap.add_argument("--sst-run", default=None)'''),
('''    def sst_at(doy):
        if snaps is None:
            return sstC''',
'''    obs_m = None
    if a.sst_obs:
        from scipy.interpolate import RegularGridInterpolator as _RGI2
        _o = xr.open_dataset(a.sst_obs)
        _v = _o["tos"] if "tos" in _o else _o[next(iter(_o.data_vars))]
        _t = _v["time"].dt
        _ms = [_v.sel(time=(_t.year == (1991 if m <= 4 else 1990)) & (_t.month == m)).squeeze("time", drop=True) for m in range(1, 13)]
        _la_n = next(d for d in _ms[0].dims if d.lower().startswith("lat"))
        _lo_n = next(d for d in _ms[0].dims if d.lower().startswith("lon"))
        _la = np.asarray(_ms[0][_la_n], float); _lo = np.mod(np.asarray(_ms[0][_lo_n], float), 360.0)
        _oi, _oj = np.argsort(_la), np.argsort(_lo)
        _la, _lo = _la[_oi], _lo[_oj]
        _lop = np.concatenate([[_lo[-1] - 360], _lo, [_lo[0] + 360]])
        _qq = np.stack([np.clip(LAd, _la.min(), _la.max()), np.mod(LOd, 360.0)], axis=-1)
        obs_m = []
        for mm in _ms:
            v = np.asarray(mm.transpose(_la_n, _lo_n).values, float)[_oi][:, _oj]
            if np.nanmedian(v) > 100:
                v = v - 273.15
            v = np.maximum(np.nan_to_num(v, nan=-1.8), -1.8)
            vp = np.concatenate([v[:, -1:], v, v[:, :1]], axis=1)
            obs_m.append(_RGI2((_la, _lop), vp, bounds_error=False, fill_value=-1.8)(_qq))
        obs_m = np.stack(obs_m)
        print(f"  observed SST from {a.sst_obs}: RYF 1990/91 months, range {obs_m.min():.1f}..{obs_m.max():.1f} C")

    def _run_at(doy):
        ds = np.array([d for d, _ in snaps])
        ext_d = np.concatenate([[ds[-1] - 365.0], ds, [ds[0] + 365.0]])
        ext_f = [snaps[-1][1]] + [f for _, f in snaps] + [snaps[0][1]]
        i = min(max(int(np.searchsorted(ext_d, doy % 365.0, side="right") - 1), 0), len(ext_d) - 2)
        w1 = (doy % 365.0 - ext_d[i]) / (ext_d[i + 1] - ext_d[i])
        return (1 - w1) * ext_f[i] + w1 * ext_f[i + 1]

    def sst_at(doy):
        if obs_m is not None:
            pos = (np.mod(doy, 365.0) + 1.0 - 15.5) / (365.0 / 12.0)
            m0 = int(np.floor(pos)); fr = pos - m0
            return np.where(ocean, (1 - fr) * obs_m[m0 % 12] + fr * obs_m[(m0 + 1) % 12], 10.0)
        if snaps is None:
            return sstC'''),
('''    names = ["sw_net",''', '''    names = ["sst_used", "sst_run_mean", "sw_net",'''),
('''        sstC = sst_at(day)
        sstK = jnp.asarray(sstC + 273.15)
''', '''        sstC = sst_at(day)
        sstK = jnp.asarray(sstC + 273.15)
        acc["sst_used"] += sstC
        if obs_m is not None and snaps is not None:
            acc["sst_run_mean"] += _run_at(day)
'''),
('''SST from {a.sst_run or a.restart}; + = into ocean''', '''SST from {a.sst_obs or a.sst_run or a.restart}; + = into ocean'''),
('''    qm = acc["q_net_ice"] if ice is not None else acc["q_net"]
    if ref is not None:''', '''    qm = acc["q_net_ice"] if ice is not None else acc["q_net"]
    if obs_m is not None and snaps is not None:
        print("\\n  annual-mean SST [C]   band         run  observed   run-obs")
        for lo_ in range(-80, 90, 10):
            mb = ocean & (LA >= lo_) & (LA < lo_ + 10)
            if mb.sum() > 20:
                print(f"                       {lo_:+4d}..{lo_ + 10:+4d} {gm(acc['sst_run_mean'], mb):8.2f} "
                      f"{gm(acc['sst_used'], mb):9.2f} {gm(acc['sst_run_mean'] - acc['sst_used'], mb):8.2f}")
    if ref is not None:'''),
]
for old, new in R:
    assert s.count(old) == 1, f"anchor not found exactly once: {old[:60]!r}"
    s = s.replace(old, new)
compile(s, p, "exec")
shutil.copy(p, p + ".bak_sstobs")
open(p, "w").write(s)
print(f"patched {p}: --sst-obs added; backup {p}.bak_sstobs")
