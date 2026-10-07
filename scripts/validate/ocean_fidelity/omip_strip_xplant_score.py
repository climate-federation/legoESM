"""Score --probe-k-transplant arms against NEMO RUN_DT150_1TS every-step strip output (2S-2N, 225.5-255.5E).

Per run (a --trd-columns dump): (1) ALIGNMENT -- max relative |applied K - NEMO avt| and |A - avm| over the
transplanted interior columns and every step (must sit at the float32 storage floor for an arm that transplants
that coefficient); (2) at hours 4/9/12/24/36/48: ratio of column-median proxy N2 ours/NEMO at 5.8/12.8 m,
staircase max-jump share (1.6-14 m) ours and NEMO, top-5-level mean T bias ours-NEMO. Interior columns only:
BUF strip cells dropped at every strip edge (transplanted columns border untransplanted ones there).
(3) EVD firing, face by step on the top interfaces: NEMO = EVD increment (avt - avt_k) - (avm - avm_k) > 50
(nn_evdm=0, so avm - avm_k is exactly NEMO's IWM); ours = applied K > 50; hourly occupancy and the overlap
(both / NEMO only / ours only, as fractions of the faces either side fires).
Usage: omip_strip_xplant_score.py RUN_DIR [RUN_DIR ...]   (env BUF, default 3).
       omip_strip_xplant_score.py --gate EN_W_FILE   (ORCA1EN rerun: reproduces 1TS avt; avm_k = max(0.1*en/dissl, avmb))."""
import os
import sys

import numpy as np

NEMO = ('/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_DT150_1TS/'
        'ORCA1_1ts_20000101_20000102_eqs1ts_')
MESH = '/burg-archive/glab/users/pg2328/_omip_merge/data/grids/eORCA1.2_mesh_mask.nc'
J0, I0, NJ, NI = 180, 153, 13, 31            # ours (j, i) = strip (j - J0, i - I0)
HOURS = (4, 9, 12, 24, 36, 48)
KI = (4, 8)                                    # interface below level k = NEMO w-level k + 1 (5.8 / 12.8 m)


def interior(sj, si, buf):
    return (sj >= buf) & (sj < NJ - buf) & (si >= buf) & (si < NI - buf)


def n2(T, S, k, zt):                           # linear-EOS proxy, same on both sides
    return 9.81 * (3e-4 * (T[..., k] - T[..., k + 1]) - 7.5e-4 * (S[..., k] - S[..., k + 1])) / (zt[k + 1] - zt[k])


def share(Tc):                                 # median max-jump share of the positive T drop, levels 1..11
    d = np.clip(Tc[:, 1:11] - Tc[:, 2:12], 0, None)
    tot = d.sum(1)
    g = tot > 0.05
    return float(np.median((d.max(1) / np.maximum(tot, 1e-9))[g])) if g.any() else np.nan


def fire_overlap(nemo_fired, ours_fired):
    """(both, nemo_only, ours_only) as fractions of the union of fired faces."""
    u = np.sum(nemo_fired | ours_fired)
    if u == 0:
        return 0.0, 0.0, 0.0
    return (float(np.sum(nemo_fired & ours_fired)) / u, float(np.sum(nemo_fired & ~ours_fired)) / u,
            float(np.sum(~nemo_fired & ours_fired)) / u)


def en_closure_residual(avm_k, en, dissl, avmb, rn_ediff=0.1):
    """max |avm_k - max(rn_ediff*en/dissl, avmb)| / avm_k over faces with finite, positive dissl
    (nn_mxl=2: l = sqrt(en)/dissl, avm_k = max(rn_ediff*l*sqrt(en), avmb))."""
    ok = np.isfinite(avm_k) & np.isfinite(en) & np.isfinite(dissl) & (dissl > 0) & (avm_k > 0)
    pred = np.maximum(rn_ediff * en[ok] / dissl[ok], avmb)
    return float(np.max(np.abs(avm_k[ok] - pred) / avm_k[ok]))


def gate(en_path, ref_path=NEMO + 'W.nc', avmb=1.2e-4):
    import netCDF4 as nc
    f = lambda v: np.where(np.abs(v) < 1e15, v, np.nan)
    E, R = nc.Dataset(en_path), nc.Dataset(ref_path)
    g = lambda d, v: f(np.asarray(d[v][:, 1:30], dtype=np.float64))
    a_en, a_ref = g(E, 'avt'), g(R, 'avt')
    rep = rel_err(a_en, a_ref)
    avm_k, en, dl = g(E, 'avm_k'), g(E, 'en_k'), g(E, 'dissl_k')
    res = en_closure_residual(avm_k, en, dl, avmb)
    planted = en_closure_residual(avm_k[1:], en[:-1], dl[:-1], avmb)
    print(f'gate: EN vs 1TS avt max rel {rep:.2e} | avm_k closure residual {res:.2e} | '
          f'planted 1-step shift {planted:.2e} | steps {a_en.shape[0]} | dissl<=0 faces {int(np.sum(dl <= 0))}')
    return rep, res, planted


def rel_err(ours, oracle):
    ok = np.isfinite(oracle) & (np.abs(oracle) > 0)
    return float(np.max(np.abs(ours[ok] - oracle[ok]) / np.abs(oracle[ok])))


def main(runs, buf):
    import netCDF4 as nc
    m = nc.Dataset(MESH)
    zt = np.asarray(m['gdept_1d'][0])
    f = lambda v: np.ma.filled(v, np.nan).astype(np.float64)
    T, W = nc.Dataset(NEMO + 'T.nc'), nc.Dataset(NEMO + 'W.nc')
    for R in runs:
        cols = {}
        for fn in ('snapshot_day0001.npz', 'snapshot_final.npz'):
            z = np.load(os.path.join(R, fn))
            for k in ('col_T', 'col_S', 'col_K', 'col_A'):
                cols.setdefault(k, []).append(z[k])
            cj, ci = z['col_j'], z['col_i']
            assert z['col_T'].shape[0] == int(z['ttrd_zdf_n_steps']), 'row count != steps'
        O = {k: np.concatenate(v).astype(np.float64) for k, v in cols.items()}
        sj, si = cj - J0, ci - I0
        ok = interior(sj, si, buf)
        ns = min(O['col_T'].shape[0], T['votemper'].shape[0])
        nemo = lambda ds, v, k: f(ds[v][:ns, :k])[:, :, sj[ok], si[ok]].transpose(0, 2, 1)
        NT, NS = nemo(T, 'votemper', 30), nemo(T, 'vosaline', 30)
        OT, OS = O['col_T'][:ns, ok], O['col_S'][:ns, ok]
        wet = np.all(np.isfinite(NT[0, :, :13]), -1) & np.all(OT[0, :, :13] != 0, -1)
        nk = O['col_K'].shape[-1]
        aK = rel_err(O['col_K'][:ns, ok][:, wet], nemo(W, 'avt', nk + 1)[:, wet, 1:])
        aA = rel_err(O['col_A'][:ns, ok][:, wet], nemo(W, 'avm', nk + 1)[:, wet, 1:])
        print(f'{os.path.basename(R.rstrip("/"))}: steps {ns} | interior wet cols {wet.sum()} (BUF {buf}) | '
              f'align max rel |K-avt| {aK:.1e} |A-avm| {aA:.1e}')
        NW = {v: nemo(W, v, nk + 1)[:, wet, 1:] for v in ('avt', 'avt_k', 'avm', 'avm_k')}
        nf = ((NW['avt'] - NW['avt_k']) - (NW['avm'] - NW['avm_k'])) > 50.0
        of = O['col_K'][:ns, ok][:, wet] > 50.0
        b, n_only, o_only = fire_overlap(nf, of)
        print(f'  EVD faces (top {nk} interfaces): NEMO {nf.mean():.4f} ours {of.mean():.4f} | of union: both {b:.2f} '
              f'NEMO-only {n_only:.2f} ours-only {o_only:.2f} | max applied K {np.nanmax(O["col_K"][:ns, ok][:, wet]):.3g}')
        print('  hr | N2 O/N 5.8 12.8 | share O N | dT top5 | EVD frac NEMO ours')
        for h in HOURS:
            st = h * 24 - 1
            if st >= ns:
                break
            r = [np.median(n2(OT[st], OS[st], k, zt)[wet]) / np.median(n2(NT[st], NS[st], k, zt)[wet]) for k in KI]
            dT = float(np.mean((OT[st, :, :5] - NT[st, :, :5])[wet]))
            hs = slice(st - 23, st + 1)
            print(f'  h{h:02d} | {r[0]:5.2f} {r[1]:5.2f} | {share(OT[st][wet]):.2f} {share(NT[st][wet]):.2f} | {dT:+.3f}'
                  f' | {nf[hs].mean():.4f} {of[hs].mean():.4f}')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--gate']:
        gate(sys.argv[2])
    else:
        main(sys.argv[1:], int(os.environ.get('BUF', 3)))
