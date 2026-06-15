"""Smoke test for the Silvestri comparison plotter: both cases run on synthetic
npz inputs and write the expected PNG files."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_PLOTTER = (Path(__file__).resolve().parents[3]
            / "scripts" / "plot" / "plot_silvestri_comparison.py")


def _load():
    spec = importlib.util.spec_from_file_location("_silv_plotter", _PLOTTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plot_turb2d(tmp_path):
    indir = tmp_path / "in"; indir.mkdir()
    t = np.linspace(0, 6, 50)
    for sch in ("W9V", "Leith2"):
        np.savez_compressed(
            indir / f"turb2d_{sch}_N64.npz", scheme=sch, N=64, Re=3.3e4,
            t=t, ke=np.exp(-0.1 * t), enstrophy=np.exp(-0.5 * t),
            k_energy=np.arange(1, 33.0), P_energy=np.arange(1, 33.0) ** -3.0,
            k_enstrophy=np.arange(1, 33.0), P_enstrophy=np.arange(1, 33.0) ** -1.0,
            zeta_36=np.random.default_rng(0).standard_normal((64, 64)),
            zeta_final=np.zeros((64, 64)))
    plt = _load()
    outdir = tmp_path / "out"
    plt.plot_turb2d(str(indir), str(outdir))
    for fig in ("fig3_turb2d_vorticity", "fig4_turb2d_timeseries",
                "fig5_turb2d_spectra"):
        assert (outdir / f"{fig}.png").exists(), fig


def test_plot_jet(tmp_path):
    indir = tmp_path / "in"; indir.mkdir()
    t = np.linspace(0, 1000 * 86400, 40)
    for sch in ("W9V", "SM2"):
        np.savez_compressed(
            indir / f"silvestri_jet_{sch}_80x64.npz", scheme=sch, blew=False,
            t=t, tke=np.ones_like(t), eke=0.1 * np.ones_like(t),
            ape=0.05 * np.ones_like(t),
            k_energy=np.arange(1, 33.0), P_energy=np.arange(1, 33.0) ** -3.0,
            k_enstrophy=np.arange(1, 33.0), P_enstrophy=np.arange(1, 33.0) ** -1.0,
            zonal_mean_buoyancy=np.random.default_rng(0).standard_normal((20, 50)),
            zeta_surface=np.random.default_rng(1).standard_normal((20, 64)))
    plt = _load()
    outdir = tmp_path / "out"
    plt.plot_jet(str(indir), str(outdir))
    for fig in ("fig8_jet_energy", "fig9_jet_spectra",
                "fig10_jet_buoyancy_vorticity"):
        assert (outdir / f"{fig}.png").exists(), fig
