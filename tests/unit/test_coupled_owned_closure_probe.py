"""Budget arithmetic of the owned-mode closure probe on a synthetic,
hand-closed budget: every row 0, and a known leak shows up in its row."""
import importlib.util
from pathlib import Path

import numpy as np

_P = (Path(__file__).resolve().parents[2] / "scripts" / "validate"
      / "coupled_owned_closure.py")
_spec = importlib.util.spec_from_file_location("coupled_owned_closure", _P)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def _closed():
    # atm: E 10 (land 4, ocean 5, ice 1 received), P 7 -> dW_atm +3
    # tiles: P_land 2, P_ow 3, P_ice 1.5, P_lake 0.5; runoff 1
    # land: dW = 2 - 4 - 1 = -3
    # ice/lake: debited evap 1 received + 0.25 pending, fw out 0.5 applied
    #      + 0.2 pending
    # ocean in: 3 + 1.5 - 5 + 1 + 0.5 = 1.0 (lake precip 0.5 -> lake -> fw)
    f = dict(P_atm=7.0, E_atm=10.0, P_ow=3.0, P_ice=1.5, P_lake=0.5,
             P_land=2.0, E_ow=5.0, R_land=1.0, H_recv=1.0, H_debit=1.25,
             fw_applied=0.5, fw_out=0.7, q_ow=40.0, q_flux=2.0,
             q_freeze=1.0, ocean_heat_extr=0.0, pend_H=0.25, pend_fw=0.2,
             pend_land=0.0)
    s0 = dict(W_atm=100.0, W_land=50.0, W_ice=20.0, H_slab=1000.0)
    # ice/lake store: + P_lake 0.5 - fw_out 0.7 - debited evap 1.25
    s1 = dict(W_atm=103.0, W_land=47.0, W_ice=20.0 - 1.45, H_slab=1043.0)
    return s0, s1, f


def test_hand_closed_budget_gives_zero_rows():
    s0, s1, f = _closed()
    rows = probe.budget_rows(s0, s1, f)
    for r in ("precip_split", "land_water", "slab_heat", "atm_water",
              "ice_water", "system_water"):
        assert abs(rows[r]) < 1e-12, (r, rows[r])


def test_known_leak_lands_in_its_row():
    s0, s1, f = _closed()
    s1 = dict(s1, W_land=s1["W_land"] + 0.3)
    rows = probe.budget_rows(s0, s1, f)
    assert np.isclose(rows["land_water"], 0.3)
    assert np.isclose(rows["system_water"], 0.3)
    assert abs(rows["atm_water"]) < 1e-12 and abs(rows["ice_water"]) < 1e-12


def test_land_in_flight_water_counts_as_land():
    # 0.4 of rain the atmosphere gave is still in the land's sample window
    s0, s1, f = _closed()
    s1 = dict(s1, W_land=s1["W_land"] - 0.4)
    f = dict(f, pend_land=0.4)
    rows = probe.budget_rows(s0, s1, f)
    assert abs(rows["land_water"]) < 1e-12
    assert abs(rows["system_water"]) < 1e-12


def test_land_in_flight_by_cadence_phase():
    fl, area = np.array([0.5, 1.0]), np.array([2.0, 3.0])
    rain = np.array([1e-3, 2e-3])     # summed rates of the n samples
    ev = np.array([1e-5, 4e-5])
    dt, K = 300.0, 3
    held = (0.5 * 1e-5 * 2.0 + 1.0 * 4e-5 * 3.0) * dt
    rained = (0.5 * 1e-3 * 2.0 + 1.0 * 2e-3 * 3.0) * dt
    # right after a land call: nothing unconsumed, K steps of held still due
    assert np.isclose(probe.land_in_flight(fl, area, 0.0, ev, 0, K, dt),
                      K * held)
    # two steps later: two samples waiting, one held step left
    assert np.isclose(probe.land_in_flight(fl, area, rain, ev, 2, K, dt),
                      rained + held)
    # land every step (K = 1) at its call: only the one held step
    assert np.isclose(probe.land_in_flight(fl, area, 0.0, ev, 0, 1, dt), held)


def test_stores_weights():
    k = dict(area=np.array([2.0, 3.0]), f_land=np.array([0.5, 0.0]),
             f_water=np.array([0.5, 1.0]), dz=np.array([0.1, 0.2]),
             C_mix=10.0, g=10.0, rho_water=1000.0, rho_ice=900.0)
    a = dict(q_tot=np.array([[0.01], [0.02]]), dp=np.array([[100.0], [50.0]]),
             theta=np.array([[0.3, 0.2], [0.0, 0.0]]),
             snow=np.array([1.0, 0.0]), pond=np.array([0.001, 0.0]),
             canopy=np.array([0.5, 0.0]), h_ice=np.array([1.0, 2.0]),
             conc=np.array([0.5, 0.25]), T_slab=np.array([270.0, 280.0]))
    s = probe.stores(a, k)
    assert np.isclose(s["W_atm"], (0.01 * 100 / 10) * 2 + (0.02 * 50 / 10) * 3)
    soil0 = (0.3 * 0.1 + 0.2 * 0.2) * 1000.0
    assert np.isclose(s["W_land"], 0.5 * (soil0 + 1.0 + 1.0 + 0.5) * 2.0)
    assert np.isclose(s["W_ice"], 900 * (1 * 0.5 * 0.5 * 2 + 2 * 0.25 * 1 * 3))
    assert np.isclose(s["H_slab"], 10 * (270 * 0.5 * 2 + 280 * 1.0 * 3))
