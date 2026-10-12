"""omip_strip_gap_onset.cohort_gap_interface: a planted low-K interface above an
active layer is found at the right index; a monotone column is not a gap."""
import numpy as np

from scripts.validate.ocean_fidelity import omip_strip_gap_onset as G


def test_cohort_gap_interface_finds_planted_gap_and_rejects_monotone():
    zw = np.array([1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 16.0, 20.0, 30.0, 50.0])
    K = np.array([
        [1e-2, 1e-2, 1e-2, 1e-5, 3e-5, 1e-2, 1e-2, 1e-3, 1e-6, 1e-6],   # gap, minimum at index 3
        [1e-2, 1e-2, 1e-2, 1e-2, 1e-3, 1e-5, 1e-6, 1e-6, 1e-6, 1e-6],   # monotone: no gap
    ])
    g, kg = G.cohort_gap_interface(K, zw, 1e-4, 40.0)
    assert g.tolist() == [True, False]
    assert kg[0] == 3
    K2 = K.copy(); K2[0, 4] = 1e-6                                       # move the minimum
    assert G.cohort_gap_interface(K2, zw, 1e-4, 40.0)[1][0] == 4


def test_face_nb_equal_for_uniform_avm_and_face_larger_next_to_turbulent_neighbour():
    rng = np.random.default_rng(1)
    U = rng.normal(size=(5, 5, 3)); V = rng.normal(size=(5, 5, 3))
    dz = np.array([1.0, 2.0])
    A = np.full((5, 5, 2), 1e-3)
    f, t = G.face_nb(U, U, V, V, A, dz, 2, 2, 0)        # now == before -> squared shear
    assert np.isclose(f, t) and f > 0
    A2 = A.copy(); A2[2, 3, 0] = 1e-1                    # turbulent east neighbour
    f2, t2 = G.face_nb(U, U, V, V, A2, dz, 2, 2, 0)
    assert np.isclose(t2, t) and f2 > 2 * f
    Ub = U.copy(); Ub[..., 1] = U[..., 0] + (U[..., 0] - U[..., 1])   # reversed before-shear
    assert G.face_nb(U, Ub, V, V, A, dz, 2, 2, 0)[0] != f


def test_west_to_east_matches_production_centring():
    # legoESM: u[j, i] = west face of cell i; production centre = 0.5*(u[i] + u[i+1])
    uw = np.array([[0.0, 1.0, 4.0, 9.0, 16.0]])
    ue = G.west_to_east(uw, axis=1)
    assert np.allclose(ue[0, :-1], uw[0, 1:]) and np.isnan(ue[0, -1])
    centre_probe = 0.5 * (ue + np.roll(ue, 1, axis=1))
    centre_prod = 0.5 * (uw[:, :-1] + uw[:, 1:])
    assert np.allclose(centre_probe[0, 1:4], centre_prod[0, 1:4])
    centre_raw = 0.5 * (uw + np.roll(uw, 1, axis=1))          # the misaligned form codex caught
    assert not np.allclose(centre_raw[0, 1:4], centre_prod[0, 1:4])


def test_onset_steps_requires_agreement_at_reference():
    ratio = np.array([[1.0, 0.8, 1.0],      # reference record
                      [1.0, 0.8, 0.95],
                      [0.85, 0.8, 0.95],
                      [0.5, 0.95, 0.85]])
    has, first = G.onset_steps(ratio)
    assert has.tolist() == [True, False, True]
    assert first[0] == 1 and first[2] == 2


def test_ratio_gate_coverage_and_iqr():
    assert G.ratio_gate(np.ones(100))[0]
    bad = np.concatenate([-np.ones(90), np.ones(10)])          # wrong sign mostly: coverage fails
    assert not G.ratio_gate(bad)[0]
    assert not G.ratio_gate(np.linspace(0.5, 1.5, 100))[0]      # IQR too wide
