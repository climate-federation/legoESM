"""``gradient_y_cgrid`` must divide by the grid's OWN v-point metric.

WHY THIS FILE EXISTS (#1728).  The operator used to rebuild a v-face spacing
from ``grid.dy`` as ``0.5*(dy_T[j] + dy_T[j-1])`` -- the finite-difference
distance between two T-points -- while the C-grid geometry already carried a
``dy_v``.  Under ``metric_convention="nemo_isotropic"`` that stored ``dy_v`` is
NEMO's ``e2v`` (``ra*rad*COS(rad*gphiv)*rn_e1_deg``, usrdef_hgr.F90:118),
evaluated AT the v-face latitude, and on Mercator rows the two differ by up to
8.2e-03.  On the NEMO-DINO card that difference was the ENTIRE residual of the
first-step meridional pressure gradient: it fell from 5.9e-05 of NEMO's own rms
to 1.2e-11 when the operator read the carried metric.

Every test below goes RED if the operator returns to the reconstruction --
that is the revert control, and it is why the reconstruction is spelled out
here rather than imported.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.latlon import create_latlon_geometry, create_latlon_grid
from legoesm.grids.operators_latlon_cgrid import gradient_y_cgrid


@pytest.fixture(autouse=True)
def _fp64():
    """Rule 1c: a metric comparison at the default float32 measures its own
    rounding.  The "exact"-convention row below reads 1.5e-08 -- 0.13 x f32
    eps -- under the default policy and 2.2e-16 under fp64, which is the
    difference between "one ulp" and "a tenth of single precision".
    """
    before = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(before)


def _reconstruction(grid):
    """What the operator used to divide by: ``0.5*(dy_T[j] + dy_T[j-1])``."""
    dy_h = np.asarray(grid.dy) * 0.5
    dyp = np.pad(dy_h, (1, 1), mode="edge")
    return 0.5 * (dyp[1:] + dyp[:-1])


def _mercator_lat(n_lat=60, e1_deg=1.0):
    """NEMO usr_def_hgr's Mercator T latitudes, so dlat really does vary."""
    j = np.arange(1, n_lat + 1) - 0.5
    return np.arcsin(np.tanh(np.deg2rad(e1_deg) * (j - n_lat / 2)))


@pytest.fixture(scope="module")
def isotropic():
    return create_latlon_geometry(
        n_lat=60, n_lon=32, lat_1d=jnp.asarray(_mercator_lat()),
        metric_convention="nemo_isotropic")


def _field(grid):
    lat = np.asarray(grid.lat)[:, None]
    lon = np.linspace(0.0, 2 * np.pi, grid.n_lon, endpoint=False)[None, :]
    return jnp.asarray(np.sin(3 * lat) * np.cos(2 * lon) + 0.5 * lat)


def test_divides_by_the_carried_metric_exactly(isotropic):
    """``df_dy * dy_v`` must reproduce the raw difference BIT for BIT."""
    f = _field(isotropic)
    got = np.asarray(gradient_y_cgrid(f, isotropic))
    dy_v = np.asarray(isotropic.dy_v)
    fp = np.asarray(f)
    diff = np.empty_like(got)
    diff[1:-1] = fp[1:] - fp[:-1]
    diff[0] = 0.0                       # zero_polar_lat_ends
    diff[-1] = 0.0
    rebuilt = np.where(got == 0.0, 0.0, got * dy_v)
    interior = slice(1, -1)
    assert np.array_equal(rebuilt[interior], diff[interior]) or np.allclose(
        rebuilt[interior], diff[interior], rtol=1e-15, atol=0.0)


def test_the_divisor_is_dy_v_and_not_a_lookalike(isotropic):
    """Under the isotropic convention ``dx_v`` is BITWISE ``dy_v`` on the
    interior -- that is what "isotropic" means -- so a test whose fixture is
    only that geometry passes on an operator dividing by ``dx_v``.  An
    independent review made exactly that substitution and all the other rows
    stayed green.  This row uses a geometry where the candidates differ.
    """
    g = create_latlon_geometry(n_lat=60, n_lon=32,
                               lat_1d=jnp.asarray(_mercator_lat()))
    dy_v = np.asarray(g.dy_v)
    for name in ("dx_v", "dy_u", "dx_u"):
        other = np.asarray(getattr(g, name))
        if other.shape != dy_v.shape:
            continue                      # a different stagger cannot be it
        assert not np.array_equal(other, dy_v), (
            f"{name} is bitwise dy_v on this geometry too, so this row "
            "cannot discriminate them either")
    f = _field(g)
    got = np.asarray(gradient_y_cgrid(f, g))
    for name in ("dx_v",):
        other = np.asarray(getattr(g, name))
        if other.shape != dy_v.shape:
            continue
        wrong = np.asarray(gradient_y_cgrid(
            f, g._replace(dy_v=jnp.asarray(other))))
        moved = np.abs(got - wrong) / np.maximum(np.abs(got), 1e-300)
        assert np.nanmax(moved[1:-1]) > 1e-6, (
            f"dividing by {name} instead of dy_v changes nothing measurable, "
            "so the operator's divisor is not pinned by these tests")


def test_it_is_not_the_t_point_reconstruction(isotropic):
    """The revert control: the two divisors give measurably different answers
    on this geometry, so an operator that went back to the reconstruction
    cannot pass the test above."""
    dy_v = np.asarray(isotropic.dy_v)[:, 0]
    rec = _reconstruction(isotropic)
    rel = np.abs(rec / dy_v - 1.0)
    assert rel.max() > 1e-4, (
        "the two divisors agree on this geometry, so nothing here could "
        f"detect the defect (max relative gap {rel.max():.3e})")
    f = _field(isotropic)
    got = np.asarray(gradient_y_cgrid(f, isotropic))
    legacy = np.asarray(gradient_y_cgrid(
        f, isotropic._replace(
            dy_v=jnp.broadcast_to(jnp.asarray(rec)[:, None],
                                  isotropic.dy_v.shape))))
    # Measured 3.809e-05 on this geometry.  The larger divisor gaps (up to
    # 4.2e-03) sit on the two polar rows, which ``zero_polar_lat_ends`` zeroes,
    # so the interior is what a revert would actually move.
    moved = np.abs(got - legacy) / np.maximum(np.abs(got), 1e-300)
    assert np.nanmax(moved[1:-1]) > 1e-5, np.nanmax(moved[1:-1])


def test_uniform_dlat_is_bit_identical_to_the_reconstruction():
    """Every regular lat-lon card must be untouched by the change."""
    g = create_latlon_geometry(n_lat=32, n_lon=64)
    rec = _reconstruction(g)[:, None]
    assert int((rec != np.asarray(g.dy_v)).sum()) == 0
    f = _field(g)
    got = np.asarray(gradient_y_cgrid(f, g))
    legacy = np.asarray(gradient_y_cgrid(
        f, g._replace(dy_v=jnp.broadcast_to(jnp.asarray(_reconstruction(g))
                                            [:, None], g.dy_v.shape))))
    assert np.array_equal(got, legacy)


def test_variable_dlat_exact_convention_moves_by_at_most_one_ulp():
    """The other side of the blast radius, measured rather than asserted away."""
    g = create_latlon_geometry(n_lat=60, n_lon=32,
                               lat_1d=jnp.asarray(_mercator_lat()))
    rel = np.abs(_reconstruction(g)[:, None] / np.asarray(g.dy_v) - 1.0)
    assert rel.max() <= 4.0 * np.finfo(np.float64).eps, rel.max()


def test_a_grid_without_a_v_metric_still_works():
    """``LatLonGrid`` carries no ``dy_v`` and is still passed by the
    atmosphere lat-lon C-grid dycores, the operator adapter, nesting and the
    MPI band extension.  It must fall back, not raise."""
    g = create_latlon_grid(n_lat=16, n_lon=32)
    assert not hasattr(g, "dy_v")
    f = jnp.asarray(np.random.default_rng(0).normal(size=(16, 32)))
    out = np.asarray(gradient_y_cgrid(f, g))
    assert out.shape == (17, 32)
    assert np.isfinite(out).all()
    # and it is exactly the reconstruction, i.e. its behaviour is unchanged
    rec = _reconstruction(g)
    fp = np.asarray(f)
    want = np.zeros_like(out)
    want[1:-1] = (fp[1:] - fp[:-1]) / rec[1:-1, None]
    assert np.array_equal(out, want)


#: Every place in the shipped packages that SELECTS NEMO's isotropic metric
#: convention, i.e. every place whose meridional gradients moved when
#: ``gradient_y_cgrid`` started reading the carried ``dy_v``.  Uniform-dlat
#: cards are bit-identical and variable-dlat "exact" cards move by one ulp, so
#: this list IS the blast radius.  Adding a card here is fine; adding one
#: WITHOUT noticing is the thing this test exists to stop.
ISOTROPIC_SELECTORS = {
    # the two NEMO-DINO oracle recipes (dino.py's own comment at :128 scopes
    # the convention to nemo_dino_kamm / nemo_dino_kamm_mlf)
    "packages/ocean/legoesm/ocean/experiments/dino.py",
    # the NEMO bridge auto-detects it from a mesh whose e1t == e2t
    "packages/ocean/legoesm/ocean/fidelity/nemo_state_bridge.py",
    # two #1226 probes read it from LEGOESM_METRIC_CONVENTION
    "scripts/validate/ocean_fidelity/dino_1226/ldf_slp_per_element.py",
    "scripts/validate/ocean_fidelity/dino_1226/sshnxt_divhor_canonical.py",
}

#: The environment variable those probes read.  A file that consults it is a
#: selector even though the literal never appears in its source.
ISOTROPIC_ENV = "LEGOESM_METRIC_CONVENTION"


def test_the_blast_radius_is_the_cards_we_named():
    """No card may join the isotropic convention silently.

    The search is an AST walk over EVERY string constant with that value that
    is not part of a comparison -- not a grep and not a match on assignment
    forms.  An independent review fed the first version seven realistic
    idioms (a positional argument, a list literal, an ``or``-default, a
    ``setattr``, a ``dict.get`` default, a variable-built string) and it
    missed all seven; it also scanned only ``packages/`` and ``src/`` while a
    probe under ``scripts/`` selects the convention through an environment
    variable.  Prose mentions live INSIDE a docstring's own constant and
    comments are not in the tree at all, so neither can be mistaken for a
    selection, and ``==``/membership tests are excluded explicitly.
    """
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    files = []
    for sub in ("packages", "src", "scripts"):
        files += list((root / sub).rglob("*.py"))
    found = set()
    for path in files:
        text = path.read_text(errors="replace")
        rel = str(path.relative_to(root))
        if ISOTROPIC_ENV in text:
            try:
                tree = ast.parse(text)
            except SyntaxError:                            # pragma: no cover
                tree = None
            if tree is not None and any(
                    isinstance(n, ast.Constant) and n.value == ISOTROPIC_ENV
                    for n in ast.walk(tree)):
                found.add(rel)
        if "nemo_isotropic" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:                                # pragma: no cover
            continue
        compared = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for sub_node in ast.walk(node):
                    compared.add(id(sub_node))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant)
                    and node.value == "nemo_isotropic"
                    and id(node) not in compared):
                found.add(rel)
    unexpected = found - ISOTROPIC_SELECTORS
    assert not unexpected, (
        "these files now select metric_convention='nemo_isotropic' (or read "
        f"{ISOTROPIC_ENV}) and were not in the named blast radius: "
        f"{sorted(unexpected)}")
    missing = ISOTROPIC_SELECTORS - found
    assert not missing, (
        f"{sorted(missing)} no longer select it, so this test would not "
        "notice a new card either -- update the list deliberately")


def test_the_blast_radius_walk_sees_the_idioms_a_review_fed_it():
    """NON-VACUITY: the seven forms that got past the first version."""
    import ast
    src = """
import os
A = ["nemo_isotropic"]
B = ("exact", "nemo_isotropic")
def f(x):
    return g(x, "nemo_isotropic")
C = os.environ.get("X") or "nemo_isotropic"
def h(o):
    setattr(o, "metric_convention", "nemo_isotropic")
D = {}.get("k", "nemo_isotropic")
E = "nemo_" + "isotropic"
def cmp(m):
    return m == "nemo_isotropic"
"""
    tree = ast.parse(src)
    compared = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for sub_node in ast.walk(node):
                compared.add(id(sub_node))
    hits = [n for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and n.value == "nemo_isotropic"
            and id(n) not in compared]
    # six of the seven are literal constants outside a comparison; the
    # variable-BUILT string ("nemo_" + "isotropic") is NOT one and is named
    # here as the hole this walk still has.
    assert len(hits) == 6, [ast.dump(h) for h in hits]
    only_compare = [n for n in ast.walk(tree)
                    if isinstance(n, ast.Constant)
                    and n.value == "nemo_isotropic" and id(n) in compared]
    assert len(only_compare) == 1, "the == form must be excluded, and only it"
