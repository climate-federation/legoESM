"""Conservative, differentiable MPAS<->MPAS remap (fine-quadrature first-order).

Jones-1999 conservation gates on two real Voronoi meshes at different
resolutions: constant-field exactness and weight-sum==1 (exact by construction),
global-integral conservation (first-order, convergent in n_sub), plus
differentiability of the apply and the coupler dispatch.  Login-trivial sizes but
builds meshes + JAX => run on a compute node (sbatch), not the login node.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


class TestMpasToMpasConservativeRemap(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from legoesm.grids.voronoi import create_voronoi_mesh
        # Two global SCVT meshes at different resolution (mesh quality is
        # irrelevant to conservation, so few Lloyd iterations keep it fast).
        cls.coarse = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)  # 162
        cls.fine = create_voronoi_mesh(subdivision_level=3, lloyd_iterations=5)    # 642

    def _weights(self, src, dst, n_sub=6):
        from legoesm.grids.conservative_regrid_unstructured import (
            compute_mpas_to_mpas_weights,
        )
        return compute_mpas_to_mpas_weights(src, dst, n_sub=n_sub)

    def test_gate1_constant_field_exact(self):
        from legoesm.grids.conservative_regrid import apply_conservative_regrid
        w = self._weights(self.fine, self.coarse)
        const = jnp.full((int(self.fine.nCells),), 3.5)
        out = apply_conservative_regrid(const, w)
        self.assertEqual(out.shape, (int(self.coarse.nCells),))
        self.assertTrue(jnp.allclose(out, 3.5, atol=1e-10))

    def test_gate3_weights_sum_to_one(self):
        w = self._weights(self.fine, self.coarse)
        dst = np.asarray(w.dst_idx_flat)
        wt = np.asarray(w.weights)
        sums = np.zeros(w.n_dst_cells)
        np.add.at(sums, dst, wt)
        # Global meshes => every target cell is fully covered => sum == 1 exactly.
        self.assertTrue(np.allclose(sums, 1.0, atol=1e-10))

    def test_gate4_global_integral_first_order(self):
        # Fine-quadrature is FIRST-ORDER conservative: overlap areas are
        # quadrature-approximated, so the global integral is conserved only to
        # the quadrature accuracy (NOT machine-exact — that needs the deferred
        # exact polygon-clip).  This is a STRESS config (642 -> 162, ~20 deg
        # cells, 4x coarsening); production fine meshes conserve far better, and
        # the companion test proves first-order convergence.  Constants +
        # partition-of-unity (gates 1, 3) ARE machine-exact regardless.
        from legoesm.grids.conservative_regrid import apply_conservative_regrid
        src, dst = self.fine, self.coarse
        w = self._weights(src, dst, n_sub=8)
        A_src = jnp.asarray(src.areaCell)
        A_dst = jnp.asarray(dst.areaCell)
        f_src = jnp.asarray(np.sin(np.asarray(src.latCell))
                            + 2.0 * np.cos(np.asarray(src.lonCell)))
        f_dst = apply_conservative_regrid(f_src, w)
        int_src = float(jnp.sum(f_src * A_src))
        int_dst = float(jnp.sum(f_dst * A_dst))
        # ``sin(lat) + 2 cos(lon)`` integrates to ~0 over the sphere (both terms
        # are antisymmetric / full-period), so the SIGNED ``int_src`` is a
        # near-cancellation residual (~0.1% of the field magnitude) and
        # ``|Δint| / |int_src|`` is a meaningless 0/0 ratio.  Normalise the
        # conservation error by the transported quantity's L1-weighted
        # magnitude — the physically correct, sign-robust denominator.
        scale = float(jnp.sum(jnp.abs(f_src) * A_src))
        rel = abs(int_dst - int_src) / scale
        self.assertLess(rel, 0.01)  # first-order conservative: |Δintegral| << |field|

    def test_gate4_conservation_converges_first_order(self):
        from legoesm.grids.conservative_regrid import apply_conservative_regrid
        src, dst = self.fine, self.coarse
        f_src = jnp.asarray(np.sin(2.0 * np.asarray(src.latCell)))
        A_src = jnp.asarray(src.areaCell)
        A_dst = jnp.asarray(dst.areaCell)
        int_src = float(jnp.sum(f_src * A_src))

        def rel_err(n_sub):
            w = self._weights(src, dst, n_sub=n_sub)
            f_dst = apply_conservative_regrid(f_src, w)
            return abs(float(jnp.sum(f_dst * A_dst)) - int_src) / abs(int_src)

        e_coarse = rel_err(3)
        e_fine = rel_err(12)
        # 4x more sub-cells => first-order => error should drop substantially.
        self.assertLess(e_fine, 0.6 * e_coarse)

    def test_differentiable(self):
        from legoesm.grids.conservative_regrid import apply_conservative_regrid
        w = self._weights(self.fine, self.coarse)
        f = jnp.asarray(np.random.RandomState(0).rand(int(self.fine.nCells)))

        def loss(x):
            return jnp.sum(apply_conservative_regrid(x, w) ** 2)

        g = jax.grad(loss)(f)
        self.assertEqual(g.shape, f.shape)
        self.assertTrue(jnp.all(jnp.isfinite(g)))
        self.assertFalse(jnp.allclose(g, 0.0))

    def test_dispatch_mpas_direct_no_latlon(self):
        from legoesm.coupler.grid_remap import make_grid_remapper
        gr = make_grid_remapper(self.fine, self.coarse)
        self.assertFalse(gr.identity)
        self.assertIsNotNone(gr.a2o)
        self.assertIsNotNone(gr.o2a)
        self.assertEqual(tuple(gr.a2o.src_shape), (int(self.fine.nCells),))
        self.assertEqual(tuple(gr.a2o.dst_shape), (int(self.coarse.nCells),))
        # Same mesh object -> identity short-circuit.
        self.assertTrue(make_grid_remapper(self.fine, self.fine).identity)

    def test_leading_axis_field_supported(self):
        """Rank-agnostic apply: an (nlev, nCells) field remaps per level."""
        from legoesm.grids.conservative_regrid import apply_conservative_regrid
        w = self._weights(self.fine, self.coarse)
        f = jnp.asarray(np.random.RandomState(1).rand(4, int(self.fine.nCells)))
        out = apply_conservative_regrid(f, w)
        self.assertEqual(out.shape, (4, int(self.coarse.nCells)))


if __name__ == "__main__":
    unittest.main()
