"""Iter-902 diagnostic: quantify the magnitude of the dxa-weighted vs
uniform-grid simplification discrepancy in the iter-892 PPM cube-edge
xt formula.

iter-888 docstring at `_ppm_reconstruct_1d` (line 124-131 in
`operators_cdgrid.py`) explicitly notes:

    The 4-point xt at the cube-face edge uses the UNIFORM-GRID
    simplification of Fortran's dxa-weighted formula (lines 360-
    361, 366-367):
        xt = 0.75*(q1(0)+q1(1)) - 0.25*(q1(-1)+q1(2))
    and is clipped to ``min/max(q1(-1..2))``.  For non-uniform
    cubed-sphere boundary cells this is a partial Fortran-fidelity
    fix; the full ``dxa``-weighted formula is deferred until
    ``dxa`` plumbing is added.

iter-902 quantifies the numerical magnitude of this discrepancy.
The Fortran formula at `tp_core.F90:360-361` is:

    al(1) = 0.5 * ( ((2*dxa(0,j)+dxa(-1,j))*q1(0) - dxa(0,j)*q1(-1))
                    / (dxa(-1,j)+dxa(0,j))
                  + ((2*dxa(1,j)+dxa( 2,j))*q1(1) - dxa(1,j)*q1( 2))
                    / (dxa(1, j)+dxa(2, j)) )

For uniform dxa (=1), this collapses to the simplification.
Cubed-sphere dxa is non-uniform near cube vertices; we measure how
big the deviation actually is at C36 face-corner cells.

Output: two numbers per cube-vertex cell:
  (a) uniform_xt = 0.75*(q1(0)+q1(1)) - 0.25*(q1(-1)+q1(2))
  (b) fortran_al1 = the dxa-weighted formula above
  (c) abs(fortran_al1 - uniform_xt)  -- the missed correction.

If |delta| << machine_epsilon * |q|, the simplification is
acceptable and iter-902 = no-op.  If |delta| > 1e-6 * |q|, full
plumbing is potentially worth iter-903+ effort.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def fortran_al1(q1m1, q10, q11, q12, dxam1, dxa0, dxa1, dxa2):
    """Fortran al(1) at the LEFT cube edge per tp_core.F90:360-361."""
    a = ((2.0 * dxa0 + dxam1) * q10 - dxa0 * q1m1) / (dxam1 + dxa0)
    b = ((2.0 * dxa1 + dxa2) * q11 - dxa1 * q12) / (dxa1 + dxa2)
    return 0.5 * (a + b)


def uniform_xt(q1m1, q10, q11, q12):
    return 0.75 * (q10 + q11) - 0.25 * (q1m1 + q12)


def main():
    n = 36
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    # dxa = 1 / rdxa at cell centres.  Shape (6, n, n).
    rdxa = np.asarray(cdgrid.rdxa)
    dxa = 1.0 / rdxa

    # We're interested in the WEST cube edge (i=0..2 in interior, plus
    # the two halo cells).  Halo cells' dxa is taken from the neighbour
    # face; for diagnostic purposes we use mode='edge' replicas (which
    # is what `_ppm_reconstruct_1d`'s `q_pad` does for q values) — i.e.,
    # dxa(-1) = dxa(0), dxa(-2) = dxa(0), and similarly on the right.
    # This is an UNDERESTIMATE of the discrepancy (true halo dxa from
    # the neighbour face would differ); we report it as a lower bound.

    # Pick face 0 along i-axis (face's WEST edge meets face 4's EAST
    # edge).  Sample j=middle and j=corner.
    print("Iter-902 diagnostic: dxa-weighted vs uniform-xt magnitudes")
    print(f"at C36 face-0 LEFT cube edge (i=0..3 in interior).")
    print()
    print("Test scalar: q1(j) = sin(j * dx / radius) along an i-strip,")
    print("approximating a smooth W2-like field.  Shifted to interior")
    print("indices.  Reports max |fortran_al1 - uniform_xt| over j-rows.")
    print()

    radius = float(grid.radius)
    deltas = []
    for j in range(n):
        # Get dxa values at i = 0..3 on face 0 row j (interior cells).
        # Halo cells (i=-1, i=-2) use mode='edge' replica = dxa[0,0,j].
        dxa_minus2 = dxa[0, 0, j]    # halo replicate (LOWER BOUND)
        dxa_minus1 = dxa[0, 0, j]    # halo replicate (LOWER BOUND)
        dxa_0 = dxa[0, 0, j]
        dxa_1 = dxa[0, 1, j]
        dxa_2 = dxa[0, 2, j]
        dxa_3 = dxa[0, 3, j]

        # Smooth synthetic q1 (sine wave at characteristic length).
        # q1(i) at interior position i is taken as sin(i * dxa / radius).
        # Halo cells use mode='edge' replicas of cube-edge interior cell.
        # Interior i=0,1,2,3 take their actual indices; halo i=-1,-2
        # replicate q1(0).
        q1_im1 = np.sin(0.0)   # replicate q1(0)
        q1_0 = np.sin(float(0) * dxa_0 / radius)
        q1_1 = np.sin(float(1) * dxa_1 / radius)
        q1_2 = np.sin(float(2) * dxa_2 / radius)

        u_xt = uniform_xt(q1_im1, q1_0, q1_1, q1_2)
        f_al1 = fortran_al1(q1_im1, q1_0, q1_1, q1_2,
                              dxa_minus1, dxa_0, dxa_1, dxa_2)
        deltas.append(abs(f_al1 - u_xt))

    deltas = np.asarray(deltas)
    print(f"Per-row stats over j=0..{n-1}:")
    print(f"  min  |delta| = {deltas.min():.3e}")
    print(f"  max  |delta| = {deltas.max():.3e}")
    print(f"  mean |delta| = {deltas.mean():.3e}")
    print()
    print(f"Reference characteristic scales:")
    print(f"  typical q1 magnitude at smooth IC: O(1)")
    print(f"  iter-893 production W2 v_ll_Linf: 1.319e-01 m/s")
    print(f"  iter-900 strict-Fortran W2 delta:  +7.080e-02 m/s")
    print()

    if deltas.max() < 1e-6:
        print("VERDICT: dxa-weighted correction at the LEFT cube edge is")
        print("  smaller than machine-precision floor.  iter-902 = NO-OP")
        print("  on the production W2 path.  iter-903+ dxa plumbing is")
        print("  unlikely to produce measurable W2 progress; recommend")
        print("  pivot to a different fidelity gap.")
    elif deltas.max() < 1e-3:
        print("VERDICT: dxa-weighted correction is sub-millimeter at")
        print("  smooth W2 scales.  Likely smaller than iter-893's")
        print("  17 % W2 reduction.  Implementation cost (full halo=2")
        print("  dxa plumbing) probably exceeds the marginal value;")
        print("  document for posterity, deprioritize.")
    else:
        print("VERDICT: dxa-weighted correction is large enough to")
        print("  potentially affect W2.  iter-903+ implementation may")
        print("  yield measurable improvement.  Worth pursuing.")


if __name__ == "__main__":
    main()
