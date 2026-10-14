"""Generator for clubb_lscale_grad_golden.npz (run from the repo root).

Records compute_mixing_length values + gradients from the SEQUENTIAL
implementation: put clubb.py as of 91962c34d in place first
(git show 91962c34d:packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py
> that file), run this, restore; the script refuses any other clubb.py.
That both smoothing branches change the answer is checked by deleting
each one in clubb.py (the test then fails), not here: comparing adjacent
levels for equality miscounts boundary cells."""
import subprocess as _sp

_CLUBB = "packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py"
_PIN = "91962c34d"


def _git(*a):
    return _sp.run(["git", "-c", "safe.directory=*", *a], capture_output=True,
                   text=True, check=True).stdout.strip()


if _git("hash-object", _CLUBB) != _git("rev-parse", f"{_PIN}:{_CLUBB}"):
    raise SystemExit(f"{_CLUBB} is not the sequential version at {_PIN}")
import sys
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "tests/unit")
import test_clubb_mixing_length as t
from legoesm.atmosphere.physics.turbulence.clubb import compute_mixing_length

kw = t._golden_inputs()
ng, nzm = kw["em"].shape
rng = np.random.default_rng(0)
# Non-uniform TKE so the up/down smoothing (running max/min) actually binds.
_ramp = 0.05 * np.exp(np.linspace(0.0, 5.0, nzm))    # TKE growing with height
# Column 1: weak stability + TKE growing steeply with height, where removing
# the DOWNWARD smoothing changes ~38 cells by up to ~350 m (searched with the
# smoothing deleted, scripts/tmp/_lscale_search.py); columns 0/2 keep the
# upward-smoothing and moist cases of the value golden.
_zt = np.asarray(kw["gr"].zt)
# Column 0: moderate stability + TKE large near the surface, where removing
# the UPWARD smoothing changes 14 cells by up to ~430 m (per-column search,
# scripts/tmp/_lscale_search_up.py; the moist column 2 never binds upward).
thv = np.asarray(kw["thvm"]).copy(); thv[1] = 300.0 + 0.001 * _zt[1]
thv[0] = 300.0 + 0.002 * _zt[0]
thv = jnp.asarray(thv)
em = jnp.asarray(np.stack([0.01 + 50.0 * np.linspace(1, 0, nzm) ** 4,
                           0.01 + 300.0 * np.linspace(0, 1, nzm) ** 4, _ramp]))
kw = dict(kw, em=em, thvm=thv, thlm=thv, thv_ds=thv)
w_up, w_dn = (jnp.asarray(rng.standard_normal((ng, kw["thlm"].shape[1]))) for _ in range(2))
names = ("thvm", "thlm", "rtm", "em", "mu")

def loss(*xs):
    L, Lu, Ld = compute_mixing_length(**dict(kw, **dict(zip(names, xs))))
    return jnp.sum(w_up * Lu) + jnp.sum(w_dn * Ld) + jnp.sum(L)

vals = compute_mixing_length(**kw)
grads = jax.grad(loss, argnums=tuple(range(len(names))))(*(kw[n] for n in names))
sha = _PIN
np.savez(sys.argv[1] if len(sys.argv) > 1 else "tests/unit/clubb_fixtures/clubb_lscale_grad_golden.npz", git=sha,
         em=np.asarray(em), thvm=np.asarray(thv), w_up=np.asarray(w_up), w_dn=np.asarray(w_dn),
         Lscale=np.asarray(vals[0]), Lscale_up=np.asarray(vals[1]), Lscale_down=np.asarray(vals[2]),
         **{f"grad_{n}": np.asarray(g) for n, g in zip(names, grads)})
print("sha", sha, "finite", all(bool(jnp.all(jnp.isfinite(g))) for g in grads))
for n, g in zip(names, grads): print(n, float(jnp.max(jnp.abs(g))))
