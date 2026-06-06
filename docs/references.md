# References

## Citing legoESM

legoESM ships citation metadata at the repository root:

- `CITATION.cff` — use GitHub's **"Cite this repository"** button, or a reference
  manager that reads Citation File Format.
- `.zenodo.json` — a Zenodo-linked release mints a DOI; cite the DOI of the version
  you used.

## Technical specification

The full equation-level specification lives at
[`docs/specs/SPECIFICATION.md`](https://github.com/gentine/legoESM/blob/main/docs/specs/SPECIFICATION.md)
(LaTeX sources `docs/legoesm_documentation.tex` and `docs/legoesm_scientific_guide.tex`).

## Validation benchmarks

The dynamical cores and physics are validated against the standard literature test
cases — see the [Dycore validation catalog](dycore_validation_catalog.md) for the
configurations and pass criteria:

- **Williamson et al. (1992)** — shallow-water test suite (geostrophic flow, isolated
  mountain, Rossby–Haurwitz wave).
- **Galewsky et al. (2004)** — barotropically unstable mid-latitude jet.
- **Jablonowski & Williamson (2006)** — baroclinic-wave dynamical-core test.
- **DCMIP** — Dynamical Core Model Intercomparison Project cases.
- **Held & Suarez (1994)** — idealized dry GCM climate benchmark.
- Ocean: OMIP/CORE2 and NEMO-faithfulness comparisons (notes under
  [`docs/md_files/`](md_files/README.md)).

## Internal development notes

Working notes, plans, audits, fidelity reviews, and development logs are collected
under [`docs/md_files/`](md_files/README.md). They are kept for history and referenced
from code by basename; they are not part of this published documentation site.
