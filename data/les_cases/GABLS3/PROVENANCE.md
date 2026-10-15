# GABLS3 forcing — provenance

`gabls3_scm_cabauw_obs_v33.nc`: the official GABLS3 single-column-model case
(Cabauw, 00-09 UTC 2 July 2006; Bosveld et al. 2014, BLM 152, 133,
doi:10.1007/s10546-014-9917-3), version 3.3, as distributed by KNMI:

- page: https://www.knmiprojects.nl/documents/2020/02/04/gabls3-scm-cabauw-obs-v33
- file: gabls3_scm_cabauw_obs_v33.zip (2020-02-04), unzipped unmodified
- staged 2026-10-15 with the user's approval
- sha256 09b4d2f81642929490828efe094226fed10d467167852d4bb20d5480a982b9f6
- md5    9973eb489281d5cdc194fcb609402438

Not yet wired into the LES suite: the case is moist and forced by time- and
height-varying geostrophic wind and advection, which the dry plane-LES builder
does not carry, and the GABLS3 LES study (Basu et al. 2012) used its own
simplified LES forcing that is not in this file.

The GABLS4 stage-3 and Ayotte (1996) forcings approved the same day are NOT
vendored: they are DEPHY-format files registered (pinned commit + SHA-256) in
`legoesm.atmosphere.forcing.scm.dephy_scm.DEPHY_SCM_CASES` and fetched into the
gitignored `data/les_cases/dephy/` cache by `scripts/data/fetch_dephy_cases.py`,
the repo's convention for DEPHY cases.
