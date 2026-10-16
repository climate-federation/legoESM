# SWG_DG portable fixtures

These text fixtures replace tests that previously depended on one developer's
NEMO checkout. `namelist_cfg.shipped` and `usrdef_zgr.F90.shipped` are from
NEMO 5.0.2 revision `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`, relative
paths `tests/SWG/EXPREF/namelist_cfg` and
`tests/SWG/MY_SRC/usrdef_zgr.F90`. `output.namelist.dyn`, `namelist_ref` and
`namelist_cfg.maddison_3yr` are the resolved echo and the two content-hashed
namelist inputs of the pinned three-year Maddison reference run
(`md_dg_dx6p25_dt50_3yr`); `namelist_ref` is NEMO's shared
`cfgs/SHARED/namelist_ref` at the same revision. They are test inputs only,
not model outputs cited as fresh acquisition evidence.

SHA-256:

- `namelist_cfg.shipped`: `6cc239d380f81be628348f655f5d0dca852c6d458f1459b05444293095a3ff7a`
- `usrdef_zgr.F90.shipped`: `14f6640b6720d5bda4cd7542dde65dd0add5aebd1d262000ed3b492172b5e06b`
- `output.namelist.dyn`: `a8f53c412dddb65128899c5f47ab7e82a4a5266c3a0d2929ce91a1d60f1970d3`
- `namelist_ref`: `b04f2ce4d12aa247ae3b707483d25cf2d38d0b3cf8817d71d33e66918fd81cfd`
- `namelist_cfg.maddison_3yr`: `0f6fc303af68549c0c78dc1f22f9e29ed5a6feb3b4fafc88b6ee4281bb42e309`
