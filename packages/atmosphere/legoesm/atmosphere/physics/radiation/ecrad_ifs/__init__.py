"""JAX port of ecRad as bundled in OpenIFS 49r1 (8f6f722), IFS configuration with
RRTMG gas optics (user decision A14). Used only by the 49r1 IFS lane; RRTMGP
remains the radiation scheme everywhere else.

Oracle: scripts/validate/oracle/ecrad_ifs/ (49r1 Fortran built unchanged).
"""
