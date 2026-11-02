"""Constant 2x2 MOM6 'file' buoyancy forcing: SW and sensible heat (W/m2, + into ocean), rest zero."""
import sys, numpy as np, netCDF4
out, sw, sens = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
with netCDF4.Dataset(out, "w") as d:
    d.createDimension("time", None); d.createDimension("lat", 2); d.createDimension("lon", 2)
    t = d.createVariable("time", "f8", ("time",)); t.units = "days since 0001-01-01 00:00:00"
    t.calendar = "noleap"; t.cartesian_axis = "T"; t[0] = 0.0
    for name, val in [("SW", sw), ("sensible", sens), ("LW", 0.0), ("evap", 0.0), ("latent", 0.0),
                      ("liq_precip", 0.0), ("froz_precip", 0.0), ("liq_runoff", 0.0), ("froz_runoff", 0.0)]:
        v = d.createVariable(name, "f8", ("time", "lat", "lon")); v[0] = np.full((2, 2), val)
