"""Plot RCE log trajectory + snapshot maps."""
import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

out_dir = Path(sys.argv[1])
log_path = out_dir / "log.txt"
times, cwv_mean, cwv_max, mse, max_w, max_qc, max_qr, precip = [], [], [], [], [], [], [], []
with open(log_path) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        c = line.strip().split(",")
        if len(c) < 10:
            continue
        times.append(float(c[1]))
        cwv_mean.append(float(c[2])); cwv_max.append(float(c[3]))
        mse.append(float(c[4])); max_w.append(float(c[5]))
        max_qc.append(float(c[6])); max_qr.append(float(c[7]))
        precip.append(float(c[8]))

t = np.array(times) * 24
fig, ax = plt.subplots(3, 2, figsize=(12, 9))
ax[0, 0].plot(t, cwv_mean, label="mean")
ax[0, 0].plot(t, cwv_max, "r--", label="max")
ax[0, 0].set_ylabel("CWV [kg/m^2]")
ax[0, 0].legend(); ax[0, 0].grid()
ax[0, 1].plot(t, np.array(mse) / 1e9)
ax[0, 1].set_ylabel("MSE [GJ/m^2]"); ax[0, 1].grid()
ax[1, 0].plot(t, max_w); ax[1, 0].set_ylabel("max|w| [m/s]"); ax[1, 0].grid()
ax[1, 1].plot(t, np.array(max_qc) * 1000); ax[1, 1].set_ylabel("max qc [g/kg]"); ax[1, 1].grid()
ax[2, 0].plot(t, np.array(max_qr) * 1000); ax[2, 0].set_ylabel("max qr [g/kg]"); ax[2, 0].grid()
ax[2, 1].plot(t, precip); ax[2, 1].set_ylabel("max precip [mm/day]"); ax[2, 1].grid()
for a in ax.flat:
    a.set_xlabel("sim hours")
fig.suptitle(f"RCE trajectory: {out_dir.name}")
fig.tight_layout()
out_png = out_dir / "trajectory.png"
fig.savefig(out_png, dpi=110)
print(f"saved {out_png}")
