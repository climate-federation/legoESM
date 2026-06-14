"""Recreate the Nikurashin & Vallis (2012)-style wind stress profile.

Target features from the screenshot:
- Trades: ~0.08 Pa easterly, peaking at ~±15 deg
- Zero crossing at ~±30 deg
- Westerlies: ~0.10 Pa, peaking at ~±50 deg
- Westerlies come back to zero (or slightly negative) by ~±65-70 deg
- Smooth, symmetric about equator
"""
import numpy as np
import matplotlib.pyplot as plt

lat_deg = np.linspace(-75, 75, 1001)
lat = np.radians(lat_deg)

# -------------------------------------------------------------------
# Use a higher-order polynomial in sin(phi) with cos(phi) envelope
# to force the curve back toward zero at high latitudes.
#
# tau(phi) = (a + b*sin^2 + c*sin^4 + d*sin^6) * cos(phi)
#
# Constraints:
#   tau(0)  = a                              = -0.08  (trades)
#   tau(30) = 0                                        (zero crossing)
#   tau(50) = +0.10                                    (westerly peak)
#   tau(70) ≈ 0                                        (return to zero)
#
# 4 constraints, 4 unknowns (a, b, c, d)
# -------------------------------------------------------------------

# sin^2 values at key latitudes
s30 = np.sin(np.radians(30))**2   # 0.25
s50 = np.sin(np.radians(50))**2   # 0.587
s70 = np.sin(np.radians(70))**2   # 0.883

# cos values for the outer envelope
c30 = np.cos(np.radians(30))  # 0.866
c50 = np.cos(np.radians(50))  # 0.643
c70 = np.cos(np.radians(70))  # 0.342

a = -0.08

# System:
# (a + b*s30 + c*s30^2 + d*s30^3) * c30 = 0        ... (1)
# (a + b*s50 + c*s50^2 + d*s50^3) * c50 = 0.10      ... (2)
# (a + b*s70 + c*s70^2 + d*s70^3) * c70 = 0          ... (3)

# Simplify by dividing out cos:
# a + b*s30 + c*s30^2 + d*s30^3 = 0                  ... (1')
# a + b*s50 + c*s50^2 + d*s50^3 = 0.10/c50 = 0.1555  ... (2')
# a + b*s70 + c*s70^2 + d*s70^3 = 0                  ... (3')

# Subtract a from all:
# b*s30 + c*s30^2 + d*s30^3 = 0.08                    ... (1'')
# b*s50 + c*s50^2 + d*s50^3 = 0.2355                  ... (2'')
# b*s70 + c*s70^2 + d*s70^3 = 0.08                    ... (3'')

# Set up 3x3 linear system
A_mat = np.array([
    [s30,   s30**2, s30**3],
    [s50,   s50**2, s50**3],
    [s70,   s70**2, s70**3],
])
rhs = np.array([0.08, 0.10/c50 - a, 0.08])  # 0.08, 0.2355, 0.08

bcd = np.linalg.solve(A_mat, rhs)
b, c, d = bcd

print(f"Coefficients: a={a}, b={b:.6f}, c={c:.6f}, d={d:.6f}")

# Compute profile
s2 = np.sin(lat)**2
tau_nv = (a + b*s2 + c*s2**2 + d*s2**3) * np.cos(lat)

# Verify at key latitudes
for phi_check in [0, 10, 15, 20, 30, 40, 50, 55, 60, 65, 70]:
    phi_r = np.radians(phi_check)
    s = np.sin(phi_r)**2
    val = (a + b*s + c*s**2 + d*s**3) * np.cos(phi_r)
    print(f"  phi={phi_check:3d} deg:  tau = {val:+.4f} Pa")

# Current profile for comparison
tau_current = -0.1 * np.cos(2.0 * lat) * np.cos(lat)**2

# -------------------------------------------------------------------
# Plot matching the screenshot style
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Panel 1: The N&V-style profile
ax = axes[0]
ax.plot(lat_deg, tau_nv, 'b-', linewidth=2.5)
ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
# Dashed vertical lines at roughly ±30 and ±65 like the screenshot
for x in [-65, -30, 30, 65]:
    ax.axvline(x, color='k', linewidth=0.5, linestyle='--', alpha=0.4)
# Gray shading for Southern Ocean
ax.axvspan(-75, -45, alpha=0.15, color='gray')
ax.set_xlim(-75, 75)
ax.set_ylim(-0.12, 0.12)
ax.set_xlabel('Latitude (deg)', fontsize=11)
ax.set_ylabel(r'$\tau$ [Pa]', fontsize=12)
ax.set_title('Proposed: Nikurashin & Vallis style', fontsize=11)
ax.grid(True, alpha=0.2)

# Panel 2: Overlay comparison
ax = axes[1]
ax.plot(lat_deg, tau_nv, 'b-', linewidth=2.5, label='Proposed (N&V style)')
ax.plot(lat_deg, tau_current, 'k--', linewidth=1.5, label=r'Current: $-0.1\cos(2\phi)\cos^2\phi$')
ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
ax.axvspan(-75, -45, alpha=0.15, color='gray')
ax.set_xlim(-75, 75)
ax.set_ylim(-0.12, 0.12)
ax.set_xlabel('Latitude (deg)', fontsize=11)
ax.set_ylabel(r'$\tau$ [Pa]', fontsize=12)
ax.set_title('Comparison', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('results/ocean/wind_profile_nikurashin_vallis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: results/ocean/wind_profile_nikurashin_vallis.png")

print(f"\nFormula: tau_x = ({a} + {b:.4f}*sin^2(phi) + {c:.4f}*sin^4(phi) + {d:.4f}*sin^6(phi)) * cos(phi)")
