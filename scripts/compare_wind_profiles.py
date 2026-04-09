"""Compare candidate wind stress profiles for global barotropic tests."""
import numpy as np
import matplotlib.pyplot as plt

lat_deg = np.linspace(-90, 90, 361)
lat = np.radians(lat_deg)

# --- Current profile ---
taper = np.cos(lat) ** 2
current = -0.1 * np.cos(2.0 * lat) * taper

# --- Option A: Bryan-Cox classic (just remove the cos^2 taper) ---
# Used by Bryan (1963), Cox (1975). Simple, well-tested.
# Zero crossings at 45 deg (not 30), but westerlies are full strength.
optA = -0.1 * np.cos(2.0 * lat)

# --- Option B: Two-harmonic fit with zero crossings at 30 and 90 deg ---
# cos(2phi) + cos(4phi) = 2*cos(3phi)*cos(phi)
# Zeros at 30 and 90 deg. No polar easterlies (westerlies extend to poles).
# Trades and westerlies of comparable magnitude.
optB = -0.05 * (np.cos(2 * lat) + np.cos(4 * lat))

# --- Option C: Realistic 3-belt with sin-based profile ---
# tau_x = tau_0 * [sin(pi*(lat_deg + 5)/55) for lat in (-50,60)]
# Piecewise smooth, tuned to observed zero crossings and amplitudes.
# Trades peak ~0.07 at 15 deg, westerlies peak ~0.15 at 45 deg,
# polar easterlies ~0.02 at 75 deg. Southern ocean westerlies
# intentionally stronger than NH.
# Simple approximation: shifted sine
tau_0 = 0.15
optC = tau_0 * np.sin(np.radians(lat_deg - 15) * 3 * np.pi / 180)
# Actually let's do something cleaner:
# Construct from: strong westerlies centered at 45, trades at 15, polar E at 75
# Use the standard Fourier fit with correct zeros at ~30, ~60:
# f(phi) = A*sin(pi*(lat-30)/30) for 0-60, modified near poles
# Simpler: just use a polynomial fit to observed data

# Clean 3-belt profile (symmetric about equator):
# Zeros at 30, 60, and 90 deg.
# tau_x = -tau_0 * (1 + 2cos(2phi) + 2cos(4phi) + cos(6phi)) / 6
# But this gives trades 6x westerlies. Modify with weighting.
#
# Better: use a "stretched cosine" approach
# Map 0->0, 30->45, 60->90, 90->135 and use cos(2*phi_s)
# phi_s = 1.5*phi gives cos(3*phi), zeros at 30, 90 but no 60.
#
# Cleanest 3-belt: just cos(2phi) with a phase shift
# tau_x = -tau_0 * cos(2*(phi + shift))
# shift=15 deg: zeros at 30 and 120... no.
# This doesn't work simply.

# Practical approach: Hellerman-Rosenstein style polynomial fit
# tau_x ~ 0.1 * [-1 + 5*sin(phi)^2 - 4*sin(phi)^4] * cos(phi)
poly = 0.1 * (-1 + 5*np.sin(lat)**2 - 4*np.sin(lat)**4) * np.cos(lat)
# Check: phi=0: 0.1*(-1)*1 = -0.1 (trades)
# phi=30: 0.1*(-1+5*0.25-4*0.0625)*cos(30) = 0.1*(0)*0.866 = 0 (zero!)
# phi=45: 0.1*(-1+5*0.5-4*0.25)*cos(45) = 0.1*(0.5)*0.707 = 0.035
# phi=60: 0.1*(-1+5*0.75-4*0.5625)*cos(60) = 0.1*(0.5)*0.5 = 0.025
# phi=90: 0 (pole)
# Hmm westerlies still weak. Let me amplify.

optC = 0.2 * (-1 + 5*np.sin(lat)**2 - 4*np.sin(lat)**4) * np.cos(lat)

# --- Option D: Vallis-style sin^2 profile ---
# From Vallis (2017), a profile resembling observed zonal mean:
# tau_x = tau_0 * (0.2 - 0.8*sin^2(lat)) for equatorward of 60 deg
# with taper poleward.
# Let me try: sin^2(lat) goes from 0 at equator to 1 at pole
# 0.2 - 0.8*sin^2 = 0 when sin^2 = 0.25, i.e., lat = 30 deg!
optD_base = 0.2 * (0.2 - 0.8 * np.sin(lat)**2)
# This gives: equator = 0.04 (easterly), 30 = 0, 45 = -0.04, 90 = -0.12
# Only one zero crossing at 30, westerlies grow with latitude. No polar easterlies.
# Add a high-latitude correction:
polar_corr = 0.15 * np.exp(-((lat_deg - 70)**2 + (lat_deg + 70)**2) / (2*15**2))
# Actually simpler: use sin(lat)^4 to bring it back near poles
optD = 0.2 * (0.2 - 0.8*np.sin(lat)**2 + 0.6*np.sin(lat)**4) * np.sign(np.cos(lat))
# Hmm getting complicated. Let me just go with a known good formula.

# --- Option D (revised): Simple and clean ---
# tau_x = -tau_0 * cos(2*phi) but with tau_0 = 0.2 Pa (compensate for
# the fact that cos(2phi) peaks at equator, not at 15 deg)
optD = -0.2 * np.cos(2.0 * lat)

# --- Option E: Best practical 3-belt ---
# Use product form that gives zero at 30 deg and taper to zero at poles:
# tau_x = tau_0 * cos(phi) * (2*sin^2(phi) - 0.5)
# sin^2(30)=0.25, so 2*0.25-0.5=0 -> zero at 30!
# sin^2(90)=1, so 2*1-0.5=1.5, cos(90)=0 -> zero at pole
# But only one zero crossing (at 30). Need to add polar easterlies.
#
# tau_x = tau_0 * cos(phi) * (6*sin^2(phi) - 1) * (1 - 2*sin^2(phi))
# = tau_0 * cos(phi) * (6s^2 - 1)(1 - 2s^2) where s=sin(phi)
# zeros at sin^2=1/6 (phi~24 deg) and sin^2=1/2 (phi=45 deg)... not quite
#
# Simplest for 3-belt: product of two factors with zeros at 30 and 60
# Factor 1: (sin^2(phi) - sin^2(30)) = sin^2(phi) - 0.25
# Factor 2: (sin^2(phi) - sin^2(60)) = sin^2(phi) - 0.75
# tau_x = A * (sin^2(phi)-0.25)(sin^2(phi)-0.75) * cos(phi)
s2 = np.sin(lat)**2
optE = -0.8 * (s2 - 0.25) * (s2 - 0.75) * np.cos(lat)
# Check: phi=0: -0.8*(-0.25)*(-0.75)*1 = -0.8*0.1875 = -0.15 (trades)
# phi=30: factor=0 -> 0 ✓
# phi=45: -0.8*(0.5-0.25)(0.5-0.75)*0.707 = -0.8*0.25*(-0.25)*0.707 = +0.035
# phi=60: factor=0 -> 0 ✓
# phi=75: s2=0.933, -0.8*(0.683)*(0.183)*cos(75) = -0.8*0.125*0.259 = -0.026 (polar E)
# phi=90: cos(90)=0 -> 0 ✓

# ======================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Candidate Wind Stress Profiles for Global Barotropic Tests',
             fontsize=14, fontweight='bold')

profiles = [
    ("Current\n-0.1·cos(2φ)·cos²(φ)", current),
    ("Option A: Bryan-Cox\n-0.1·cos(2φ)", optA),
    ("Option B: Two-harmonic\n-0.05·[cos(2φ)+cos(4φ)]", optB),
    ("Option C: Polynomial\n0.2·(-1+5s²-4s⁴)·cos(φ)", optC),
    ("Option D: Stronger Bryan-Cox\n-0.2·cos(2φ)", optD),
    ("Option E: 3-belt quartic\n-0.8·(s²-¼)(s²-¾)·cos(φ)", optE),
]

for ax, (title, tau) in zip(axes.flat, profiles):
    ax.plot(lat_deg, tau, 'k-', linewidth=2)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(30, color='blue', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(-30, color='blue', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(60, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(-60, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlim(-90, 90)
    ax.set_xlabel('Latitude (deg)')
    ax.set_ylabel('τ_x (Pa)')
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    # Annotate peak values
    idx_pos = np.argmax(tau[181:]) + 181  # NH
    idx_neg_nh = np.argmin(tau[181:]) + 181
    ax.text(0.02, 0.98,
            f'max={tau.max():.3f}\nmin={tau.min():.3f}',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/ocean/wind_profile_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/ocean/wind_profile_comparison.png")

# Also make the overlay comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(lat_deg, current, 'k--', linewidth=1.5, label='Current: -0.1·cos(2φ)·cos²(φ)')
ax.plot(lat_deg, optA, 'b-', linewidth=1.5, label='A: -0.1·cos(2φ) [Bryan-Cox]')
ax.plot(lat_deg, optE, 'r-', linewidth=2, label='E: -0.8·(s²-¼)(s²-¾)·cos(φ) [3-belt]')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(30, color='gray', linewidth=0.5, linestyle=':', alpha=0.7, label='30°/60° (observed zero crossings)')
ax.axvline(-30, color='gray', linewidth=0.5, linestyle=':', alpha=0.7)
ax.axvline(60, color='gray', linewidth=0.5, linestyle=':', alpha=0.7)
ax.axvline(-60, color='gray', linewidth=0.5, linestyle=':', alpha=0.7)
ax.set_xlim(-90, 90)
ax.set_xlabel('Latitude (deg)', fontsize=12)
ax.set_ylabel('τ_x (N/m²)', fontsize=12)
ax.set_title('Wind Stress Profile Comparison', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/ocean/wind_profile_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/ocean/wind_profile_overlay.png")
