# 2Δx grid-mode DECAY-RATE reference (the §5 vorticity-flux residual gate).
#
# The §5 blow-up is a 2Δx-in-lon, v-dominant ROTATIONAL grid mode that legoESM
# UNDER-dissipates vs Oceananigans. The single-step per-node operators already match
# (tendency.nc), so the residual is the INTEGRATED dissipation of a developed grid mode.
# This deck gives the LOCALIZABLE, fast, deterministic gate: seed the bickley jet with a
# small 2Δx-in-lon v perturbation and dump v over a SHORT fixed-dt integration so the
# mode's DECAY RATE can be compared code-to-code (legoESM must match the oracle's rate).
#
# The perturbation is a CONTINUOUS function: A*sin(2π*(Nh/2)*(λ+180)/360)*env(φ). At the
# C-grid v-point lon CENTERS λ_i = -180+(i+1/2)Δλ this is exactly A*(-1)^i*env(φ) — the
# 2Δx-in-lon checkerboard — and since both codes share the same lon centers they get the
# IDENTICAL seed with no index bookkeeping (only an envelope in φ to sit in the jet core).
#
# Run (working env):
#   JULIA_DEPOT_PATH=/tmp/ocn_j11_depot julia +1.10.11 --project=/tmp/ocn_j11_gen \
#     scripts/data/generate_oceananigans_gridmode_decay_reference.jl <out_dir> [Nsteps] [dt] [Nh] [amp]

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection
using NCDatasets
using Printf

out_dir = length(ARGS) >= 1 ? ARGS[1] : "."
Nsteps  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 40
dt      = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.005
Nh      = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 64
amp     = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.01
mkpath(out_dir)

Nφ = Int(Nh / 2)
grid = LatitudeLongitudeGrid(size = (Nh, Nφ, 1), radius = 1,
                             longitude = (-180, 180), latitude = (-80, 80),
                             z = (0, 1), halo = (7, 7, 7))

# Bickley jet (identical to the bickley reference deck) + a 2Δx-in-lon v seed.
U(y) = sech(y)^2
ψ̃(x, y, ℓ, k) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k * x) * cos(k * y)
ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + y / ℓ^2)
ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x)
ϵ = 0.1; ℓ = 0.5; k = 0.5
dr(x) = deg2rad(x)
# 2Δx-in-lon checkerboard (sin of the Nyquist-in-lon at cell centers = (-1)^i), localised
# to the jet core by a Gaussian in φ; amplitude `amp` (small → linear decay regime).
m_nyq = Nh ÷ 2
vpert(x, y) = amp * sin(2π * m_nyq * (x + 180) / 360) * exp(-(y / 20)^2)
uᵢ(x, y, z) = U(dr(y)*8) + ϵ * ũ(dr(x)*2, dr(y)*8, ℓ, k)
vᵢ(x, y, z) = ϵ * ṽ(dr(x)*2, dr(y)*4, ℓ, k) + vpert(x, y)

momentum_advection = WENOVectorInvariant(vorticity_order = 9)
free_surface = ImplicitFreeSurface(gravitational_acceleration = 1)
model = HydrostaticFreeSurfaceModel(grid; momentum_advection,
                                    tracer_advection = WENO(),
                                    coriolis = HydrostaticSphericalCoriolis(rotation_rate = 1),
                                    free_surface, tracers = :c, buoyancy = nothing)
set!(model, u = uᵢ, v = vᵢ)

# SHORT FIXED-dt integration (no wizard): we measure the intrinsic decay of the seeded
# 2Δx mode over t = Nsteps*dt ≪ eddy time, so the base jet barely evolves and its own
# Nyquist content stays ~0 — the seeded mode dominates the Nyquist band.
vsnaps = Array{Float64,2}[]
push!(vsnaps, Array(interior(model.velocities.v)[:, :, 1]))
for n in 1:Nsteps
    time_step!(model, dt)
    push!(vsnaps, Array(interior(model.velocities.v)[:, :, 1]))
end

# 2Δx-in-lon amplitude per step: RMS over lat of |rfft_lon(v)[Nyquist]| (last bin).
function amp2dx(v)
    nx, ny = size(v)
    s = 0.0
    for j in 1:ny
        f = abs(sum(v[i, j] * cispi(-2 * m_nyq * (i - 1) / nx) for i in 1:nx)) / nx
        s += f^2
    end
    return sqrt(s / ny)
end
amps = [amp2dx(vs) for vs in vsnaps]
@printf("gridmode decay ref: Nh=%d Nsteps=%d dt=%.4f amp0=%.4e ampN=%.4e ratio=%.4f\n",
        Nh, Nsteps, dt, amps[1], amps[end], amps[end] / amps[1])

nc = joinpath(out_dir, "gridmode_decay.nc")
NCDataset(nc, "c") do ds
    nt = length(vsnaps); nx, ny = size(vsnaps[1])
    defDim(ds, "t", nt); defDim(ds, "xv", nx); defDim(ds, "yv", ny)
    ds.attrib["dt"] = dt
    ds.attrib["amp_seed"] = amp
    ds.attrib["m_nyq"] = m_nyq
    defVar(ds, "amp2dx", amps, ("t",))
    vv = defVar(ds, "v", Float64, ("t", "xv", "yv"))
    for n in 1:nt; vv[n, :, :] = vsnaps[n]; end
end
@info "DONE: wrote $nc (amp ratio $(amps[end]/amps[1]))"
