# Silvestri §5 forced-jet ORACLE reference (issue #673) — the previously-MISSING
# reference: every other fidelity case has a generated .nc; the §5 forced jet's
# "oracle saturates ~0.10 m/s" was only ever a short-run/estimate. This deck is
# the exact matched §5 oracle (physics verbatim from the proven probe
# /tmp/ocn_silvestri/oracle_saturation.jl, run to day 80 on 2026-06-20), promoted
# to a repo generator with NetCDF output: daily max|u| + EKE trajectory, surface
# b/u/v snapshots, and final zonal-mean profiles — enough for both the #673
# amplitude verdict AND the eddy-energy-budget diagnosis.
#
# Matched §5 (legoESM silvestri_baroclinic_jet): 128x160x50, Lx=16°·R·cos50 (the
# oracle is a CARTESIAN beta-plane, Ly=20°·R), dz=20 m, BetaPlane(-50), front
# Eqs 52-53 (B(φ) profile, Δb=5e-3, N²=4e-6), thermal-wind-balanced IC,
# ZONAL-MEAN buoyancy restoring τ=50 d (Soufflet-style: forces ⟨b⟩_x only, eddies
# untouched — matching legoESM's apply_zonal_mean_restoring; NOTE legoESM also
# restores u,v means, this deck b only, faithful to the original probe),
# WENOVectorInvariant(vorticity_order=9) + WENO(order=7) tracer, NO closure,
# TimeStepWizard cfl=0.3 Δt≤15 min.
#
# Run (known-good harness env, GPU):
#   JULIA_DEPOT_PATH=/tmp/ocn_j11_depot CUDA_VISIBLE_DEVICES=1 julia +1.10.11 \
#     --project=/tmp/ocn_j11_gen \
#     scripts/data/generate_oceananigans_silvestri_s5_reference.jl <ref_root> [stop_days] [dump_days]
# Output: <ref_root>/silvestri_s5/silvestri_s5.nc
using Oceananigans, Oceananigans.Units, NCDatasets, Printf, Random, Statistics
using Oceananigans.Fields: Field, compute!, interior
using Oceananigans.AbstractOperations: Average
using Oceananigans.Grids: znode, ynode, Center
using CUDA
Random.seed!(42)
const ARCH = get(ENV, "ARCH", "GPU") == "GPU" ? GPU(CUDA.CUDABackend()) : CPU()

ref_root  = length(ARGS) >= 1 ? ARGS[1] : "."
stop_days = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 200.0
dump_days = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 10.0
out_dir = joinpath(ref_root, "silvestri_s5"); mkpath(out_dir)

# --- §5 setup (verbatim physics from the proven oracle_saturation.jl probe) ---
const R = 6.371e6; const N2c = 4.0e-6; const DB = 5.0e-3
const DPHI = deg2rad(20.0); const Ly = DPHI*R; const Lx = deg2rad(16.0)*R*cosd(50)
const H = 1000.0
@inline function Bprof(y)
    γ = π/2 - 2π*(y/R)/DPHI
    γ < 0 && return 0.0; γ > π && return 1.0
    return (γ - sin(γ)*cos(γ))/π
end
b_ini(x,y,z) = N2c*z + DB*Bprof(y) + 1e-3*DB*randn()
const FCOR = 2*7.292e-5*sind(-50)
@inline function dBdy(y)
    γ = π/2 - 2π*(y/R)/DPHI
    (γ < 0 || γ > π) && return 0.0
    return -4*sin(γ)^2 / (R*DPHI)
end
u_tw(x,y,z) = -(DB*dBdy(y)/FCOR)*(z+H)

grid = RectilinearGrid(ARCH, size=(128,160,50), x=(0,Lx), y=(-Ly/2,Ly/2), z=(-H,0),
                       topology=(Periodic,Bounded,Bounded), halo=(6,6,4))
b̄ = Field{Nothing,Center,Center}(grid)
@inline function zm_restore(i, j, k, grid, clock, fields, p)
    @inbounds bbar = p.b̄[1, j, k]
    y = ynode(j, grid, Center()); z = znode(k, grid, Center())
    return -p.γ * (bbar - (N2c*z + DB*Bprof(y)))
end
b_forcing = Forcing(zm_restore, discrete_form=true, parameters=(; γ=1/50days, b̄=b̄))
model = HydrostaticFreeSurfaceModel(grid; coriolis=BetaPlane(latitude=-50),
    buoyancy=BuoyancyTracer(), tracers=:b,
    momentum_advection=WENOVectorInvariant(vorticity_order=9),
    tracer_advection=WENO(order=7), forcing=(; b=b_forcing), closure=nothing)
set!(model, b=b_ini, u=u_tw)
bbar_field = Field(Average(model.tracers.b, dims=1))
update_bbar!(sim) = (compute!(bbar_field); parent(b̄) .= parent(bbar_field); nothing)

function eke(model)
    u = Array(interior(model.velocities.u)); v = Array(interior(model.velocities.v))
    up = u .- mean(u, dims=1); vp = v .- mean(v, dims=1)
    return 0.5*(sum(up.^2) + sum(vp.^2))
end

sim = Simulation(model, Δt=5minutes, stop_time=stop_days*days)
conjure_time_step_wizard!(sim, IterationInterval(20), cfl=0.3, max_Δt=15minutes)
add_callback!(sim, update_bbar!, IterationInterval(1))

# daily trajectory + periodic surface snapshots
traj_t = Float64[]; traj_umax = Float64[]; traj_eke = Float64[]
snap_t = Float64[]; snap_b = Array{Float64,2}[]; snap_u = Array{Float64,2}[]; snap_v = Array{Float64,2}[]
Nzs = 50
function snap!()
    push!(snap_t, time(sim))
    push!(snap_b, Array{Float64}(interior(model.tracers.b)[:, :, Nzs]))
    push!(snap_u, Array{Float64}(interior(model.velocities.u)[:, :, Nzs]))
    push!(snap_v, Array{Float64}(interior(model.velocities.v)[:, :, Nzs]))
end
blew = Ref(false)
add_callback!(sim, s->(um=maximum(abs,s.model.velocities.u); ek=eke(s.model);
    push!(traj_t, time(s)); push!(traj_umax, um); push!(traj_eke, ek);
    @printf("  day %5.1f  max|u|=%.4e  EKE=%.4e  dt=%.1fmin\n",
            time(s)/day, um, ek, s.Δt/60); flush(stdout);
    (!isfinite(um)||um>50) && (blew[]=true; stop!(s))), TimeInterval(1days))
add_callback!(sim, _ -> snap!(), TimeInterval(dump_days*days))

@printf("silvestri_s5 ORACLE: 128x160x50 W9V no-closure, zonal-mean b-restore τ=50d, stop=%.0fd\n", stop_days)
flush(stdout)
update_bbar!(sim); snap!()
try; run!(sim); catch e; blew[]=true; @printf("  ERR %s\n", e); end
snap!()  # final state

# final zonal means for the budget diagnosis
ubar_f = Field(Average(model.velocities.u, dims=1)); compute!(ubar_f)
compute!(bbar_field)
u_zm = Array{Float64}(interior(ubar_f)[1, :, :]); b_zm = Array{Float64}(interior(bbar_field)[1, :, :])

@printf(">>> ORACLE §5: %s  max|u|=%.3e EKE=%.3e t=%.1fd\n",
        blew[] ? "BLEW" : @sprintf("SURVIVED %.0fd", stop_days),
        maximum(abs, model.velocities.u), eke(model), time(sim)/day); flush(stdout)

nc = joinpath(out_dir, "silvestri_s5.nc")
NCDataset(nc, "c") do ds
    defDim(ds, "traj_t", length(traj_t)); defDim(ds, "snap_t", length(snap_t))
    nxu, nyu = size(snap_u[1]); nxv, nyv = size(snap_v[1]); nxb, nyb = size(snap_b[1])
    nyz1, nyz2 = size(u_zm); nbz1, nbz2 = size(b_zm)
    defDim(ds, "xu", nxu); defDim(ds, "yu", nyu); defDim(ds, "xv", nxv); defDim(ds, "yv", nyv)
    defDim(ds, "xb", nxb); defDim(ds, "yb", nyb)
    defDim(ds, "y_zm_u", nyz1); defDim(ds, "z_zm_u", nyz2)
    defDim(ds, "y_zm_b", nbz1); defDim(ds, "z_zm_b", nbz2)
    defVar(ds, "traj_times_s", traj_t, ("traj_t",))
    defVar(ds, "traj_umax", traj_umax, ("traj_t",))
    defVar(ds, "traj_eke", traj_eke, ("traj_t",))
    defVar(ds, "snap_times_s", snap_t, ("snap_t",))
    bv = defVar(ds, "b_surf", Float64, ("snap_t","xb","yb"))
    uv = defVar(ds, "u_surf", Float64, ("snap_t","xu","yu"))
    vv = defVar(ds, "v_surf", Float64, ("snap_t","xv","yv"))
    for i in 1:length(snap_t); bv[i,:,:] = snap_b[i]; uv[i,:,:] = snap_u[i]; vv[i,:,:] = snap_v[i]; end
    defVar(ds, "u_zonal_mean_final", u_zm, ("y_zm_u","z_zm_u"))
    defVar(ds, "b_zonal_mean_final", b_zm, ("y_zm_b","z_zm_b"))
    ds.attrib["blew_up"] = blew[] ? 1 : 0
    ds.attrib["description"] = "Oceananigans 0.110.4 Silvestri S5 forced-jet oracle (issue #673): W9V no-closure, zonal-mean b restoring tau=50d, Cartesian beta-plane 128x160x50"
end
@info "DONE: wrote $nc"
