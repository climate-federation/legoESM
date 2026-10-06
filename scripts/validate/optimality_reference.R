# Reference table from the ORIGINAL optimality codes, for the leaf-level
# comparison against legoESM's g1-based P-model / P-hydro wiring
# (scripts/validate/optimality_gs_gap.py).
#
#   R1  rpmodel  (Stocker et al. 2020, geco-bern/rpmodel)       least-cost
#   R2  rphydro  (Joshi et al. 2022, jaideep777/phydro @a59e2c6) P-hydro
#
# Usage (conda env legoesm-pmodel):
#   Rscript scripts/validate/optimality_reference.R tests/land/unit/data/optimality_reference.csv
#
# Units out: gs = stomatal conductance to CO2 [mol m-2 s-1]; a [umol m-2 s-1];
# e [mol H2O m-2 s-1]; psi/dpsi [MPa]; vcmax/jmax at tc [umol m-2 s-1].

suppressPackageStartupMessages({library(rpmodel); library(rphydro)})

out_path <- commandArgs(trailingOnly = TRUE)[1]
stopifnot(!is.na(out_path))

kphio <- 0.081785; co2 <- 400; patm <- 101325; elv <- 0; fapar <- 1; rdark <- 0
par_cost <- list(alpha = 0.1, gamma = 1)
vpd_ref <- 1000  # [Pa] acclimation VPD for the instantaneous (fast) response

# Plant conductivity [m] whose scaled value at 25 degC equals legoESM's
# PHydroConfig.gplant_mol = 4e-3 mol m-2 s-1 MPa-1 (phydro scale_conductivity).
k_scaled <- function(K, tc) K / calc_viscosity_h2o(tc, patm) * calc_density_h2o(tc, patm) * 55.5 * 1e6
conductivity <- 4e-3 / k_scaled(1, 25)

grid <- expand.grid(
  tc = c(10, 20, 30), ppfd = c(300, 1000),
  vpd = c(300, 600, 1000, 2000, 3000, 4000),
  psi_soil = c(0, -0.25, -0.5, -1, -1.5, -2, -3),
  psi50 = c(-1, -2, -4, -1000),  # -1000: P(psi) ~ 1, the no-vulnerability limit
  KEEP.OUT.ATTRS = FALSE)

rows <- lapply(seq_len(nrow(grid)), function(i) {
  g <- grid[i, ]
  pp <- list(conductivity = conductivity, psi50 = g$psi50, b = 1.5)

  r1 <- rpmodel(tc = g$tc, vpd = g$vpd, co2 = co2, fapar = fapar, ppfd = g$ppfd,
                patm = patm, kphio = kphio, beta = 146, method_jmaxlim = "wang17",
                do_ftemp_kphio = TRUE, do_soilmstress = FALSE)
  r1_a <- r1$gs * (r1$ca - r1$ci)        # rpmodel's own assim (not returned)
  r1_gs <- r1$gs * patm * 1e-6           # umol/m2/s/Pa -> mol/m2/s

  acc <- tryCatch(rphydro_analytical(g$tc, g$ppfd, g$vpd, co2, elv, fapar, kphio,
                                     g$psi_soil, rdark, pp, par_cost),
                  error = function(e) NULL)
  ref <- tryCatch(rphydro_analytical(g$tc, g$ppfd, vpd_ref, co2, elv, fapar, kphio,
                                     g$psi_soil, rdark, pp, par_cost),
                  error = function(e) NULL)
  ins <- if (is.null(ref)) NULL else tryCatch(
    rphydro_instantaneous_analytical(ref$vcmax, ref$jmax, g$tc, g$ppfd, g$vpd, co2,
                                     elv, fapar, kphio, g$psi_soil, rdark, pp, par_cost),
    error = function(e) NULL)
  f <- function(x, n) if (is.null(x)) NA_real_ else x[[n]]

  data.frame(g, k_scaled = k_scaled(conductivity, g$tc),
    r1_chi = r1$chi, r1_gs = r1_gs, r1_a = r1_a, r1_e = 1.6 * r1_gs * g$vpd / patm,
    r1_vcmax = r1$vcmax, r1_jmax = r1$jmax, r1_xi = r1$xi,
    r2acc_chi = f(acc, "chi"), r2acc_gs = f(acc, "gs"), r2acc_a = f(acc, "a"),
    r2acc_e = f(acc, "e"), r2acc_dpsi = f(acc, "dpsi"), r2acc_psi_l = f(acc, "psi_l"),
    r2acc_vcmax = f(acc, "vcmax"), r2acc_jmax = f(acc, "jmax"),
    r2ins_chi = f(ins, "chi"), r2ins_gs = f(ins, "gs"), r2ins_a = f(ins, "a"),
    r2ins_e = f(ins, "e"), r2ins_dpsi = f(ins, "dpsi"), r2ins_psi_l = f(ins, "psi_l"))
})

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, rows), out_path, row.names = FALSE)
cat("wrote", nrow(grid), "rows to", out_path, "\n")
