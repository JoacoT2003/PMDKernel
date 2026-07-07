include(joinpath(@__DIR__, "..", "src", "simulate.jl"))

geometrias = [
    geom_grilla_xyz(:grid, -100:10:100, -100:10:100, -225:10:225),
    geom_sensores_csv(:sens),
]

simular_dataset(
    name        = "v1_xy100_z225_step10_n5000",
    geometrias  = geometrias,
    perturb_cfg = (kind=:both, sigma_deg=1f0,
                   mu1=2.035f0, sigma1=0.1f0,
                   mu2=8.48f0,  sigma2=0.85f0),
    n_samples   = 5000,
    seed_base   = 0,
    out_dir     = joinpath(@__DIR__, "..", "..", "data", "datasets"),
    verbose     = true,
)
