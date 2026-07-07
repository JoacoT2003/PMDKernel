using NPZ
using Distributions
using Random

const NOMINAL_SCALE_1 = 2.035f0   # array2 (1936 imanes)
const NOMINAL_SCALE_2 = 8.48f0    # array4 (384 imanes)

# --- Helpers internos ---

function rotate_xy(x::Float32, y::Float32, delta_rad::Float32)
    c = cos(delta_rad); s = sin(delta_rad)
    return x*c - y*s, x*s + y*c
end

# Rota en el plano XY los vectores de momento de cada imán.
# delta ~ Normal(0, sigma_deg) grados, independiente por columna.
function apply_rotation!(mom::Matrix{Float32}, rng, dist)
    for j in axes(mom, 2)
        delta = Float32(deg2rad(rand(rng, dist)))
        mom[1,j], mom[2,j] = rotate_xy(mom[1,j], mom[2,j], delta)
    end
end

# Muestrea un factor de escala por imán de Normal(mu, sigma). Asume mu > 0.
function sample_scales(rng, mu::Float32, sigma::Float32, m::Int)::Vector{Float32}
    scales = Float32.(rand(rng, Normal(mu, sigma), m))
    @assert all(scales .> 0f0) "Escala negativa muestreada — verifica mu y sigma"
    return scales
end

"""
    perturb(; kind, sigma_deg, mu1, sigma1, mu2, sigma2, seed) -> (P, M)

Genera una configuración perturbada de imanes y la devuelve **en memoria**:
- `P` :: Matrix{Float32} `[3, m]` posiciones (mm).
- `M` :: Matrix{Float32} `[3, m]` momentos ya escalados (A·m²).

Parámetros
----------
- `kind`       : `:rotation` | `:magnitude` | `:both` (default)
- `sigma_deg`  : desviación angular (grados) para la rotación XY (default = 1.0)
- `mu1,sigma1` : media y sigma para la magnitud del array2 (default = 2.035 ± 0.1)
- `mu2,sigma2` : media y sigma para la magnitud del array4 (default = 8.48 ± 0.85)
- `seed`       : semilla RNG (default = nothing → aleatorio)

No escribe archivos. El loop sobre N seeds vive en `simular_dataset`.
"""
function perturb(; kind::Symbol     = :both,
        sigma_deg::Float32 = 1f0,
        mu1::Float32 = NOMINAL_SCALE_1, sigma1::Float32 = 0.1f0,
        mu2::Float32 = NOMINAL_SCALE_2, sigma2::Float32 = 0.85f0,
        seed              = nothing)

    kind in (:rotation, :magnitude, :both) ||
        error("kind debe ser :rotation, :magnitude o :both; recibido $(kind)")

    rng  = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    data = npzread(joinpath(@__DIR__, "..", "..", "data", "B0.npz"))
    pos1 = Float32.(data["array1"]); pos2 = Float32.(data["array3"])
    mom1 = Float32.(data["array2"]) .* NOMINAL_SCALE_1
    mom2 = Float32.(data["array4"]) .* NOMINAL_SCALE_2

    if kind in (:magnitude, :both)
        # Los momentos ya están multiplicados por NOMINAL_SCALE_*, por eso el
        # factor que aplicamos debe centrarse en mu/NOMINAL para que el resultado
        # neto tenga media mu, no mu * NOMINAL.
        mom1 .*= sample_scales(rng, mu1 / NOMINAL_SCALE_1,
                                sigma1 / NOMINAL_SCALE_1, size(mom1, 2))'
        mom2 .*= sample_scales(rng, mu2 / NOMINAL_SCALE_2,
                                sigma2 / NOMINAL_SCALE_2, size(mom2, 2))'
    end
    if kind in (:rotation, :both)
        dist = Normal(0, sigma_deg)
        apply_rotation!(mom1, rng, dist)
        apply_rotation!(mom2, rng, dist)
    end

    P = hcat(pos1, pos2)
    M = hcat(mom1, mom2)
    return P, M
end
