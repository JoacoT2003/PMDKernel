using NPZ
using Distributions
using Random

# Factores nominales: convierten los vectores raw de B0.npz 
const NOMINAL_SCALE_1 = 2.035f0   # para array2 (1936 imanes)
const NOMINAL_SCALE_2 = 3.051f0   # para array4 (384 imanes)

####FUNCIONES AUXILIARES

# Aplica rotación en el plano XY a un vector [x, y, z]. Devuelve x', y'.
function rotate_xy(x::Float32, y::Float32, delta_rad::Float32)
    c = cos(delta_rad)
    s = sin(delta_rad)
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

# Muestrea un factor de escala por imán de Normal(mu, sigma)
function sample_scales(rng, mu::Float32, sigma::Float32, m::Int)::Vector{Float32}
    scales = Float32.(rand(rng, Normal(mu, sigma), m))
    @assert all(scales .> 0f0) "Escala negativa muestreada — verifica mu y sigma"
    return scales
end

### Nota transicion (Python->Julia): En Julia se definen los docstring fuera de la función, 
### sin linea en blanco entre el final del docstring y la función.
### ; se usa para separar los argumentos posicionales de los keyword arguments.
### $(i) inserta i en el string, equivalente a f"{i}" en python.

"""
    perturb_rotation(; sigma_deg, seed, out_path) -> String

Perturba la orientación de los imanes (rotación aleatoria en XY por imán,
delta ~ Normal(0, sigma_deg) grados). Los momentos se guardan escalados.
Esta función esta pensada en ser usada solo para generar rotaciones de los imanes.

Parámetros
----------
- `sigma_deg` : desviación estándar del error angular (grados, default = 1.0)
- `seed`      : semilla RNG para reproducibilidad (default = nada → aleatorio)
- `out_path`  : ruta del .npz de salida

Ejemplo
-------
    include("utils/perturb.jl")
    paths = [perturb_rotation(sigma_deg=1f0, seed=i, out_path="data/rot_seed\$(i).npz") for i in 1:100]
    for p in paths
        sensores(256; data_path=p)
    end
"""
function perturb_rotation(;
        sigma_deg::Float32 = 1f0,
        seed               = nothing,
        out_path::String   = joinpath(@__DIR__, "..", "data", "B0_rotation_perturbed.npz"))

    rng  = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    dist = Normal(0, sigma_deg)

    data = npzread(joinpath(@__DIR__, "..", "data", "B0.npz"))
    pos1 = Float32.(data["array1"])
    pos2 = Float32.(data["array3"])
    mom1 = Float32.(data["array2"]) .* NOMINAL_SCALE_1
    mom2 = Float32.(data["array4"]) .* NOMINAL_SCALE_2

    apply_rotation!(mom1, rng, dist)
    apply_rotation!(mom2, rng, dist)

    npzwrite(out_path, Dict("array1"=>pos1, "array2"=>mom1, "array3"=>pos2, "array4"=>mom2))
    println("NPZ Perturbado con rotaciones guardado: ", out_path)
    return out_path
end



"""
    perturb_magnitude(; mu1, sigma1, mu2, sigma2, seed, out_path) -> String

Perturba la magnitud de los imanes: cada imán recibe un factor independiente
scale_j ~ Normal(mu_k, sigma_k). Los momentos se guardan escalados.

Parámetros
----------
- `mu1`, `sigma1`  : media y sigma para array2 (tipo 1; mu1 ≈ 2.035) --> MODIFICAR
- `mu2`, `sigma2`  : media y sigma para array4 (tipo 2; mu2 ≈ 3.051) --> MODIFICAR
- `seed`           : semilla RNG
- `out_path`       : ruta del .npz de salida

Ejemplo
-------
    paths = [perturb_magnitude(mu1=2.035f0, sigma1=0.05f0,
                               mu2=3.051f0, sigma2=0.08f0,
                               seed=i, out_path="data/mag_seed\$(i).npz")
             for i in 1:100]
    for p in paths
        sensores(256; data_path=p)
    end
"""
function perturb_magnitude(;
        mu1::Float32,
        sigma1::Float32,
        mu2::Float32,
        sigma2::Float32,
        seed = nothing,
        out_path::String = joinpath(@__DIR__, "..", "data", "B0_magnitude_perturbed.npz"))

    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    data = npzread(joinpath(@__DIR__, "..", "data", "B0.npz"))
    pos1 = Float32.(data["array1"])
    pos2 = Float32.(data["array3"])
    mom1 = Float32.(data["array2"])
    mom2 = Float32.(data["array4"])

    scales1 = sample_scales(rng, mu1, sigma1, size(mom1, 2))
    scales2 = sample_scales(rng, mu2, sigma2, size(mom2, 2))

    mom1 .*= scales1'
    mom2 .*= scales2'

    npzwrite(out_path, Dict("array1"=>pos1, "array2"=>mom1, "array3"=>pos2, "array4"=>mom2))
    println("NPZ Perturbado con magnitudes guardado: ", out_path)
    return out_path
end



"""
    perturb_both(; sigma_deg, mu1, sigma1, mu2, sigma2, seed, out_path) -> String

Perturba simultáneamente magnitud y orientación de los imanes.
Primero aplica la perturbación de magnitud, luego la rotación.
Los momentos se guardan escalados.

Parámetros
----------
- `sigma_deg`      : desviación estándar del error angular (grados, default = 1.0)
- `mu1`, `sigma1`  : media y sigma para array2 (tipo 1; mu1 ≈ 2.035) --> MODIFICAR
- `mu2`, `sigma2`  : media y sigma para array4 (tipo 2; mu2 ≈ 3.051) --> MODIFICAR
- `seed`           : semilla RNG
- `out_path`       : ruta del .npz de salida

Ejemplo
-------
    perturb_both(sigma_deg=1f0, mu1=2.035f0, sigma1=0.05f0, mu2=3.051f0, sigma2=0.08f0, seed=42)
"""
function perturb_both(;
        sigma_deg::Float32 = 1f0,
        mu1::Float32,
        sigma1::Float32,
        mu2::Float32,
        sigma2::Float32,
        seed             = nothing,
        out_path::String = joinpath(@__DIR__, "..", "data", "B0_perturbed.npz"))

    rng  = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    dist = Normal(0, sigma_deg)

    data = npzread(joinpath(@__DIR__, "..", "data", "B0.npz"))
    pos1 = Float32.(data["array1"])
    pos2 = Float32.(data["array3"])
    mom1 = Float32.(data["array2"])
    mom2 = Float32.(data["array4"])

    scales1 = sample_scales(rng, mu1, sigma1, size(mom1, 2))
    scales2 = sample_scales(rng, mu2, sigma2, size(mom2, 2))

    mom1 .*= scales1'
    mom2 .*= scales2'

    apply_rotation!(mom1, rng, dist)
    apply_rotation!(mom2, rng, dist)

    npzwrite(out_path, Dict("array1"=>pos1, "array2"=>mom1, "array3"=>pos2, "array4"=>mom2))
    println("NPZ Perturbado con magnitudes y rotaciones guardado:: ", out_path)
    return out_path
end