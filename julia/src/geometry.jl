using DelimitedFiles

"""
    Geometria

Una geometría con nombre = un conjunto de puntos donde evaluar B, más metadata
de cómo se construyó.

# Campos
- `name::Symbol`           identificador único dentro de un dataset (ej `:grid`, `:sens`).
- `R::Matrix{Float32}`     `[n, 3]` xyz mm — los puntos. Es lo que se le pasa a `B0(R, P, M)`.
- `kind::Symbol`           etiqueta del builder usado: `:cartesian`, `:cylindrical`,
                           `:sensor_csv`, `:custom`.
- `meta::Dict{String,Any}` bag flexible. Convención por kind (no enforced):
                             `:cartesian`   → `"x"`, `"y"`, `"z"` (Vector{Float32}).
                             `:cylindrical` → `"r"`, `"theta"`, `"z"`.
                             `:sensor_csv`  → `"path"` (String).
                             `:custom`      → libre.
"""
struct Geometria
    name::Symbol
    R::Matrix{Float32}
    kind::Symbol
    meta::Dict{String,Any}
end

# --- Conversores puros -----------------------------------------------------

# Convención cilíndrica del proyecto: x = -r·sin(theta), y = r·cos(theta)
# (preserva la orientación "+x crece derecha→izquierda" del resonador).
@inline function _cyl_to_xyz(r::Float32, theta_deg::Float32, z::Float32)
    return -r * sind(theta_deg), r * cosd(theta_deg), z
end

"""
    cyl_to_xyz(points::AbstractMatrix) -> Matrix{Float32}

Convierte una matriz `[n, 3]` con filas `(r_mm, θ_deg, z_mm)` a xyz mm.
"""
function cyl_to_xyz(points::AbstractMatrix)
    @assert size(points, 2) == 3 "cyl_to_xyz: points debe ser [n, 3]"
    n = size(points, 1)
    R = Matrix{Float32}(undef, n, 3)
    @inbounds for i in 1:n
        x, y, z = _cyl_to_xyz(Float32(points[i, 1]),
                              Float32(points[i, 2]),
                              Float32(points[i, 3]))
        R[i, 1] = x; R[i, 2] = y; R[i, 3] = z
    end
    return R
end

"""
    xyz_to_cyl(points::AbstractMatrix) -> Matrix{Float32}

Inversa de `cyl_to_xyz`. Devuelve `[n, 3]` con filas `(r_mm, θ_deg, z_mm)`.
"""
function xyz_to_cyl(points::AbstractMatrix)
    @assert size(points, 2) == 3 "xyz_to_cyl: points debe ser [n, 3]"
    n = size(points, 1)
    R = Matrix{Float32}(undef, n, 3)
    @inbounds for i in 1:n
        x = Float32(points[i, 1]); y = Float32(points[i, 2]); z = Float32(points[i, 3])
        r = sqrt(x*x + y*y)
        # Inversa de x = -r·sin(θ), y = r·cos(θ)  ⇒  θ = atan(-x, y)
        theta = atand(-x, y)
        R[i, 1] = r; R[i, 2] = theta; R[i, 3] = z
    end
    return R
end

# --- Builders --------------------------------------------------------------

"""
    geom_grilla_xyz(name, x, y, z) -> Geometria

Producto cartesiano de los rangos `x, y, z` (mm). Aplanamiento x-outer, z-inner
(loop `for xi in x, yi in y, zi in z`). `meta` carga `"x"`, `"y"`, `"z"` para
que un consumer pueda reshapear `(N, n, 3) → (N, Nx, Ny, Nz, 3)`.
"""
function geom_grilla_xyz(name::Symbol, x, y, z)
    Nx, Ny, Nz = length(x), length(y), length(z)
    R = Matrix{Float32}(undef, Nx*Ny*Nz, 3)
    idx = 1
    for xi in x, yi in y, zi in z
        R[idx, 1] = Float32(xi); R[idx, 2] = Float32(yi); R[idx, 3] = Float32(zi)
        idx += 1
    end
    meta = Dict{String,Any}(
        "x" => collect(Float32, x),
        "y" => collect(Float32, y),
        "z" => collect(Float32, z),
    )
    return Geometria(name, R, :cartesian, meta)
end

"""
    geom_grilla_cilindrica(name, r, theta, z) -> Geometria

Producto cilíndrico de `r` (mm), `theta` (grados), `z` (mm) → xyz mm.
Aplanamiento r-outer, z-inner. `meta` carga los rangos originales.
"""
function geom_grilla_cilindrica(name::Symbol, r, theta, z)
    Nr, Nt, Nz = length(r), length(theta), length(z)
    R = Matrix{Float32}(undef, Nr*Nt*Nz, 3)
    idx = 1
    for ri in r, ti in theta, zi in z
        x, y, zo = _cyl_to_xyz(Float32(ri), Float32(ti), Float32(zi))
        R[idx, 1] = x; R[idx, 2] = y; R[idx, 3] = zo
        idx += 1
    end
    meta = Dict{String,Any}(
        "r"     => collect(Float32, r),
        "theta" => collect(Float32, theta),
        "z"     => collect(Float32, z),
    )
    return Geometria(name, R, :cylindrical, meta)
end

"""
    geom_sensores_csv(name=:sens, csv_path=...) -> Geometria

Lee la geometría de los sensores del CSV (fuente de verdad del hardware).

Columnas del CSV: `(r_cm, θ_deg, z_cm)`. Pre-procesa unidades y offset:
r: cm→mm, z: (z_cm − 22) × 10 mm. Luego cilíndricas → xyz vía `cyl_to_xyz`.

`meta["path"]` lleva el path absoluto del CSV usado.
"""
function geom_sensores_csv(name::Symbol = :sens,
        csv_path::AbstractString = joinpath(@__DIR__, "..", "..", "data", "coordenadas_sensores.csv"))
    coords = readdlm(csv_path, ',', Float32)            # [n, 3] (r_cm, θ_deg, z_cm)
    points_mm = hcat(coords[:, 1] .* 10f0,
                     coords[:, 2],
                     (coords[:, 3] .- 22f0) .* 10f0)
    R = cyl_to_xyz(points_mm)
    meta = Dict{String,Any}("path" => String(csv_path))
    return Geometria(name, R, :sensor_csv, meta)
end

"""
    geom_puntos(name, R; kind=:custom, meta=Dict()) -> Geometria

Builder genérico desde una matriz `[n, 3]` xyz mm pre-armada. Útil para
geometrías que no encajan en grillas regulares (líneas, esferas, nubes).
"""
function geom_puntos(name::Symbol, R::AbstractMatrix;
        kind::Symbol = :custom,
        meta::AbstractDict = Dict{String,Any}())
    @assert size(R, 2) == 3 "geom_puntos: R debe ser [n, 3]"
    return Geometria(name, Matrix{Float32}(R), kind, Dict{String,Any}(meta))
end
