using CUDA
using NPZ
using DelimitedFiles

# Uso standalone:
#   const BATCH_M = 64
#   include("kernel.jl"); include("utils/disco.jl")
#   simular_disco(0f0)

"""
    simular_disco(z_mm; xy_step, grid_range, thickness, threads, data_path, out_name)

Genera un dataset tipo "disco" para entrenamiento de ML. Un disco es una losa
delgada en Z centrada en `z_mm` (grosor configurable, default 1 mm). Evalúa el
campo B en:
  1. Una grilla reducida (paso `xy_step` mm en X e Y) dentro del resonador, en
     la capa `z = z_mm`.
  2. Los sensores del CSV cuya Z esté dentro del disco, i.e.
     |z_sensor − z_mm| ≤ thickness/2.

Ambos conjuntos se guardan en un único NPZ en `data/simulaciones/`.

Parámetros
----------
- `z_mm`       : posición Z (mm) en que se centra el disco
- `xy_step`    : paso de la grilla en X e Y (mm, default = 20)
- `grid_range` : rango de la grilla X-Y como tupla `(min, max)` en mm (default = (-60, 60))
- `thickness`  : grosor del disco en Z (mm, default = 1)
- `threads`    : hilos CUDA por bloque (default = 256)
- `data_path`  : ruta al NPZ de imanes (default = `data/B0.npz`)
- `out_name`   : nombre del archivo de salida sin extensión (default = `"disco_z<z_mm>"`)

Ejemplo
-------
    include("B0.jl")                     # carga kernel, BATCH_M, T
    include("utils/disco.jl")
    simular_disco(0f0; xy_step=20f0)     # disco en z = 0 mm (capa central)
    simular_disco(-64f0; xy_step=20f0)   # disco en z = -64 mm (profundidad 15.6 cm)
"""
function simular_disco(z_mm::Real;
        xy_step::Real = 20f0,
        grid_range::Tuple{<:Real,<:Real} = (-60f0, 60f0),
        thickness::Real = 1f0,
        threads::Int = 256,
        data_path = joinpath(@__DIR__, "..", "data", "B0.npz"),
        out_name::String = "disco_z$(z_mm)")

    z_disco = Float32(z_mm)
    half_t  = Float32(thickness) / 2f0

    # Sensores: lee CSV y convierte cilíndricas (cm, °, cm) → cartesianas (mm)
    sensores_path = joinpath(@__DIR__, "..", "data", "coordenadas_sensores.csv")
    coordenadas = readdlm(sensores_path, ',', Float32)
    n_tot = size(coordenadas, 1)

    sens_xyz = zeros(Float32, n_tot, 3)
    for i in 1:n_tot
        radio = coordenadas[i, 1]
        theta = coordenadas[i, 2]
        sens_xyz[i, 1] = radio * -sind(theta) * 10f0
        sens_xyz[i, 2] = radio *  cosd(theta) * 10f0
        sens_xyz[i, 3] = (coordenadas[i, 3] - 22f0) * 10f0
    end

    # Filtra sensores dentro del disco
    disco_mask = abs.(sens_xyz[:, 3] .- z_disco) .<= half_t
    sens_disco = sens_xyz[disco_mask, :]
    sens_indices = Int32.(findall(disco_mask))
    n_sens = size(sens_disco, 1)

    if n_sens == 0
        @warn "No hay sensores en |z − $(z_disco)| ≤ $(half_t) mm; el NPZ tendrá arrays vacíos para sensores."
    end

    # Grilla reducida X-Y, Z = z_disco (capa única, el grosor es implícito)
    gx = range(Float32(grid_range[1]); stop=Float32(grid_range[2]), step=Float32(xy_step))
    gy = gx
    Nx, Ny = length(gx), length(gy)
    n_grid = Nx * Ny

    R_grid = zeros(Float32, n_grid, 3)
    idx = 1
    for xi in gx, yi in gy
        R_grid[idx, 1] = xi
        R_grid[idx, 2] = yi
        R_grid[idx, 3] = z_disco
        idx += 1
    end

    # Une grid + sensores en un único tensor R para una sola invocación del kernel
    R_all = vcat(R_grid, sens_disco)
    n_all = size(R_all, 1)

    # Imanes (misma lógica que B0.jl y sensores.jl)
    data = npzread(data_path)
    P_cpu = Float32.(hcat(data["array1"], data["array3"]))

    default_path = joinpath(@__DIR__, "..", "data", "B0.npz")
    if data_path == default_path
        M_cpu = Float32.(hcat(data["array2"] .* 2.035f0, data["array4"] .* 3.051f0))
    else
        M_cpu = Float32.(hcat(data["array2"], data["array4"]))
    end
    m = size(P_cpu, 2)

    # GPU
    R = CuArray(R_all .* 0.001f0)
    P = CuArray(P_cpu .* 0.001f0)
    M = CuArray(M_cpu)
    B = CuArray(zeros(Float32, n_all, 3))

    blocks = cld(n_all, threads)
    shmem  = 6 * BATCH_M * sizeof(Float32)

    @cuda threads=threads blocks=blocks shmem=shmem kernel_fused_B!(R, P, M, B, n_all, m)

    B_all = Array(B) .* 1000f0   # mT

    B_grid = B_all[1:n_grid, :]
    B_sens = n_sens > 0 ? B_all[n_grid+1:end, :] : zeros(Float32, 0, 3)

    # Guarda NPZ
    out_dir = joinpath(@__DIR__, "..", "data", "simulaciones")
    mkpath(out_dir)
    out_path = joinpath(out_dir, "$(out_name).npz")
    npzwrite(out_path, Dict(
        "z_disco_mm"    => Float32[z_disco],
        "thickness_mm"  => Float32[Float32(thickness)],
        "grid_x_mm"     => reshape(R_grid[:, 1], Nx, Ny),
        "grid_y_mm"     => reshape(R_grid[:, 2], Nx, Ny),
        "grid_Bx_mT"    => reshape(B_grid[:, 1], Nx, Ny),
        "grid_By_mT"    => reshape(B_grid[:, 2], Nx, Ny),
        "grid_Bz_mT"    => reshape(B_grid[:, 3], Nx, Ny),
        "sensor_idx"    => sens_indices,
        "sensor_x_mm"   => sens_disco[:, 1],
        "sensor_y_mm"   => sens_disco[:, 2],
        "sensor_z_mm"   => sens_disco[:, 3],
        "sensor_Bx_mT"  => B_sens[:, 1],
        "sensor_By_mT"  => B_sens[:, 2],
        "sensor_Bz_mT"  => B_sens[:, 3],
    ))

    println("Disco guardado: ", out_path)
    println("  Z disco       : $(z_disco) mm  (grosor $(thickness) mm)")
    println("  Grilla        : $(Nx)×$(Ny) = $(n_grid) puntos  (paso $(xy_step) mm)")
    println("  Sensores disco: $(n_sens)")
    return out_path
end
