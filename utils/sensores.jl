using CUDA
using NPZ
using DelimitedFiles

# Uso standalone:
#   const BATCH_M = 64
#   include("kernel.jl"); include("utils/sensores.jl")
#   sensores(256)

"""
    sensores(threads; data_path)

Evalúa el campo B en las posiciones de los sensores (coordenadas_sensores.csv)
ejecutando el kernel GPU directamente sobre esos puntos.

Acepta cualquier archivo .npz con el formato de B0.npz (array1..array4), lo que
permite calcular el campo de configuraciones perturbadas generadas por perturb.jl.

Guarda `data/simulaciones/Bsensores.npz` con arrays: x_mm, y_mm, z_mm, Bx_mT, By_mT, Bz_mT.

Parámetros
----------
- `threads`   : hilos CUDA por bloque (default = 256)
- `data_path` : ruta al .npz de imanes (default = data/B0.npz)

Ejemplo
-------
    # Nominal
    sensores(256)

    # Con configuración perturbada
    sensores(256; data_path="data/rot_seed1.npz")

    # Loop sobre múltiples configuraciones
    for p in paths
        sensores(256; data_path=p)
    end
"""
function sensores(threads::Int = 256;
        data_path = joinpath(@__DIR__, "..", "data", "B0.npz"))

    #Imanes 
    data  = npzread(data_path)
    P_cpu = Float32.(hcat(data["array1"], data["array3"]))

    default_path = joinpath(@__DIR__, "..", "data", "B0.npz")
    if data_path == default_path
        M_cpu = Float32.(hcat(data["array2"] .* 2.035f0, data["array4"] .* 3.051f0))
    else
        M_cpu = Float32.(hcat(data["array2"], data["array4"]))
    end
    m = size(P_cpu, 2)

    # Sensores: cilíndricas (cm, °, cm) → cartesianas (mm) 
    sensores_path = joinpath(@__DIR__, "..", "data", "coordenadas_sensores.csv")
    coordenadas = readdlm(sensores_path, ',', Float32)
    n = size(coordenadas, 1)

    R_cpu = zeros(Float32, n, 3)
    for i in 1:n
        radio = coordenadas[i, 1]
        theta = coordenadas[i, 2]
        R_cpu[i, 1] = radio * -sind(theta) * 10f0    # x = r*sin(sigma), sigma=0° → +0, +Y (RAS) , sigma=90° → -X,0 (RAS)
        R_cpu[i, 2] = radio * cosd(theta) * 10f0    # y = r*cos(sigma)
        R_cpu[i, 3] = (coordenadas[i, 3] - 22f0) * 10f0
    end

    # GPU
    R = CuArray(R_cpu .* 0.001f0)
    P = CuArray(P_cpu .* 0.001f0)
    M = CuArray(M_cpu)
    B = CuArray(zeros(Float32, n, 3))

    blocks = cld(n, threads)
    shmem  = 6 * BATCH_M * sizeof(Float32)

    @cuda threads=threads blocks=blocks shmem=shmem kernel_fused_B!(R, P, M, B, n, m)

    B_res = Array(B')   # [3 × n]

    # Salida
    out_dir = joinpath(@__DIR__, "..", "data", "simulaciones")
    mkpath(out_dir)
    out_path = joinpath(out_dir, "Bsensores.npz")
    npzwrite(out_path, Dict(
        "x_mm"  => R_cpu[:, 1],
        "y_mm"  => R_cpu[:, 2],
        "z_mm" => R_cpu[:, 3],
        "Bx_mT" => B_res[1, :] .* 1000f0,
        "By_mT" => B_res[2, :] .* 1000f0,
        "Bz_mT" => B_res[3, :] .* 1000f0,
    ))

    println("Bsensores.npz guardado en: ", out_path)
end
