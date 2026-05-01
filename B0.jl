using NPZ
using Printf
using Distributions
using DelimitedFiles
using GPUArrays: @allowscalar
include("kernel.jl")
#include("utils/gru.jl")
include("utils/bench.jl")
include("utils/sensores.jl")

const BATCH_M =  64

"""
    B0(viz, threads; data_path, sensores)

Calcula el campo B0.

Acepta cualquier archivo .npz con el formato de B0.npz (array1..array4), lo que
permite calcular el campo de configuraciones perturbadas generadas por perturb.jl.

Parámetros
----------
- `viz`       : true → visualización 3D + guarda B_field.npz; false → benchmark
- `threads`   : hilos CUDA por bloque (default = 256)
- `data_path` : ruta al .npz de imanes (default = data/B0.npz)
- `sens`      : si true, evalúa campo en posiciones de sensores tras el cálculo principal
"""
function B0(viz::Bool, threads = 256;data_path = joinpath(@__DIR__, "data", "B0.npz"), sens = false)
    # Generación de la grilla de puntos R a evaluar
    gx = -170:10:170; gy = gx; gz = gx
    n = length(gx) * length(gy) * length(gz)

    Nx = length(gx)
    Ny = length(gy)
    Nz = length(gz)

    R_cpu = zeros(Float32, n, 3)
    idx = 1
    for xi in gx, yi in gy, zi in gz
        R_cpu[idx, 1] = xi
        R_cpu[idx, 2] = yi
        R_cpu[idx, 3] = zi
        idx += 1
    end

    # Carga de datos y preparación de matrices de posición y momento
    data   = npzread(data_path)
    M1_cpu = Float32.(hcat(data["array1"], data["array3"]))
    default_path = joinpath(@__DIR__, "data", "B0.npz")
    if data_path == default_path
        M2_cpu = Float32.(hcat(data["array2"] .* 2.035f0, data["array4"] .* 3.051f0))
    else
        M2_cpu = Float32.(hcat(data["array2"], data["array4"]))
    end

    n = size(R_cpu, 1)
    m = size(M2_cpu, 2)

    B_cpu = zeros(Float32, n, 3)

    R = CuArray(R_cpu .* 0.001f0)
    M = CuArray(M2_cpu)
    P = CuArray(M1_cpu .* 0.001f0)
    B = CuArray(B_cpu)

    blocks = cld(n, threads)
    shmem = 6 * BATCH_M * sizeof(Float32)

    if viz
        #include("utils/gru.jl")
        @cuda threads=threads blocks=blocks shmem=shmem kernel_fused_B!(R, P, M, B, n, m)
        B_res = Array(B')

        #XX = [xi for xi in gx, yi in gy, zi in gz]
        #mask = trues(size(XX))
        #By = zeros(size(XX))
        #@allowscalar begin
        #    By[mask] = B_res[2,:] .* -1000 # mT, el menos es unicamente para invertir los colores del heatmap
        #    fig = Figure(size=(600,600))
        #    saxi = Slicer3D(fig,By,zoom=3)
        #    display(fig)
        #end

        out_dir = joinpath(@__DIR__, "data", "simulaciones")
        mkpath(out_dir)
        npzwrite(joinpath(out_dir, "B_field.npz"), Dict(
            "Bx" => reshape(B_res[1,:] .* 1000f0, Nx, Ny, Nz),
            "By" => reshape(B_res[2,:] .* 1000f0, Nx, Ny, Nz),
            "Bz" => reshape(B_res[3,:] .* 1000f0, Nx, Ny, Nz),
            "gx" => collect(Float32, gx),
            "gy" => collect(Float32, gy),
            "gz" => collect(Float32, gz)
        ))
    else
        benchmark_kernel(R, P, M, B, n, m, threads)
    end

    if sens
        sensores(threads; data_path=data_path)
    end
end

viz = parse(Int, ARGS[1]) == 1
B0(viz, 256)