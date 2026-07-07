using CUDA
include("kernel.jl")
isdefined(@__MODULE__, :StepTimer) || include("timer.jl")

"""
    B0(R, P, M; threads=256, timer=nothing, prefix=:B0) -> Matrix{Float32}

Función pura: calcula el campo dipolar en los puntos `R` dadas las posiciones
`P` y momentos `M` de los imanes.

Parámetros
----------
- `R` :: Matrix{Float32}, shape `[n, 3]`, posiciones en **mm** (xyz).
- `P` :: Matrix{Float32}, shape `[3, m]`, posiciones de imanes en **mm**.
- `M` :: Matrix{Float32}, shape `[3, m]`, momentos en **A·m²**.
- `threads` :: hilos CUDA por bloque (default = 256).
- `timer` :: si se pasa un `StepTimer`, registra el desglose interno con keys
  `<prefix>_to_gpu`, `<prefix>_kernel`, `<prefix>_to_cpu`, `<prefix>_free`.
  Cuando es `nothing` (default), no se hacen sincronizaciones extra.
- `prefix` :: prefijo para las keys del timer (ej. `:B0_grid`, `:B0_sens`).

Retorna `B::Matrix{Float32}` shape `[n, 3]` en **Tesla**.

No carga archivos, no escribe archivos, no abre GUI.
"""
function B0(R::AbstractMatrix, P::AbstractMatrix, M::AbstractMatrix;
        threads::Int = 256,
        timer = nothing,
        prefix::Symbol = :B0)
        
    n = size(R, 1)
    m = size(P, 2)

    # Host prep + H2D transfer + alloc B_gpu
    t = time_ns()
    R_gpu = CuArray(Float32.(R) .* 0.001f0)
    P_gpu = CuArray(Float32.(P) .* 0.001f0)
    M_gpu = CuArray(Float32.(M))
    B_gpu = CUDA.zeros(Float32, n, 3)
    if timer !== nothing
        CUDA.synchronize()
        record!(timer, Symbol(prefix, :_to_gpu), time_ns() - t)
    end

    # Kernel
    blocks = cld(n, threads)
    t = time_ns()
    @cuda threads=threads blocks=blocks _Bnu!(R_gpu, P_gpu, M_gpu, B_gpu, n, m)
    if timer !== nothing
        CUDA.synchronize()
        record!(timer, Symbol(prefix, :_kernel), time_ns() - t)
    end

    # D2H
    t = time_ns()
    B_cpu = Array(B_gpu)
    timer !== nothing && record!(timer, Symbol(prefix, :_to_cpu), time_ns() - t)

    # Free
    t = time_ns()
    CUDA.unsafe_free!(R_gpu)
    CUDA.unsafe_free!(P_gpu)
    CUDA.unsafe_free!(M_gpu)
    CUDA.unsafe_free!(B_gpu)
    timer !== nothing && record!(timer, Symbol(prefix, :_free), time_ns() - t)

    return B_cpu
end
