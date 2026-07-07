using BenchmarkTools
using Random
include("../src/kernel.jl")
Random.seed!(1234)
const T = Float32

function setup_benchmark(n_points=1_000_000, n_dipoles=50_000)
    # Generate test data on host
    R_h = rand(T, n_points, 3) .* T(100)   # Evaluation points
    P_h = rand(T, 3, n_dipoles) .* T(50)   # Dipole positions
    M_h = rand(T, 3, n_dipoles)            # Dipole moments
    
    # Transfer to GPU (use column-major for better coalescing)
    R_d = CuArray(R_h)
    P_d = CuArray(P_h)
    M_d = CuArray(M_h)
    B_d = CUDA.zeros(T, 3, n_points)
    
    return R_d, P_d, M_d, B_d, n_points, n_dipoles
end

function benchmark_kernel(R_d, P_d, M_d, B_d, n, m, threads)
    blocks = cld(n, threads)

    println("=== Kernel Configuration ===")
    println("Threads/block: $threads")
    println("Blocks: $blocks")
    println("Total threads: $(threads * blocks)")
    println("Dipoles: $m")

    # Warmup
    @cuda threads=threads blocks=blocks _Bnu!(R_d, P_d, M_d, B_d, n, m)
    synchronize()

    # Timing con CUDA events
    println("\n=== Timing (CUDA Events) ===")
    start_event = CuEvent()
    end_event = CuEvent()

    CUDA.record(start_event)
    @cuda threads=threads blocks=blocks _Bnu!(R_d, P_d, M_d, B_d, n, m)
    CUDA.record(end_event)
    synchronize(end_event)
    gpu_time = elapsed(start_event, end_event)

    println("Kernel time: $(gpu_time*1e3) ms")
    println("Throughput: $(n*m / gpu_time / 1e9) G interactions/sec")
    return
end
