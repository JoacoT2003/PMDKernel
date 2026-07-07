using Statistics
using Printf

"""
    StepTimer()

Acumulador de tiempos por paso. Se usa para medir cuánto tarda cada
fase de un pipeline iterativo (ej. `simular_dataset`).

Internamente: `Dict{Symbol, Vector{Float64}}` con tiempos en segundos.
"""
mutable struct StepTimer
    times::Dict{Symbol, Vector{Float64}}
end
StepTimer() = StepTimer(Dict{Symbol, Vector{Float64}}())

"""
    record!(timer, step, dt_ns)

Acumula un tiempo (en nanosegundos) bajo la clave `step`.
Pensado para usarse junto con `time_ns()`:

    t0 = time_ns()
    foo()
    record!(timer, :foo, time_ns() - t0)
"""
@inline function record!(t::StepTimer, step::Symbol, dt_ns::Integer)
    push!(get!(t.times, step, Float64[]), Float64(dt_ns) * 1e-9)
    return nothing
end

"""
    last_ms(timer, step) -> Float64

Último tiempo registrado para `step`, en milisegundos. Útil para logs por iteración.
"""
@inline last_ms(t::StepTimer, step::Symbol) = t.times[step][end] * 1000

"""
    report(timer; title="Tiempos por paso") -> Float64

Imprime una tabla con n / mean / std / min / max / total por paso,
y devuelve el tiempo total acumulado (segundos).
"""
function report(t::StepTimer; title::AbstractString = "Tiempos por paso")
    println("\n=== $title ===")
    @printf "%-12s %6s %10s %10s %10s %10s %10s\n" "step" "n" "mean(ms)" "std(ms)" "min(ms)" "max(ms)" "total(s)"
    total = 0.0
    for (step, times) in sort(collect(t.times); by = first)
        ms = times .* 1000
        s  = sum(times)
        total += s
        sd = length(ms) > 1 ? std(ms) : 0.0
        @printf "%-12s %6d %10.2f %10.2f %10.2f %10.2f %10.3f\n" String(step) length(ms) mean(ms) sd minimum(ms) maximum(ms) s
    end
    @printf "TOTAL: %.3f s\n" total
    return total
end

"""
    summary_attrs(timer) -> Dict{String, Float64}

Devuelve un Dict con `t_<step>_mean_ms`, `t_<step>_std_ms`, `t_<step>_total_s`
por paso, listo para guardar como `attrs` de un archivo HDF5. Para pasos con
una sola medición, `std_ms = 0`.
"""
function summary_attrs(t::StepTimer)
    out = Dict{String, Float64}()
    for (step, times) in t.times
        s  = String(step)
        ms = times .* 1000
        out["t_$(s)_mean_ms"]  = mean(ms)
        out["t_$(s)_std_ms"]   = length(ms) > 1 ? std(ms) : 0.0
        out["t_$(s)_total_s"]  = sum(times)
    end
    return out
end
