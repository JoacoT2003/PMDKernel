using HDF5
using CUDA
using Printf
include("B0.jl")
include("geometry.jl")
include("perturb.jl")
include("timer.jl")

"""
    simular_dataset(; name, geometrias, perturb_cfg, n_samples, seed_base,
                      out_dir, verbose=false) -> String

Pipeline live: para cada muestra `i = 1..N`,
    perturb(seed = seed_base + i) --> para cada `g ∈ geometrias`: B0(g.R, P, M).
Escribe a `out_dir/<name>.h5` con layout v2 (ver más abajo).

# Argumentos
- `name::String`                           — usado en el path de salida.
- `geometrias::Vector{Geometria}`          — lista de geometrías a evaluar (ver `geometry.jl`).
- `perturb_cfg::NamedTuple`                — kwargs pasados a `perturb(; ...)`.
- `n_samples::Int`                         — número de muestras (semillas seed_base+1..seed_base+N).
- `seed_base::Int`
- `out_dir::AbstractString`
- `verbose::Bool`                          — imprime una línea por iteración con tiempos.

# Layout HDF5 (v2)

```
<name>.h5
├── geometria/
│   ├── <g.name>/
│   │   ├── B          (N, n, 3)  Float32 mT     # vista Python tras transponer
│   │   ├── R          (n, 3)     Float32 mm
│   │   ├── meta/<k>   (varía)                   # entries del g.meta
│   │   └── attrs: { kind = String(g.kind) }
│   └── ...
└── attrs:
    ├── n_samples, seed_base
    ├── perturb_<k>          # cada campo del perturb_cfg como attr
    └── timing: t_<step>_mean_ms / t_<step>_std_ms / t_<step>_total_s   (vía StepTimer.summary_attrs)
```

Convención de transposición: HDF5.jl escribe column-major (Julia), por eso
declaramos `(3, n, N)` y guardamos `permutedims(B)` — Python ve `(N, n, 3)`.

Convención de dimensiones (ver README):
- N = nº de muestras (datasets)
- n = nº de puntos de la geometría correspondiente (índice variable por probe)

Datasets siempre en mT. R en mm xyz.
"""
function simular_dataset(; name::AbstractString,
        geometrias::Vector{Geometria},
        perturb_cfg::NamedTuple,
        n_samples::Int,
        seed_base::Int,
        out_dir::AbstractString,
        verbose::Bool = false)

    out_path = joinpath(out_dir, "$(name).h5")
    mkpath(out_dir)
    timer = StepTimer()

    # Validación temprana de nombres únicos de geometrías
    nombres = [String(g.name) for g in geometrias]
    @assert length(unique(nombres)) == length(nombres) "simular_dataset: nombres de geometrías duplicados: $(nombres)"

    h5open(out_path, "w") do f
        geom_root = create_group(f, "geometria")
        ds_B = Dict{Symbol, HDF5.Dataset}()

        for g in geometrias
            grp = create_group(geom_root, String(g.name))
            n   = size(g.R, 1)
            grp["R"] = permutedims(g.R)        # Julia (3,n) → Python lee (n,3)
            attrs(grp)["kind"] = String(g.kind)

            ds_B[g.name] = create_dataset(grp, "B", Float32,
                                           ((3, n, n_samples), (3, n, -1));
                                           chunk = (3, n, 1))

            meta_g = create_group(grp, "meta")
            for (k, v) in g.meta
                _escribir_meta!(meta_g, String(k), v)
            end
        end

        # Attrs globales del dataset
        attrs(f)["n_samples"] = n_samples
        attrs(f)["seed_base"] = seed_base
        for (k, v) in pairs(perturb_cfg)
            attr_key = "perturb_" * String(k)
            attrs(f)[attr_key] = v isa Symbol ? String(v) : v
        end

        # Loop principal
        for i in 1:n_samples
            tp = time_ns()
            P, M = perturb(; perturb_cfg..., seed = seed_base + i)
            record!(timer, :perturb, time_ns() - tp)

            for g in geometrias
                tg = time_ns()
                B = B0(g.R, P, M; timer = timer, prefix = Symbol(:B0_, g.name))
                record!(timer, Symbol(:B0_, g.name, :_total), time_ns() - tg)

                tw = time_ns()
                ds_B[g.name][:, :, i] = permutedims(B .* 1000f0)   # mT
                record!(timer, Symbol(:hdf5_write_, g.name), time_ns() - tw)
            end

            if verbose
                _print_iter_line(i, n_samples, timer, geometrias)
            end

            i % 50 == 0 && (GC.gc(); CUDA.reclaim())
        end

        for (k, v) in summary_attrs(timer)
            attrs(f)[k] = v
        end
    end

    println("\nsimular_dataset: $n_samples muestras → $out_path")
    report(timer)

    return out_path
end

"""
    simular_dataset_chunked(; name, geometrias, perturb_cfg, n_total, chunk_size,
                              seed_base, out_dir, verbose=false, skip_existing=true) -> Vector{String}

Genera `n_total` muestras repartidas en archivos de a `chunk_size` muestras.
Los seeds son contiguos entre archivos: bit-exacto equivalente a un único
`simular_dataset` con `n_samples = n_total`.

Cada archivo se nombra `<name>_part<NN>.h5` con padding según el total de chunks.
Los seeds del chunk `c` (1-based) son
`seed_base + (c-1)*chunk_size + 1 .. seed_base + c*chunk_size`.

Si `skip_existing=true` (default), saltea chunks cuyo `.h5` ya existe (resumible).
"""
function simular_dataset_chunked(; name::AbstractString,
        geometrias::Vector{Geometria},
        perturb_cfg::NamedTuple,
        n_total::Int,
        chunk_size::Int,
        seed_base::Int,
        out_dir::AbstractString,
        verbose::Bool       = false,
        skip_existing::Bool = true)

    n_total > 0    || error("n_total debe ser > 0")
    chunk_size > 0 || error("chunk_size debe ser > 0")

    n_chunks = cld(n_total, chunk_size)
    pad      = ndigits(n_chunks)
    paths    = String[]

    println("simular_dataset_chunked: $n_total muestras en $n_chunks archivos de hasta $chunk_size c/u")

    for c in 1:n_chunks
        chunk_n    = min(chunk_size, n_total - (c-1)*chunk_size)
        chunk_seed = seed_base + (c-1)*chunk_size
        chunk_name = "$(name)_part$(lpad(c, pad, '0'))"
        out_path   = joinpath(out_dir, "$(chunk_name).h5")

        println("\n>>> Chunk $c/$n_chunks: $(chunk_name) (seeds $(chunk_seed+1)..$(chunk_seed+chunk_n))")

        if skip_existing && isfile(out_path)
            println("    ✓ ya existe, salteando: $out_path")
            push!(paths, out_path)
            continue
        end

        push!(paths, simular_dataset(
            name        = chunk_name,
            geometrias  = geometrias,
            perturb_cfg = perturb_cfg,
            n_samples   = chunk_n,
            seed_base   = chunk_seed,
            out_dir     = out_dir,
            verbose     = verbose,
        ))
    end

    println("\nsimular_dataset_chunked: listo, $n_total muestras en $(length(paths)) archivos.")
    return paths
end

"""
    benchmark_dataset(; geometrias, perturb_cfg, n_samples, seed_base,
                        warmup=1) -> StepTimer

Igual que `simular_dataset` pero **sin escribir HDF5**: corre el loop completo
y devuelve el `StepTimer` con todas las mediciones. Útil para estimar tiempos
antes de un experimento real.

`warmup` corre N iteraciones descartadas antes de medir (incluye JIT compile +
primer kernel launch, que es notablemente más lento).
"""
function benchmark_dataset(; geometrias::Vector{Geometria},
        perturb_cfg::NamedTuple,
        n_samples::Int,
        seed_base::Int,
        warmup::Int = 1)

    for i in 1:warmup
        P, M = perturb(; perturb_cfg..., seed = seed_base + i)
        for g in geometrias
            B0(g.R, P, M)
        end
    end

    timer = StepTimer()
    for i in 1:n_samples
        tp = time_ns()
        P, M = perturb(; perturb_cfg..., seed = seed_base + i)
        record!(timer, :perturb, time_ns() - tp)

        for g in geometrias
            tg = time_ns()
            B0(g.R, P, M; timer = timer, prefix = Symbol(:B0_, g.name))
            record!(timer, Symbol(:B0_, g.name, :_total), time_ns() - tg)
        end

        i % 50 == 0 && (GC.gc(); CUDA.reclaim())
    end

    println("\nbenchmark_dataset: $n_samples muestras (warmup=$warmup)")
    report(timer; title = "Benchmark simular_dataset")
    return timer
end

# --- Helpers internos ------------------------------------------------------

# Escribe una entry de meta::Dict en HDF5: array → dataset, escalar/string → attr.
function _escribir_meta!(grp::HDF5.Group, key::String, val)
    if val isa AbstractArray
        grp[key] = collect(val)
    else
        attrs(grp)[key] = val isa Symbol ? String(val) : val
    end
end

function _print_iter_line(i, n_samples, timer, geometrias)
    @printf "[%4d/%d] perturb=%5.1f " i n_samples last_ms(timer, :perturb)
    for g in geometrias
        total = last_ms(timer, Symbol(:B0_, g.name, :_total))
        write_ = last_ms(timer, Symbol(:hdf5_write_, g.name))
        @printf " %s=%5.1f (write=%5.1f)" String(g.name) total write_
    end
    println(" ms")
end

# Ejemplo de uso (desde el REPL parado en PMDKernel/):
#
# include("julia/src/simulate.jl")
# geometrias = [
#     geom_grilla_xyz(:grid, -60:2:60, -60:2:60, -60:2:60),
#     geom_sensores_csv(:sens),
# ]
# config = (
#     name        = "grid60_step2_n100",
#     geometrias  = geometrias,
#     perturb_cfg = (kind=:both, sigma_deg=1f0,
#                    mu1=2.035f0, sigma1=0.1f0,
#                    mu2=8.48f0,  sigma2=0.85f0),
#     n_samples   = 100, seed_base = 0,
#     out_dir     = "data/datasets",
# )
# simular_dataset(; config..., verbose=true)
#
# # Estimar tiempo sin escribir disco:
# benchmark_dataset(; geometrias, perturb_cfg=config.perturb_cfg,
#                     n_samples=10, seed_base=0, warmup=2)
#
# # Splittear un experimento grande en chunks (resumible):
# simular_dataset_chunked(; config..., n_total=10_000, chunk_size=1_000)
