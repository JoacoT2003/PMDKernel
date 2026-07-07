# PMDKernel

Kernel CUDA en Julia para calcular el campo magnético $B_0$ de imanes permanentes
modelados como dipolos. Aplicación principal: simulación del **OSI² OpenMRI**
(Open Source Imaging Initiative).

El repo contiene tres capas:

1. **Kernel CUDA + simulación** (`B0.jl`, `kernel.jl`) — Biot-Savart sobre dipolos
   en cualquier conjunto de puntos.
2. **Pipeline de datasets** (`simulate.jl` + `geometry.jl`) — perturbar imanes +
   correr kernel sobre N geometrías arbitrarias + escribir HDF5. Fuente de datos
   para entrenar redes.
3. **Modelos de ML en Python** (`Python/Models/<NOMBRE>/`) — paquetes con
   `data, model, train, metrics`. Notebooks homónimos en `Python/` los consumen
   como wrappers visuales para iterar.

---

## Convención de dimensiones

Todos los archivos del pipeline (Julia y Python) usan la misma notación.

### Pipeline / dataset / red

| Símbolo | Significado                                       | Índice          |
|---------|---------------------------------------------------|-----------------|
| `N`     | nº de muestras (datasets)                         | n ∈ {1..N}      |
| `I`     | nº de sensores                                    | i ∈ {1..I}      |
| `J`     | nº de puntos de grilla (= Nx·Ny·Nz aplanado)      | j ∈ {1..J}      |
| `Nx,Ny,Nz` | nº de puntos por eje (grilla cartesiana)       | —               |
| `Nr,Nθ,Nz` | nº de puntos por eje (grilla cilíndrica)       | —               |


### Kernel CUDA (genérico, no distingue sensores de grilla)

| Símbolo | Significado                  |
|---------|------------------------------|
| `n`     | puntos de evaluación (cualquier conjunto: `n = I` para sensores, `n = J` para grilla) |
| `m`     | dipolos / imanes             |

El kernel toma `(R[n,3], P[3,m], M[3,m])` y devuelve `B[n,3]`. No sabe si
estás evaluando en sensores o en grilla — eso lo decide el caller.

### Ejemplos de shapes

| Tensor                    | Shape                | Unidades |
|---------------------------|----------------------|----------|
| `geometria/grid/B` (HDF5) | `(N, J, 3)`          | mT       |
| `geometria/sens/B` (HDF5) | `(N, I, 3)`          | mT       |
| Vista 3D grilla           | `(N, Nx, Ny, Nz, 3)` | mT       |
---

## Estructura del repo

```
PMDKernel/
├── julia/                              Lado Julia (kernel + simulación)
│   ├── src/                            Biblioteca core
│   │   ├── kernel.jl                   CUDA kernel _Bnu! (Biot-Savart, 4× unrolled)
│   │   ├── B0.jl                       Pure B0(R,P,M)
│   │   ├── geometry.jl                 struct Geometria + builders
│   │   ├── perturb.jl                  perturb(...) → (P, M); incluye NOMINAL_SCALE_*
│   │   ├── simulate.jl                 Pipeline: perturb → B0(g.R) por geometría → HDF5
│   │   └── timer.jl                    StepTimer para benchmarks por paso
│   ├── viz/
│   │   └── gru.jl                      Slicer3D + mostrar_grilla (GLMakie)
│   ├── bench/
│   │   └── kernel_bench.jl             Benchmark sintético del kernel
│   └── scripts/
│       └── run_v1_dataset.jl           Driver de ejemplo
│
├── python/                             Lado Python (entrenamiento ML)
│   ├── notebooks/
│   │   ├── v1_fcnn.ipynb               Wrapper visual de Models/v1_fcnn
│   │   └── v2_pinn.ipynb               Wrapper visual de Models/v2_pinn
│   ├── Models/
│   │   ├── v1_fcnn/                    Paquete: data, model, train, metrics
│   │   │   └── logs/<run_tag>/         ckpt + aux.pt + csv (gitignored, autogenerado)
│   │   ├── v2_pinn/
│   │   │   └── logs/<run_tag>/
│   │   └── v2_1_pinn/
│   │       └── logs/<run_tag>/
│   ├── train_v2_1.py                   CLI entry para v2_1_pinn
│   └── comet_smoke_test.py
│
└── data/                               Inputs compartidos Julia/Python
    ├── B0.npz                          Geometría nominal de imanes
    ├── coordenadas_sensores.csv        Coordenadas (r_cm, θ°, z_cm) de los I sensores
    └── datasets/                       Salida HDF5 de simular_dataset 
```

---

## Pipeline de generación de datasets (Julia)

`simulate.jl` es el entry point. Una geometría = un conjunto de puntos donde
evaluar B (`struct Geometria(name, R, kind, meta)`, ver `geometry.jl`). Una
simulación = N muestras × K geometrías. Para cada muestra `n = 1..N`:

```
perturb(seed = seed_base + n)         →  P, M
para cada g ∈ geometrias:
    B0(g.R, P, M)                     →  B (n.points, 3)  Tesla
    HDF5.write geometria/<g.name>/B[n]
```

Las geometrías se construyen una sola vez al inicio (no cambian entre muestras).
Las salidas se almacenan en mT.

### Uso básico

Desde el REPL parado en `PMDKernel/`:

```julia
include("julia/src/simulate.jl")

geometrias = [
    geom_grilla_xyz(:grid, -100:10:100, -100:10:100, -100:10:100),
    geom_sensores_csv(:sens),
]

simular_dataset(
    name        = "mi_experimento",
    geometrias  = geometrias,
    perturb_cfg = (kind=:both, sigma_deg=1f0,
                   mu1=2.035f0, sigma1=0.2f0,
                   mu2=8.48f0,  sigma2=0.85f0),
    n_samples   = 500, seed_base = 0,
    out_dir     = "data/datasets",
    verbose     = true,
)
```

O ejecutando el driver: `julia --project=.. julia/scripts/run_v1_dataset.jl`.

Resultado: `data/datasets/mi_experimento.h5`.

### Estructura del HDF5 (layout v2)

```
mi_experimento.h5
├── geometria/
│   ├── grid/
│   │   ├── B          (N, J, 3)    Float32  mT
│   │   ├── R          (J, 3)       Float32  mm
│   │   ├── meta/
│   │   │   ├── x      (Nx,)        Float32  mm
│   │   │   ├── y      (Ny,)
│   │   │   └── z      (Nz,)
│   │   └── attrs: { kind = "cartesian" }
│   └── sens/
│       ├── B          (N, I, 3)    Float32  mT
│       ├── R          (I, 3)       Float32  mm
│       ├── meta/      (path del CSV como attr)
│       └── attrs: { kind = "sensor_csv" }
└── attrs:
    n_samples, seed_base, perturb_<k>, t_<step>_mean_ms / t_<step>_std_ms / t_<step>_total_s
```

**Aplanamiento del grid cartesiano:** orden x-outer, z-inner (loop Julia
`for xi in x, yi in y, zi in z`). En NumPy:
`f["geometria/grid/B"].reshape(N, Nx, Ny, Nz, 3)` con C-order. Verificable contra
`np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")` (lo hace el loader).

**Convención del meta::Dict** según `kind`:
- `:cartesian`   → `meta/x`, `meta/y`, `meta/z` (Vector{Float32}).
- `:cylindrical` → `meta/r`, `meta/theta`, `meta/z`.
- `:sensor_csv`  → `path` (attr String).
- `:custom`      → libre.

### Variantes

```julia
# Benchmark sin escribir HDF5 (estima cuánto tarda el experimento real):
benchmark_dataset(; geometrias, perturb_cfg, n_samples=10, seed_base=0, warmup=2)

# Particionar en múltiples archivos (resumible si se cae):
simular_dataset_chunked(
    name = "mi_experimento", geometrias = geometrias,
    perturb_cfg = ..., n_total = 10_000, chunk_size = 1_000,
    seed_base = 0, out_dir = "data/datasets",
)
# → mi_experimento_part01.h5 .. _part10.h5
```

Si un chunk se cae, volver a correr la misma llamada retoma desde donde quedó:
`skip_existing = true` (default) saltea cualquier `<name>_part<NN>.h5` ya escrito.
Para forzar la regeneración de un chunk, borrar el archivo correspondiente o
pasar `skip_existing = false`.

### Grillas no cartesianas / geometrías arbitrarias

```julia
# Cilíndrica
geom_grilla_cilindrica(:cyl, 0:5:160, 0:10:350, -60:5:60)

# Lista de puntos arbitraria (línea, esfera, nube custom...)
puntos = [...] :: Matrix{Float32}    # (n, 3) xyz mm
geom_puntos(:linea_x, puntos)
```

---

## Modelos en Python

Cada iteración de modelo vive en su propio paquete bajo `Python/Models/<NOMBRE>/`,
con la siguiente estructura mínima:

```
Models/<NOMBRE>/
├── __init__.py
├── data.py     # load_dataset(h5_path), split_and_normalize(...), make_loader(...)
├── model.py    # clase del modelo (subclass de nn.Module) + build_model()
├── train.py    # train(model, loader_tr, loader_va, ...) → dict(history, best_state, best_val)
└── metrics.py  # predict(...), metrics(y_true, y_pred), report(name, m)
```

El **notebook homónimo** (`Python/<NOMBRE>.ipynb`) sirve sólo para iterar
visualmente: importa los módulos y orquesta prints/plots. **No define lógica
entrenable.** Esto permite ejecutar lo mismo en headless sin Jupyter (cluster
HPC).

### Crear una iteración nueva

1. `mkdir Python/Models/v2_<algo>`, copiar `__init__.py` y los cuatro módulos
   desde `v1_fcnn` como base.
2. Ajustar `model.py` (arquitectura), `data.py` si cambia el preprocesamiento, y
   `train.py` si cambia el optimizador / scheduler / loss.
3. Crear `Python/v2_<algo>.ipynb` que importe del nuevo paquete.
4. Componentes compartidos entre iteraciones → refactorear a
   `Python/Models/_common/` antes de duplicar.

### Iteración actual

`v2_pinn` — Función de campo condicional `f(B_sens, x, y, z) → B(x,y,z)`. FCNN
de capas crecientes (~280k params) con activación SiLU. Loss combinada:

- **Datos**: MSE vs `B_grid` del simulador.
- **Maxwell**: `∇·B = 0` y `∇×B = 0`, vía `torch.autograd.grad` sobre las
  coordenadas (regiones libres de corrientes).
- **TV**: `mean(|∇B|)` para suavizado espacial.

Cada step de SGD ve **una sola configuración** con K puntos random — necesario
para que las losses físicas tengan sentido sobre un campo coherente.

Iteración anterior (`v1_fcnn`) abandonada: era un mapeo `B_sens (540) →
B_grid_aplanada (60.858)` puramente data-driven (MSE solo), con ~63M params.
La arquitectura de v2 es ~225× más chica y aprende un campo continuo.


---

## Layout de arrays en el kernel

| Array | Shape   | Layout       | Unidades | Rol                    |
|-------|---------|--------------|----------|------------------------|
| `R`   | `[n,3]` | row-major    | mm       | Puntos de evaluación   |
| `P`   | `[3,m]` | column-major | mm       | Posiciones de imanes   |
| `M`   | `[3,m]` | column-major | A·m²     | Momentos magnéticos    |
| `B`   | `[n,3]` | row-major    | T        | Campo de salida        |

La asimetría row/column-major es intencional. El kernel usa unrolling de 4 dipolos por iteración (sin shared-memory
tiling). `B0.jl` convierte mm → m internamente — la API de usuario es siempre mm.

`R` puede ser cualquier conjunto de puntos: `R = R_grid` (para evaluar el
volumen) o `R = R_sens` (para evaluar sólo en los sensores). El kernel no
distingue.

### Singularidad y clamp

Las contribuciones cerca de los dipolos pueden divergir (`1/r³`); el kernel
clampea la magnitud de salida a `B_MAX_T = 0.06 T` (60 mT) preservando dirección.
La guarda interna es `r² > 1e-8` m² (equivalente a `r > 0.1 mm`): cualquier pareja
punto-dipolo a menos de 0.1 mm aporta 0 a la suma. Es un cutoff físico, no sólo
un epsilon numérico — a escalas menores el modelo dipolar puntual deja de
describir al imán real, así que descartar esas contribuciones es más razonable
que extrapolar la singularidad.

---

## Convención del eje X

El proyecto usa `+x` creciendo de **derecha a izquierda** `+y` creciendo de **suelo a techo**. Plots XY deben invertir el eje X (`ax.invert_xaxis()`)
para respetar la orientación física.

---
