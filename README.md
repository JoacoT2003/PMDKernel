# PMDKernel

This repository contains a custom CUDA Julia kernel optimized for calculating the magnetic field of large sets of permanent magnets, modeled as dipole moment vectors, in any point in space.

As an application, `B0.jl` provides a full simulation of the magnetic field $B_0$ of the Open Source Imaging Initiative's (OSI²) OpenMRI.

---

## Estructura

| Archivo                    | Rol                                                                 |
|----------------------------|---------------------------------------------------------------------|
| `B0.jl`                    | Entry point: grilla 3D completa del resonador + (opcional) sensores |
| `kernel.jl`                | Kernel CUDA (Biot-Savart, tiling en memoria compartida)             |
| `utils/sensores.jl`        | Evalúa el campo en todos los sensores del CSV                       |
| `utils/disco.jl`           | Genera datasets tipo "disco" para entrenamiento de ML               |
| `utils/perturb.jl`         | Genera configuraciones perturbadas (rotación, magnitud, ambas)      |
| `utils/bench.jl`           | Benchmark del kernel                                                |
| `utils/gru.jl`             | Visualización 3D (slicer ortogonal)                                 |
| `data/B0.npz`              | Geometría nominal de imanes                                         |
| `data/coordenadas_sensores.csv` | Coordenadas cilíndricas de sensores (radio cm, θ°, profundidad cm) |
| `data/simulaciones/`       | Carpeta de salida para todos los NPZ generados                      |

---

## Uso básico

```julia
include("B0.jl")

B0(true)                  # Grilla 121³ a 1 mm + visualización 3D
B0(true; sens=true)       # + campo en los sensores
B0(false)                 # Benchmark sin GUI
```

Salidas:
- `data/simulaciones/B_field.npz` — campo en la grilla completa
- `data/simulaciones/Bsensores.npz` — campo en todos los sensores

---

## Generación de datasets para ML (discos)

Para entrenar un modelo de ML sin tener que simular el volumen completo en una
tanda, dividimos el resonador en **discos**: losas delgadas (1 mm de grosor)
perpendiculares al eje Z. Cada disco produce un par:

- **entrada** → campo B en los sensores que caen dentro del disco
- **salida**  → campo B en una grilla X-Y reducida (paso 20 mm) a la misma Z

Esto reduce drásticamente el cómputo por muestra y permite barrer capas Z y
configuraciones perturbadas para armar el dataset.

### Función principal: `simular_disco(z_mm; ...)`

El primer argumento es la **coordenada Z** (en mm) donde se centra el disco.
La función:

1. Construye la grilla X-Y en esa Z (default: `-60:20:60` mm, 7×7 puntos).
2. Filtra los sensores cuya Z caiga dentro del disco (`|z_sensor − z_mm| ≤ 0.5`).
3. Corre el kernel sobre ambos conjuntos y guarda todo en un NPZ en
   `data/simulaciones/`.

### Ejemplo mínimo

```julia
include("B0.jl")             # carga kernel, BATCH_M, T
include("utils/disco.jl")

# Disco en z = 0 mm (capa central del resonador, depth = 22 cm en sensores)
simular_disco(0f0; xy_step=20f0)
# → data/simulaciones/disco_z0.0.npz
```

### Capas Z de los sensores

Con el CSV actual (`coordenadas_sensores.csv`), los sensores se agrupan en
5 profundidades (cm), convertidas a Z (mm) via `(depth − 22) × 10`:

| depth (cm) | z (mm) | Nº sensores por capa |
|------------|--------|----------------------|
| 9.2        | −128   | 36                   |
| 15.6       | −64    | 36                   |
| 22         |   0    | 36                   |
| 28.4       |  +64   | 36                   |
| 34.8       | +128   | 36                   |

Para barrer las 5 capas con la configuración nominal:

```julia
for z in (-128f0, -64f0, 0f0, 64f0, 128f0)
    simular_disco(z; xy_step=20f0, out_name="disco_nominal_z$(Int(z))")
end
```

### Estructura del NPZ generado

| Array            | Forma   | Unidades | Descripción                            |
|------------------|---------|----------|----------------------------------------|
| `z_disco_mm`     | (1,)    | mm       | Z del disco                            |
| `thickness_mm`   | (1,)    | mm       | Grosor del disco (default 1)           |
| `grid_x_mm`      | (Nx,Ny) | mm       | Coordenada X de cada nodo              |
| `grid_y_mm`      | (Nx,Ny) | mm       | Coordenada Y                           |
| `grid_Bx_mT`     | (Nx,Ny) | mT       | Componente X del campo en la grilla    |
| `grid_By_mT`     | (Nx,Ny) | mT       | Componente Y                           |
| `grid_Bz_mT`     | (Nx,Ny) | mT       | Componente Z                           |
| `sensor_idx`     | (k,)    | —        | Índices 1-based (fila del CSV) de los sensores del disco |
| `sensor_x_mm`    | (k,)    | mm       | Posición X del sensor                  |
| `sensor_y_mm`    | (k,)    | mm       | Posición Y                             |
| `sensor_z_mm`    | (k,)    | mm       | Posición Z                             |
| `sensor_Bx_mT`   | (k,)    | mT       | Campo X medido en el sensor            |
| `sensor_By_mT`   | (k,)    | mT       | Campo Y                                |
| `sensor_Bz_mT`   | (k,)    | mT       | Campo Z                                |

Con `xy_step=20` y `grid_range=(-60,60)` → `Nx=Ny=7` (49 puntos). Con el CSV
actual, cada capa contiene `k = 36` sensores (36 ángulos × 1 profundidad).

### Pipeline para un dataset completo

Combina `perturb_*` con `simular_disco` para generar muchas muestras con
configuraciones perturbadas de los imanes:

```julia
include("B0.jl")
include("utils/disco.jl")
include("utils/perturb.jl")

N_MUESTRAS = 100
capas_z    = (-128f0, -64f0, 0f0, 64f0, 128f0)

for seed in 1:N_MUESTRAS
    # 1. Genera una configuración perturbada (rotación de imanes σ = 1°)
    p = perturb_rotation(sigma_deg=1f0, seed=seed,
                         out_path="data/rot_seed$(seed).npz")

    # 2. Evalúa las 5 capas sobre esa configuración
    for z in capas_z
        simular_disco(z; xy_step=20f0,
                      data_path=p,
                      out_name="disco_seed$(seed)_z$(Int(z))")
    end
end
```

Esto genera `N_MUESTRAS × 5 = 500` archivos en `data/simulaciones/`, cada uno
un par (sensores → grilla) listo para entrenamiento. Otras variantes:
`perturb_magnitude`, `perturb_both` (ver `utils/perturb.jl`).

### Parámetros de `simular_disco`

| Parámetro    | Default         | Descripción                                   |
|--------------|-----------------|-----------------------------------------------|
| `z_mm`       | —               | Z del disco (mm, requerido)                   |
| `xy_step`    | `20`            | Paso de la grilla en X e Y (mm)               |
| `grid_range` | `(-60, 60)`     | Rango X-Y de la grilla (mm)                   |
| `thickness`  | `1`             | Grosor del disco en Z (mm)                    |
| `threads`    | `256`           | Hilos CUDA por bloque                         |
| `data_path`  | `data/B0.npz`   | NPZ de imanes (cámbialo para perturbados)     |
| `out_name`   | `"disco_z<z>"`  | Nombre del archivo de salida (sin extensión)  |

---

## Layout de arrays en el kernel

| Array | Shape  | Layout       | Unidades | Rol                    |
|-------|--------|--------------|----------|------------------------|
| R     | [n, 3] | row-major    | m        | Puntos de evaluación   |
| P     | [3, m] | column-major | m        | Posiciones de imanes   |
| M     | [3, m] | column-major | A·m²     | Momentos magnéticos    |
| B     | [n, 3] | row-major    | T        | Campo de salida        |

La asimetría row/column-major es intencional: garantiza accesos coalescentes en
GPU y tiling eficiente en memoria compartida (`BATCH_M = 64` dipolos por tile).

## Requisitos de scope

`BATCH_M` y `T` deben estar definidos en el scope **antes** de incluir
`kernel.jl` o cualquier utilidad. `B0.jl` los define al incluirse:

```julia
const BATCH_M = 64
const T       = Float32
```
