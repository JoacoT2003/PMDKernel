#!/bin/bash

# Nombre del trabajo
#SBATCH --job-name=Prueba7pmdkernel
# Archivo de salida
#SBATCH --output=salida.txt
# Partición
#SBATCH --partition=gpus
# Solicitud de gpus
#SBATCH --gres=gpu:1
# Reporte por correo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jats@ing.puc.cl

export JULIA_CUDA_RUNTIME_VERSION=12.0

echo "=== GPU CHECK ==="
nvidia-smi

echo "=== TEST CUDA ==="
~/.juliaup/bin/julia -e '
using CUDA
println("CUDA functional: ", CUDA.functional())
println(CUDA.device())
'

echo "=== RUNNING JULIA ==="

~/.juliaup/bin/julia B0.jl 1