# Entrenar en el cluster (UC ING)

Guía mínima para correr un modelo de `python/Models/<NOMBRE>/` en el cluster
`cluster.ing.uc.cl` (SLURM, partición `gpus`). Ejemplo: `v5_deepsets_pinn`.

---

## 1. Cuenta de Comet ML

Comet loguea las métricas de cada run (loss, RMSE, plots, ckpts) en un dashboard
web. Sin él, los runs corren igual pero pierdes la traza.

1. Crear cuenta en https://www.comet.com (Sign Up, gratis con mail UC).
2. Una vez dentro: avatar arriba a la derecha → **Account Settings** → pestaña
   **API Keys** → copiar la key.
3. Anotarla. La vas a pegar en el cluster en el paso 3.

Workspace y project del proyecto:

- workspace: `<NOMBRE>`
- project:   `pmdkernel`

Si querés guardar bajo tu propio workspace, cambialo en `submit.sh` con
`--comet-project <tu-project>`; el trainer usa el workspace asociado a la API
key.

---

## 3. Configurar la API key de Comet en el cluster

Comet lee credenciales desde `~/.comet.config`. En el cluster:

```bash
cat > ~/.comet.config <<'EOF'
[comet]
api_key = TU_API_KEY_AQUI
workspace = <NOMBRE>
project_name = pmdkernel
EOF
chmod 600 ~/.comet.config
```

La verificación de que Comet lee bien la key la hacemos en el paso 4, después
de instalar la librería.

---

## 4. Instalar conda + dependencias


Crear el env del proyecto:

```bash
conda create -n ipre python=3.11 -y
conda activate ipre

pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install pytorch-lightning comet-ml torchmetrics h5py numpy pandas scikit-learn matplotlib
```

Confirmar instalación y que la API-KEY funcione (`get_workspaces` pega contra
el servidor de Comet; si la key está mal, tira `InvalidAPIKey`):

```bash

python -c "import comet_ml; print(comet_ml.API().get_workspaces())"
```

> debe imprimir una lista con tu workspace (ej.
> `['sebasti-n-vallejos']`). Si imprime `[]` o tira error de auth, revisar
> `~/.comet.config`.

---

## 5. Clonar el repo en el cluster

```bash
mkdir -p ~/IPRE && cd ~/IPRE
git clone <url-del-repo> PMDKernel
cd PMDKernel
```

---

## 7. submit.sh

Crear `submit.sh` en la raíz del repo (`~/IPRE/PMDKernel/submit.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=v5_pinn
#SBATCH --output=logs/v5_pinn_%j.out
#SBATCH --error=logs/v5_pinn_%j.err
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<tu-mail>@uc.cl

module load cuda/12.9

source $HOME/.bashrc
conda activate ipre

cd ~/IPRE/PMDKernel
mkdir -p logs

echo "=== ENV ==="
python -c "import torch; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}')"
nvidia-smi

echo "=== TRAIN ==="
python python/train_v5.py \
    --h5 data/datasets/v2_xyz100_step50_n5000.h5 \
    --run-tag v5_pinn_n5000 \
    --epochs 100 \
    --patience 10 \
    --batch-size 32 \
    --no-progress \
    ```

Flags clave:

- `--batch-size 32` — un step procesa 32 configs × `points-per-sample` puntos.
- `--no-progress` — apaga el progress bar de Lightning. En logs SLURM el bar
  emite una línea por refresh y satura el `.out`.
- `--num-workers 4 --pin-memory` — paralelismo del DataLoader, sólo útil si

---

## 9. Bajar checkpoints

Los pesos quedan en `python/Models/v5_deepsets_pinn/logs/<run_tag>/`. Para
bajarlos:

```powershell
scp -r <tu-usuario>@cluster.ing.uc.cl:~/IPRE/PMDKernel/python/Models/v5_deepsets_pinn/logs/v5_pinn_n100k `
    python/Models/v5_deepsets_pinn/logs/
```
