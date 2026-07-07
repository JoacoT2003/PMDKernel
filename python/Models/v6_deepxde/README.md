# v6_deepxde — PI-DeepONet con DeepXDE

Reemplazo de `v5_deepsets_pinn` (DeepSets + autograd manual, lento incluso en grilla 5×5×5).
Misma tarea: mapear mediciones interiores (`By` en `I` sensores) → campo exterior `(Bx,By,Bz)`
en la grilla, con restricciones de Maxwell `∇·B=0` y `∇×B=0`.

## Estado: IMPLEMENTADO (Lightning)


El cuello de v5 no eran los 125 puntos de grilla ni "código lento": era estructural —
DeepSets re-encodea los 180 sensores en **cada** punto query (O(J·I)) + 3 backward de
segundo orden. DeepONet corre el branch 1×/muestra y el trunk 1×/punto (O(I)+O(J)).

Harness: **PyTorch Lightning**, reusando la infra de Colab de v5 (`_DriveMirror`,
checkpoint local→Drive atómico, resume, EarlyStopping, CometLogger).

| Archivo | Rol |
|---|---|
| `model.py` | `LitDeepONetPINN` — DeepONetCartesianProd(3 out, independent) + física forward-mode. Doble factor de denorm `b_std[i]/coord_std[j]` (trunk normalizado). |
| `data.py` | Formato CartesianProd: branch `(B,I)` By normalizado per-sensor, target `(B,J,3)`, trunk = buffer compartido del modelo. Stats de train (b/branch/coord). |
| `metrics.py` | `evaluate` — RMSE/R² mT + residuos físicos vía jacobian forward-mode. |
| `train.py` | `train()` (copia de v5, tags v6). |
| `spike_plumbing.py` | Test de humo de la API DeepXDE (forward-mode + div/curl). |
| `../../train_v6.py` | Driver CLI. |

Uso (REPL parado en `PMDKernel/`):

```bash
python python/train_v6.py \
    --h5 data/datasets/v3_xyz100_step50_n100k.h5 \
    --epochs 200 --batch-size 64 \
    --lambda-div 1e-3 --lambda-curl 1e-3 \
    --run-tag v6_deeponet_n100k --no-progress
```

> Nota de entorno: DeepXDE (backend pytorch+cuda) fija el *default device* en cuda;
> `make_loader` pasa un `torch.Generator(device="cuda")` explícito al `RandomSampler`
> para que `shuffle=True` no choque.

---


### Stack fijado
- **DeepXDE 1.15.0**, backend **PyTorch** (`DDE_BACKEND=pytorch`). Integra con torch 2.5.1+cu121 ya instalado; no arrastra TensorFlow.
- **Autodiff: FORWARD mode** → `dde.config.set_default_autodiff("forward")`. **Obligatorio**, ver hallazgo abajo.
- Red: `dde.nn.DeepONetCartesianProd(num_outputs=3, multi_output_strategy="independent")`.
  - `layer_sizes_branch=[I, ..., P]` (entrada = `By` de los `I` sensores).
  - `layer_sizes_trunk=[3, ..., P]` (entrada = coords `(x,y,z)`).
  - Branch ⊗ trunk; salida `(N, J, 3)`.

### Formato de datos (CartesianProd — de acá viene el speedup vs v5)
- branch input `v`: `(N, I)` mediciones de sensores (compartido nada; una fila por muestra).
- trunk input `x`: `(J, 3)` coords de grilla **compartidas** entre las N muestras.
- target `y`: `(N, J, 3)` campo full.
- DeepONet evalúa branch 1×/muestra + trunk 1×/punto y los combina con einsum: **O(I)+O(J)** vs el DeepSets de v5 que corría un MLP sobre los `I` sensores **por cada** punto query (**O(J·I)**).

### Hallazgo crítico #1 — physics sobre output 3D necesita forward mode
`dde.grad.jacobian` en **reverse mode** (default) tira `NotImplementedError: Reverse-mode autodiff doesn't support 3D output` sobre el output `(N, J, 3)` de CartesianProd.
- **Forward mode SÍ funciona** (usa `torch.func.jvp`). Validado: `div` y `curl` salen `(N, J, 1)`, per-sample y per-punto.
- En forward mode `dde.grad.jacobian` requiere `ys = (tensor, callable)` (la red como función del trunk) y **devuelve una tupla `(tensor, callable)`** → hay que tomar `[0]`:
  ```python
  forward_call = lambda trunk: aux[0]((inputs[0], trunk))   # aux[0] = model.net en fwd mode
  dBx_dx = dde.grad.jacobian((outputs, forward_call), inputs[1], i=0, j=0)[0]  # ∂Bx/∂x  (N,J,1)
  ```

### Hallazgo crítico #2 — data+physics no tiene clase turnkey → custom `dde.data.Data`
- `TripleCartesianProd`: solo data loss (no physics; jacobian reverse falla).
- `PDEOperatorCartesianProd`: solo physics loss (+BC); **genera** branch inputs desde un `FunctionSpace` analítico y **no usa `train_y`** (no hay data loss). No encaja con nuestro dataset HDF5 fijo + etiquetado.
- **Solución: subclase custom de `dde.data.Data`** que combine ambas. Plumbing confirmado en `model.py:319-351` (backend pytorch):
  - `losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None)` recibe:
    - `targets` = campo `(N,J,3)` → `data_loss = loss_fn(targets, outputs)`.
    - `inputs = (branch, trunk)`; `aux = [model.net]` (solo en forward mode).
  - Devuelve `[data_loss, div_loss, curl_loss]`; `model.compile(loss_weights=[1.0, λ_div, λ_curl])` los pondera (equivale a los `lambda_*` de v5).
  - Replica el branch forward de `PDEOperatorCartesianProd._losses` (pde_operator.py:265-284) para armar `forward_call`.

### Física (portar de v5/model.py)
La red predice `B_norm=(B-b_mean)/b_std` con `b_std` **per-componente (3,)**. La cantidad física es
`∇·B = Σᵢ b_std[i]·∂B_norm[i]/∂xᵢ` (T/m = mT/mm, sin conversión). El factor `b_std[i]` es crítico
(Bx,Bz ~mT vs By ~50 mT). Curl análogo con los mismos pesos.

### Notas de entorno
- GPU local: RTX 2050 (~4 GB VRAM) → cuidar `N`/`P`/batch. Producción en cluster (ver memorias).
- `torch.func.jvp` requiere torch ≥ 2.1 (tenemos 2.5.1). OK.
