"""Evaluación de v1_fcnn — métricas físicas + supervisión sobre el volumen 3D.

Reconstruye `B_pred (J, 3)` flat → `(Nx, Ny, Nz, 3)` y aplica diferencias
finitas centradas para chequear coherencia física:

- **∇·B** (Gauss): debería ser 0 — sin monopolos magnéticos.
- **∇×B** (Ampère sin corrientes): 0 en aire/vacío.
- **∇²B** (Laplaciano vectorial): 0 — campo armónico en regiones libres
  de fuentes (consecuencia de las dos anteriores).
- **TV** (variación total): debería matchear TV(truth). El campo de un
  dipolo es C∞ smooth fuera del propio imán → la suma de dipolos hereda
  la suavidad. Además B0 en MRI tiene un requisito de homogeneidad por
  especificación. NO es regularización ad-hoc, es un test físico —
  pero el target es TV(truth), no cero.

Las diferencias finitas tienen error de discretización propio (~O(h²) con
central differences); aún el campo verdadero tendrá residuo no-nulo cuando
lo medimos por FD. Por eso evaluamos las cuatro métricas sobre PRED y TRUTH
y reportamos el cociente — un ratio ≈1 = la red matchea al campo verdadero.

**Sanity check supervisado**: MSE/MAE/RMSE sobre el flat. Sin esto, `B = 0`
constante satisface las tres primeras físicas trivialmente (TV no, pero
MSE da el control de fidelidad mejor).

Convención de unidades: B en mT, spacings en mm.
- ∇·B y ‖∇×B‖ en mT/mm
- ‖∇²B‖ en mT/mm²
- TV en mT (amplitud media del salto entre voxels adyacentes)

**Magnitudes de campos vectoriales** (curl, laplacian) usan L2:
`‖v‖ = √(vx² + vy² + vz²)`. Esto matchea la `_maxwell_loss` de v2_pinn
(`div² + ‖curl‖²`) y es lo natural para vectores.

**Bordes**: 1 voxel descartado por las central differences (1ª y 2ª derivada
operan sobre `[1:-1]` en cada eje), no NaN — directamente shape interior.
"""
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_full_sample


# --- Reshape ---------------------------------------------------------------

def reshape_to_grid(b_flat: torch.Tensor, grid_metadata: Dict) -> torch.Tensor:
    """`(batch, J, 3)` o `(J, 3)` → `(batch, Nx, Ny, Nz, 3)` o `(Nx, Ny, Nz, 3)`."""
    Nx, Ny, Nz = grid_metadata["Nx"], grid_metadata["Ny"], grid_metadata["Nz"]
    if b_flat.dim() == 2:
        return b_flat.reshape(Nx, Ny, Nz, 3)
    return b_flat.reshape(b_flat.shape[0], Nx, Ny, Nz, 3)


# --- Operadores diferenciales (interior, sin batch) -----------------------

def _divergence_single(b: torch.Tensor, dx: float, dy: float, dz: float) -> torch.Tensor:
    """∇·B en interior, central differences. Returns shape `(Nx-2, Ny-2, Nz-2)`."""
    dBx_dx = (b[2:,    1:-1, 1:-1, 0] - b[:-2,   1:-1, 1:-1, 0]) / (2.0 * dx)
    dBy_dy = (b[1:-1, 2:,    1:-1, 1] - b[1:-1, :-2,   1:-1, 1]) / (2.0 * dy)
    dBz_dz = (b[1:-1, 1:-1, 2:,    2] - b[1:-1, 1:-1, :-2,   2]) / (2.0 * dz)
    return dBx_dx + dBy_dy + dBz_dz


def _curl_single(b: torch.Tensor, dx: float, dy: float, dz: float) -> torch.Tensor:
    """∇×B en interior. Returns shape `(Nx-2, Ny-2, Nz-2, 3)`."""
    dBz_dy = (b[1:-1, 2:,    1:-1, 2] - b[1:-1, :-2,   1:-1, 2]) / (2.0 * dy)
    dBy_dz = (b[1:-1, 1:-1, 2:,    1] - b[1:-1, 1:-1, :-2,   1]) / (2.0 * dz)
    dBx_dz = (b[1:-1, 1:-1, 2:,    0] - b[1:-1, 1:-1, :-2,   0]) / (2.0 * dz)
    dBz_dx = (b[2:,    1:-1, 1:-1, 2] - b[:-2,   1:-1, 1:-1, 2]) / (2.0 * dx)
    dBy_dx = (b[2:,    1:-1, 1:-1, 1] - b[:-2,   1:-1, 1:-1, 1]) / (2.0 * dx)
    dBx_dy = (b[1:-1, 2:,    1:-1, 0] - b[1:-1, :-2,   1:-1, 0]) / (2.0 * dy)
    return torch.stack([
        dBz_dy - dBy_dz,        # (∇×B)_x
        dBx_dz - dBz_dx,        # (∇×B)_y
        dBy_dx - dBx_dy,        # (∇×B)_z
    ], dim=-1)


def _laplacian_single(b: torch.Tensor, dx: float, dy: float, dz: float) -> torch.Tensor:
    """∇²B per componente, stencil 7-puntos. Returns shape `(Nx-2, Ny-2, Nz-2, 3)`."""
    out = []
    for c in range(3):
        d2_dx2 = (b[2:,    1:-1, 1:-1, c] - 2.0*b[1:-1, 1:-1, 1:-1, c] + b[:-2,   1:-1, 1:-1, c]) / (dx*dx)
        d2_dy2 = (b[1:-1, 2:,    1:-1, c] - 2.0*b[1:-1, 1:-1, 1:-1, c] + b[1:-1, :-2,   1:-1, c]) / (dy*dy)
        d2_dz2 = (b[1:-1, 1:-1, 2:,    c] - 2.0*b[1:-1, 1:-1, 1:-1, c] + b[1:-1, 1:-1, :-2,   c]) / (dz*dz)
        out.append(d2_dx2 + d2_dy2 + d2_dz2)
    return torch.stack(out, dim=-1)


def _spacings(grid_x, grid_y, grid_z):
    return (float(grid_x[1] - grid_x[0]),
            float(grid_y[1] - grid_y[0]),
            float(grid_z[1] - grid_z[0]))


# --- Métricas (acepta volumen single o batched) ---------------------------

def divergence_metric(b_3d: torch.Tensor, grid_x, grid_y, grid_z) -> Dict[str, float]:
    """Mean, max, RMS de |∇·B| sobre el interior."""
    dx, dy, dz = _spacings(grid_x, grid_y, grid_z)
    if b_3d.dim() == 5:
        divs = torch.stack([_divergence_single(x, dx, dy, dz) for x in b_3d])
    else:
        divs = _divergence_single(b_3d, dx, dy, dz)
    return {
        "div_mean": float(divs.abs().mean()),
        "div_max":  float(divs.abs().max()),
        "div_rms":  float(torch.sqrt((divs ** 2).mean())),
    }


def curl_metric(b_3d: torch.Tensor, grid_x, grid_y, grid_z) -> Dict[str, float]:
    """Magnitud `‖∇×B‖` (L2) y per-componente sobre el interior."""
    dx, dy, dz = _spacings(grid_x, grid_y, grid_z)
    if b_3d.dim() == 5:
        curls = torch.stack([_curl_single(x, dx, dy, dz) for x in b_3d])
    else:
        curls = _curl_single(b_3d, dx, dy, dz)
    mag_sq = (curls ** 2).sum(dim=-1)
    mag    = torch.sqrt(mag_sq + 1e-30)
    return {
        "curl_mean":   float(mag.mean()),
        "curl_max":    float(mag.max()),
        "curl_rms":    float(torch.sqrt(mag_sq.mean())),
        "curl_x_mean": float(curls[..., 0].abs().mean()),
        "curl_y_mean": float(curls[..., 1].abs().mean()),
        "curl_z_mean": float(curls[..., 2].abs().mean()),
    }


def laplacian_metric(b_3d: torch.Tensor, grid_x, grid_y, grid_z) -> Dict[str, float]:
    """Magnitud `‖∇²B‖` (L2) y per-componente sobre el interior."""
    dx, dy, dz = _spacings(grid_x, grid_y, grid_z)
    if b_3d.dim() == 5:
        laps = torch.stack([_laplacian_single(x, dx, dy, dz) for x in b_3d])
    else:
        laps = _laplacian_single(b_3d, dx, dy, dz)
    mag_sq = (laps ** 2).sum(dim=-1)
    mag    = torch.sqrt(mag_sq + 1e-30)
    return {
        "lap_mean":   float(mag.mean()),
        "lap_max":    float(mag.max()),
        "lap_rms":    float(torch.sqrt(mag_sq.mean())),
        "lap_x_mean": float(laps[..., 0].abs().mean()),
        "lap_y_mean": float(laps[..., 1].abs().mean()),
        "lap_z_mean": float(laps[..., 2].abs().mean()),
    }


def total_variation_metric(b_3d: torch.Tensor) -> Dict[str, float]:
    """Variación total — promedio de `|B[i+1] - B[i]|` en cada eje.

    Test físico de suavidad — el target NO es cero (el campo varía: ~50mT en
    centro vs ~5mT en bordes), es el TV del ground truth. Comparar pred vs true.
    """
    if b_3d.dim() == 5:
        tvs = [_tv_single(x) for x in b_3d]
        return {
            "total_variation": float(np.mean([t["total_variation"] for t in tvs])),
            "gradient_x":      float(np.mean([t["gradient_x"]      for t in tvs])),
            "gradient_y":      float(np.mean([t["gradient_y"]      for t in tvs])),
            "gradient_z":      float(np.mean([t["gradient_z"]      for t in tvs])),
        }
    return _tv_single(b_3d)


def _tv_single(b: torch.Tensor) -> Dict[str, float]:
    grad_x = float((b[1:, :, :] - b[:-1, :, :]).abs().mean())
    grad_y = float((b[:, 1:, :] - b[:, :-1, :]).abs().mean())
    grad_z = float((b[:, :, 1:] - b[:, :, :-1]).abs().mean())
    return {
        "total_variation": grad_x + grad_y + grad_z,
        "gradient_x": grad_x,
        "gradient_y": grad_y,
        "gradient_z": grad_z,
    }


# --- Wrapper unificado por sample (entry point) --------------------------

def compute_spatial_metrics(b_pred_flat: torch.Tensor,
                            b_true_flat: torch.Tensor,
                            grid_metadata: Dict) -> Dict[str, Dict[str, float]]:
    """Físicas (div, curl, lap, TV) sobre PRED y TRUTH + supervision.

    Returns (jerárquico para que `report` lo lea limpio):
        {
          "physics_pred": {div_mean, div_max, div_rms,
                           curl_mean, curl_max, curl_rms, curl_{x,y,z}_mean,
                           lap_mean,  lap_max,  lap_rms,  lap_{x,y,z}_mean},
          "physics_true": {idem},
          "tv_pred":      {total_variation, gradient_{x,y,z}},
          "tv_true":      {idem},
          "supervised":   {mse, mae, rmse_mt},
        }
    """
    grid_x = grid_metadata["grid_x"]
    grid_y = grid_metadata["grid_y"]
    grid_z = grid_metadata["grid_z"]

    b_pred_3d = reshape_to_grid(b_pred_flat, grid_metadata)
    b_true_3d = reshape_to_grid(b_true_flat, grid_metadata)

    physics_pred = {}
    physics_pred.update(divergence_metric(b_pred_3d, grid_x, grid_y, grid_z))
    physics_pred.update(curl_metric      (b_pred_3d, grid_x, grid_y, grid_z))
    physics_pred.update(laplacian_metric (b_pred_3d, grid_x, grid_y, grid_z))

    physics_true = {}
    physics_true.update(divergence_metric(b_true_3d, grid_x, grid_y, grid_z))
    physics_true.update(curl_metric      (b_true_3d, grid_x, grid_y, grid_z))
    physics_true.update(laplacian_metric (b_true_3d, grid_x, grid_y, grid_z))

    tv_pred = total_variation_metric(b_pred_3d)
    tv_true = total_variation_metric(b_true_3d)

    mse = float(F.mse_loss(b_pred_flat, b_true_flat))
    mae = float(F.l1_loss (b_pred_flat, b_true_flat))
    rmse = mse ** 0.5

    return {
        "physics_pred": physics_pred,
        "physics_true": physics_true,
        "tv_pred":      tv_pred,
        "tv_true":      tv_true,
        "supervised":   {"mse": mse, "mae": mae, "rmse_mt": rmse},
    }


# --- Streaming evaluation (entry point del notebook) --------------------

def evaluate(lit_model, h5_path, sample_indices,
             x_scaler, b_scale, device, *,
             grid_metadata: Dict,
             rmse_per_component: bool = False) -> Dict[str, dict]:
    """Itera samples del split, predice todo el grid, agrega métricas.

    Para cada sample i:
      1. sensores normalizados → `lit_model` → `B_pred_norm`
      2. desnormalizar × `b_scale` → `B_pred` en mT
      3. `compute_spatial_metrics(B_pred, B_true, grid_metadata)`

    Promedia los dicts por clave sobre todos los samples del split.
    """
    lit_model.eval().to(device)
    accum = {"physics_pred": [], "physics_true": [],
             "tv_pred": [], "tv_true": [], "supervised": []}

    yt_chunks, yp_chunks = [], []   # solo si rmse_per_component
    with torch.no_grad():
        for sample_i in sample_indices:
            sensors_norm, b_true_mt = load_full_sample(
                h5_path, int(sample_i), x_scaler=x_scaler,
            )
            s = sensors_norm[None, :].to(device)
            b_pred_norm = lit_model(s).cpu()
            b_pred_mt   = b_pred_norm.reshape(-1, 3) * float(b_scale)
            b_true_mt   = b_true_mt.reshape(-1, 3)   # ya en mT

            m = compute_spatial_metrics(b_pred_mt, b_true_mt, grid_metadata)
            for k in accum:
                accum[k].append(m[k])

            if rmse_per_component:
                yt_chunks.append(b_true_mt.numpy())
                yp_chunks.append(b_pred_mt.numpy())

    out: Dict[str, dict] = {"n": len(accum["physics_pred"])}
    for k, dicts in accum.items():
        keys = dicts[0].keys()
        out[k] = {kk: float(np.mean([d[kk] for d in dicts])) for kk in keys}

    if rmse_per_component:
        yt = np.concatenate(yt_chunks, axis=0).astype(np.float32)
        yp = np.concatenate(yp_chunks, axis=0).astype(np.float32)
        out["rmse_per_component"] = {
            name: float(np.sqrt(np.mean((yp[:, c] - yt[:, c]) ** 2)))
            for c, name in enumerate(["Bx", "By", "Bz"])
        }
    return out


def report(name: str, m: Dict[str, dict]) -> None:
    """Imprime resumen jerárquico de un dict devuelto por `evaluate`.

    Para las cuatro físicas reporta `pred / true` ratio. Target en todos los
    casos es ratio ≈ 1× (la red matchea al campo verdadero en esa propiedad).
    """
    s  = m["supervised"]
    pp, pt = m["physics_pred"], m["physics_true"]
    tvp, tvt = m["tv_pred"], m["tv_true"]

    def _ratio(p, t):
        return p / t if t > 1e-12 else float("inf")

    print(f"[{name:>5}]  RMSE={s['rmse_mt']:.4e} mT   MAE={s['mae']:.4e} mT   (N={m['n']})")
    if "rmse_per_component" in m:
        per = m["rmse_per_component"]
        print(f"          RMSE Bx={per['Bx']:.4e}  By={per['By']:.4e}  Bz={per['Bz']:.4e}  mT")

    print(f"          ∇·B    pred={pp['div_rms']:.4e}  true={pt['div_rms']:.4e}  "
          f"ratio={_ratio(pp['div_rms'], pt['div_rms']):.2f}×   [mT/mm]")
    print(f"          ‖∇×B‖  pred={pp['curl_rms']:.4e}  true={pt['curl_rms']:.4e}  "
          f"ratio={_ratio(pp['curl_rms'], pt['curl_rms']):.2f}×   [mT/mm]")
    print(f"          ‖∇²B‖  pred={pp['lap_rms']:.4e}  true={pt['lap_rms']:.4e}  "
          f"ratio={_ratio(pp['lap_rms'], pt['lap_rms']):.2f}×   [mT/mm²]")
    print(f"          TV     pred={tvp['total_variation']:.4e}  true={tvt['total_variation']:.4e}  "
          f"ratio={_ratio(tvp['total_variation'], tvt['total_variation']):.2f}×   [mT]")
