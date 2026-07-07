"""Evaluación de v2_1_pinn — predice todo el grid por sample y reporta RMSE/R² en mT.

La red es per-punto: para evaluar reconstruimos
el grid completo iterando sobre los J puntos de cada sample.

Desnormalización per-componente: la red devuelve `B_norm` y la salida física
es `B_real = B_norm · b_std + b_mean`, con `b_std`, `b_mean` shape (3,).
"""
import numpy as np
import torch
from torchmetrics.regression import MeanSquaredError, R2Score

from .data import load_full_grid_for_sample


def _to_t3(arr, device):
    """np/list shape (3,) → torch tensor float32 en `device`."""
    return torch.as_tensor(np.asarray(arr, dtype=np.float32).reshape(3),
                           dtype=torch.float32, device=device)


def _predict_full_grid_mT(lit_model, sensors_J, points_J_norm,
                          b_mean_t, b_std_t, device, *, chunk=8192):
    """Forward chunked sobre los J puntos de una sample. Devuelve (J, 3) numpy
    en mT (ya desnormalizado per-componente)."""
    lit_model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, sensors_J.shape[0], chunk):
            end = start + chunk
            s = sensors_J[start:end].to(device)
            p = points_J_norm[start:end].to(device)
            b_norm = lit_model(s, p)                    # (chunk, 3) normalizado
            b_mt = b_norm * b_std_t + b_mean_t          # (chunk, 3) mT
            outs.append(b_mt.cpu().numpy())
    return np.concatenate(outs, axis=0)


def evaluate(lit_model, h5_path, sample_indices,
             x_scaler, b_mean, b_std, pts_mean, pts_std, device, *,
             rmse_per_component=False, predict_chunk=8192):
    """Evalúa `lit_model` sobre `sample_indices` reconstruyendo todo el grid
    de cada sample. Devuelve dict con RMSE y R² globales en mT físicas.
    """
    lit_model.eval().to(device)
    b_mean_t = _to_t3(b_mean, device)
    b_std_t  = _to_t3(b_std,  device)

    yt_chunks, yp_chunks = [], []
    for sample_i in sample_indices:
        sensors_J, points_J_norm, b_true_mt = load_full_grid_for_sample(
            h5_path, int(sample_i),
            x_scaler=x_scaler, pts_mean=pts_mean, pts_std=pts_std,
        )
        b_pred_mt = _predict_full_grid_mT(
            lit_model, sensors_J, points_J_norm,
            b_mean_t, b_std_t, device, chunk=predict_chunk,
        )
        yt_chunks.append(b_true_mt.numpy())
        yp_chunks.append(b_pred_mt)
    yt = np.concatenate(yt_chunks, axis=0).astype(np.float32)
    yp = np.concatenate(yp_chunks, axis=0).astype(np.float32)

    yt_t = torch.from_numpy(yt).to(device)
    yp_t = torch.from_numpy(yp).to(device)

    rmse = MeanSquaredError(squared=False).to(device)(yp_t, yt_t).item()
    r2   = R2Score(multioutput="uniform_average").to(device)(yp_t, yt_t).item()

    out = {"rmse_mt": float(rmse), "r2": float(r2), "n": int(len(yt))}

    if rmse_per_component:
        per = {}
        for c, name in enumerate(["Bx", "By", "Bz"]):
            err = yp[:, c] - yt[:, c]
            per[name] = float(np.sqrt(np.mean(err ** 2)))
        out["rmse_per_component"] = per

    return out


def report(name, m):
    line = (f"[{name:>5}]  RMSE={m['rmse_mt']:.4e} mT   "
            f"R²={m['r2']:+.4f}   (N={m['n']})")
    print(line)
    if "rmse_per_component" in m:
        per = m["rmse_per_component"]
        print(f"          RMSE Bx={per['Bx']:.4e}  By={per['By']:.4e}  Bz={per['Bz']:.4e}  mT")
