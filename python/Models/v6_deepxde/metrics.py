"""Evaluación de v6_deepxde — predice (Bx,By,Bz) sobre la grilla y reporta
RMSE/R² mT + residuos físicos (|∇·B|, |∇×B|) vía jacobian forward-mode.

Formato CartesianProd: por muestra el branch es `(1, I)` y el output `(1, J, 3)`;
no hay chunking de puntos (la grilla entera entra de una, J~125).
"""
import deepxde as dde
import numpy as np
import torch
from torchmetrics.regression import MeanSquaredError, R2Score

from .data import load_full_grid_for_sample


def _branch_norm(lit_model, sensors_by, device):
    by = torch.as_tensor(sensors_by, dtype=torch.float32, device=device).reshape(1, -1)  # (1, I)
    return (by - lit_model.branch_mean) / lit_model.branch_std


def evaluate(lit_model, h5_path, sample_indices, device, *,
             rmse_per_component=True, compute_physics=True):
    lit_model.eval().to(device)

    yt_chunks, yp_chunks = [], []
    div_chunks, curl_chunks = [], []
    for sample_i in sample_indices:
        sensors_by, _R_grid, b_true_mt = load_full_grid_for_sample(h5_path, int(sample_i))
        branch_norm = _branch_norm(lit_model, sensors_by, device)            # (1, I)

        if compute_physics:
            with torch.enable_grad():
                b_norm, div, cx, cy, cz = lit_model._physics_residuals(branch_norm)
                b_mt = (b_norm * lit_model.b_std + lit_model.b_mean)          # (1, J, 3)
                curl_mag = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)           # (1, J)
            div_chunks.append(div.reshape(-1).detach().cpu().numpy())
            curl_chunks.append(curl_mag.reshape(-1).detach().cpu().numpy())
            b_pred_mt = b_mt.reshape(-1, 3).detach().cpu().numpy()           # (J, 3)
            dde.grad.clear()  # libera el cache de jvp; sin esto fuga 1 grafo/muestra -> OOM
        else:
            with torch.no_grad():
                b_pred_mt = lit_model.predict_mT(branch_norm).reshape(-1, 3).cpu().numpy()

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
        out["rmse_per_component"] = {
            name: float(np.sqrt(np.mean((yp[:, c] - yt[:, c]) ** 2)))
            for c, name in enumerate(["Bx", "By", "Bz"])
        }

    if compute_physics:
        div_all  = np.concatenate(div_chunks,  axis=0)
        curl_all = np.concatenate(curl_chunks, axis=0)
        out["physics"] = {
            "mean_abs_div_T_per_m":  float(np.abs(div_all).mean()),
            "max_abs_div_T_per_m":   float(np.abs(div_all).max()),
            "p95_abs_div_T_per_m":   float(np.percentile(np.abs(div_all), 95)),
            "mean_curl_mag_T_per_m": float(curl_all.mean()),
            "max_curl_mag_T_per_m":  float(curl_all.max()),
            "p95_curl_mag_T_per_m":  float(np.percentile(curl_all, 95)),
            "n_points": int(len(div_all)),
        }
    return out


def report(name, m):
    print(f"[{name:>5}]  RMSE={m['rmse_mt']:.4e} mT   R²={m['r2']:+.4f}   (N={m['n']})")
    if "rmse_per_component" in m:
        per = m["rmse_per_component"]
        print(f"          RMSE Bx={per['Bx']:.4e}  By={per['By']:.4e}  Bz={per['Bz']:.4e}  mT")
    if "physics" in m:
        ph = m["physics"]
        print(f"          |div B|  mean={ph['mean_abs_div_T_per_m']:.3e}  p95={ph['p95_abs_div_T_per_m']:.3e}  max={ph['max_abs_div_T_per_m']:.3e}  T/m")
        print(f"          |curl B| mean={ph['mean_curl_mag_T_per_m']:.3e}  p95={ph['p95_curl_mag_T_per_m']:.3e}  max={ph['max_curl_mag_T_per_m']:.3e}  T/m")
