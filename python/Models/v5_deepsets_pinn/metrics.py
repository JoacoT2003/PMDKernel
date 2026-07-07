"""Evaluación de v5_deepsets_pinn — predice (Bx, By, Bz) y reporta RMSE/R² mT
+ residuos físicos (|∇·B|, |∇×B|) vía autograd sobre la predicción.
"""
import numpy as np
import torch
from torchmetrics.regression import MeanSquaredError, R2Score

from .data import load_full_grid_for_sample


def _predict_with_residuals(lit_model, sensors_J, query_xyz_J, device, *, chunk=8192):
    """Forward chunked + residuos físicos (vía autograd). Por cada chunk:
    - b_pred: (chunk, 3) en mT
    - div:    (chunk,)   en T/m  (= mT/mm)
    - curl_mag: (chunk,) en T/m, norma euclídea de (curl_x, curl_y, curl_z)
    """
    lit_model.eval()
    b_out, div_out, curl_out = [], [], []
    for start in range(0, sensors_J.shape[0], chunk):
        end = start + chunk
        s = sensors_J[start:end].to(device)
        q = query_xyz_J[start:end].to(device)
        with torch.enable_grad():
            q_g = q.detach().clone().requires_grad_(True)
            b_norm, div, cx, cy, cz = lit_model._physics_residuals(s, q_g)
            b_mt = b_norm * lit_model.b_std + lit_model.b_mean
            curl_mag = torch.sqrt(cx**2 + cy**2 + cz**2)
        b_out.append(b_mt.detach().cpu().numpy())
        div_out.append(div.detach().cpu().numpy())
        curl_out.append(curl_mag.detach().cpu().numpy())
    return (np.concatenate(b_out, axis=0),
            np.concatenate(div_out, axis=0),
            np.concatenate(curl_out, axis=0))


def evaluate(lit_model, h5_path, sample_indices, device, *,
             rmse_per_component=True, compute_physics=True, predict_chunk=8192):
    lit_model.eval().to(device)

    yt_chunks, yp_chunks = [], []
    div_chunks, curl_chunks = [], []
    for sample_i in sample_indices:
        sensors_J, query_xyz_J, b_true_mt = load_full_grid_for_sample(h5_path, int(sample_i))
        if compute_physics:
            b_pred_mt, div_pred, curl_pred = _predict_with_residuals(
                lit_model, sensors_J, query_xyz_J, device, chunk=predict_chunk)
            div_chunks.append(div_pred)
            curl_chunks.append(curl_pred)
        else:
            from torch import no_grad
            with no_grad():
                outs = []
                for start in range(0, sensors_J.shape[0], predict_chunk):
                    end = start + predict_chunk
                    outs.append(lit_model.predict_mT(
                        sensors_J[start:end].to(device),
                        query_xyz_J[start:end].to(device)).cpu().numpy())
                b_pred_mt = np.concatenate(outs, axis=0)
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
    line = (f"[{name:>5}]  RMSE={m['rmse_mt']:.4e} mT   "
            f"R²={m['r2']:+.4f}   (N={m['n']})")
    print(line)
    if "rmse_per_component" in m:
        per = m["rmse_per_component"]
        print(f"          RMSE Bx={per['Bx']:.4e}  By={per['By']:.4e}  Bz={per['Bz']:.4e}  mT")
    if "physics" in m:
        ph = m["physics"]
        print(f"          |div B|  mean={ph['mean_abs_div_T_per_m']:.3e}  p95={ph['p95_abs_div_T_per_m']:.3e}  max={ph['max_abs_div_T_per_m']:.3e}  T/m")
        print(f"          |curl B| mean={ph['mean_curl_mag_T_per_m']:.3e}  p95={ph['p95_curl_mag_T_per_m']:.3e}  max={ph['max_curl_mag_T_per_m']:.3e}  T/m")
