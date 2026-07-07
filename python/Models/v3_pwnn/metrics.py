"""Evaluación de v3_pwnn — predice By en todo el grid y reporta RMSE/R² en mT.

La red es per-punto sobre By: para evaluar reconstruimos el grid completo
iterando sobre los J puntos de cada sample.

Desnormalización escalar: la red devuelve `By_norm` y la salida física es
`By_real = By_norm · b_std + b_mean`, con `b_std`, `b_mean` escalares.
"""
import numpy as np
import torch
from torchmetrics.regression import MeanSquaredError, R2Score

from .data import load_full_grid_for_sample


def _predict_full_grid_mT(lit_model, sensors_J, points_J_norm,
                          b_mean_s, b_std_s, device, *, chunk=8192):
    """Forward chunked sobre los J puntos de una sample. Devuelve (J,) numpy
    en mT (ya desnormalizado)."""
    lit_model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, sensors_J.shape[0], chunk):
            end = start + chunk
            s = sensors_J[start:end].to(device)
            p = points_J_norm[start:end].to(device)
            by_norm = lit_model(s, p)                    # (chunk, 1) normalizado
            by_mt = by_norm * b_std_s + b_mean_s         # (chunk, 1) mT
            outs.append(by_mt.squeeze(-1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def evaluate(lit_model, h5_path, sample_indices,
             x_scaler, b_mean, b_std, pts_mean, pts_std, device, *,
             predict_chunk=8192):
    """Evalúa `lit_model` sobre `sample_indices` reconstruyendo el grid de
    cada sample. Devuelve dict con RMSE y R² globales en mT físicas.
    """
    lit_model.eval().to(device)
    b_mean_s = torch.tensor(float(b_mean), dtype=torch.float32, device=device)
    b_std_s  = torch.tensor(float(b_std),  dtype=torch.float32, device=device)

    yt_chunks, yp_chunks = [], []
    for sample_i in sample_indices:
        sensors_J, points_J_norm, by_true_mt = load_full_grid_for_sample(
            h5_path, int(sample_i),
            x_scaler=x_scaler, pts_mean=pts_mean, pts_std=pts_std,
        )
        by_pred_mt = _predict_full_grid_mT(
            lit_model, sensors_J, points_J_norm,
            b_mean_s, b_std_s, device, chunk=predict_chunk,
        )
        yt_chunks.append(by_true_mt.numpy())
        yp_chunks.append(by_pred_mt)
    yt = np.concatenate(yt_chunks, axis=0).astype(np.float32)
    yp = np.concatenate(yp_chunks, axis=0).astype(np.float32)

    yt_t = torch.from_numpy(yt).to(device)
    yp_t = torch.from_numpy(yp).to(device)

    rmse = MeanSquaredError(squared=False).to(device)(yp_t, yt_t).item()
    r2   = R2Score().to(device)(yp_t, yt_t).item()

    return {"rmse_mt": float(rmse), "r2": float(r2), "n": int(len(yt))}


def report(name, m):
    line = (f"[{name:>5}]  RMSE={m['rmse_mt']:.4e} mT   "
            f"R²={m['r2']:+.4f}   (N={m['n']})")
    print(line)
