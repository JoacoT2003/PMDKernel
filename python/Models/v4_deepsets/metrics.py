"""Evaluación de v4_deepsets — predice By en todo el grid y reporta RMSE/R² mT."""
import numpy as np
import torch
from torchmetrics.regression import MeanSquaredError, R2Score

from .data import load_full_grid_for_sample


def _predict_full_grid_mT(lit_model, sensors_J, query_xyz_J, device, *, chunk=8192):
    """Forward chunked sobre los J puntos de una sample. Devuelve (J,) numpy mT
    (ya desnormalizado via los buffers internos del modelo)."""
    lit_model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, sensors_J.shape[0], chunk):
            end = start + chunk
            s = sensors_J[start:end].to(device)
            q = query_xyz_J[start:end].to(device)
            by_mt = lit_model.predict_mT(s, q)                # (chunk, 1) mT
            outs.append(by_mt.squeeze(-1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def evaluate(lit_model, h5_path, sample_indices, device, *, predict_chunk=8192):
    lit_model.eval().to(device)

    yt_chunks, yp_chunks = [], []
    for sample_i in sample_indices:
        sensors_J, query_xyz_J, by_true_mt = load_full_grid_for_sample(h5_path, int(sample_i))
        by_pred_mt = _predict_full_grid_mT(lit_model, sensors_J, query_xyz_J, device,
                                            chunk=predict_chunk)
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
    print(f"[{name:>5}]  RMSE={m['rmse_mt']:.4e} mT   R²={m['r2']:+.4f}   (N={m['n']})")
