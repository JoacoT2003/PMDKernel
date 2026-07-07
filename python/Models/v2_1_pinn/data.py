"""HDF5 → RAM + scalers per-componente + PointDataset para v2_1_pinn.

Idéntico a v2_pinn/data.py — la diferencia entre v2 y v2.1 es solamente el
modelo (v2.1 usa MSE puro, sin physics losses). El pipeline de datos
(normalización per-componente, PointDataset, DataLoader con batch_size=None)
se conserva tal cual.

La unidad de entrenamiento es una **tupla (sensores, punto, B_en_punto)**. La
red es `f(B_sens, x, y, z) → B(x, y, z)`.

`PointDataset.__getitem__(k)` devuelve UN sample con K puntos al azar de su
grilla. `PointDataset` levanta el split entero a RAM como tensores torch
float32 al inicializar y pre-aplica las normalizaciones — `__getitem__` queda
como indexing puro sin tocar el disco.

**Normalización**

  - sensores : `StandardScaler` per-feature (I·3 medias, I·3 escalas).
  - puntos   : `(xyz - pts_mean) / pts_std`, vector (3,) independiente por eje.
  - B target : `(B - b_mean) / b_std`, vector (3,) independiente por componente.

Convención:
- N = nº de muestras (configuraciones de perturbación)
- I = nº de sensores
- J = nº de puntos de grilla
- K = nº de puntos por step (hyperparam de batching)
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def load_dataset(h5_path):
    """Lee SÓLO metadatos del HDF5. NO carga `B_grid`/`B_sens`."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        N = int(f["B_grid"].shape[0])
        J = int(f["B_grid"].shape[1])
        I = int(f["B_sens"].shape[1])
        R_grid_xyz = f["R_grid_xyz_mm"][:].astype(np.float32)
        sens_xyz   = f["sens_xyz_mm"][:].astype(np.float32)
        grid_x     = f["grid_x"][:].astype(np.float32)
        grid_y     = f["grid_y"][:].astype(np.float32)
        grid_z     = f["grid_z"][:].astype(np.float32)
        attrs      = dict(f.attrs)

    Nx, Ny, Nz = len(grid_x), len(grid_y), len(grid_z)
    assert J == Nx * Ny * Nz, f"J={J} != Nx*Ny*Nz={Nx*Ny*Nz}"

    R_grid_3d = R_grid_xyz.reshape(Nx, Ny, Nz, 3)
    gxx, gyy, gzz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    assert (np.allclose(R_grid_3d[..., 0], gxx)
            and np.allclose(R_grid_3d[..., 1], gyy)
            and np.allclose(R_grid_3d[..., 2], gzz)), \
        "Layout inesperado: posiciones reshapeadas no coinciden con meshgrid (Nx,Ny,Nz)."

    return {
        "h5_path": str(h5_path),
        "N": N, "I": I, "J": J,
        "Nx": Nx, "Ny": Ny, "Nz": Nz,
        "grid_x": grid_x, "grid_y": grid_y, "grid_z": grid_z,
        "R_grid_xyz": R_grid_xyz, "R_grid_3d": R_grid_3d,
        "sens_xyz": sens_xyz,
        "attrs": attrs,
    }


def split_indices(N, *, val_frac, test_frac, seed):
    """Split vía `sklearn.model_selection.train_test_split`."""
    idx = np.arange(N, dtype=np.int64)
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_frac, random_state=seed,
    )
    val_size_rel = val_frac / (1.0 - test_frac)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_size_rel, random_state=seed,
    )
    return {
        "train":   idx_train,
        "val":     idx_val,
        "test":    idx_test,
        "n_train": len(idx_train),
        "n_val":   len(idx_val),
        "n_test":  len(idx_test),
    }


def compute_train_stats(h5_path, train_indices, *, chunk=64):
    """Stats per-componente para normalización.

    - `x_scaler`  : `StandardScaler` per-feature sobre `B_sens` (input).
    - `b_mean`    : np.array shape (3,) — media de Bx, By, Bz.
    - `b_std`     : np.array shape (3,) — std de Bx, By, Bz.
    - `pts_mean`  : np.array shape (3,) — media de x, y, z.
    - `pts_std`   : np.array shape (3,) — std de x, y, z.

    Streaming sobre HDF5 con Welford incremental para B (numéricamente estable).
    """
    if chunk < 2:
        raise ValueError(f"chunk debe ser >= 2 (dado: {chunk})")

    h5_path = Path(h5_path)
    train_idx = np.asarray(train_indices, dtype=np.int64)

    x_scaler = StandardScaler()
    # Welford incremental sobre las 3 componentes de B (mean, M2 = Σ(x - mean)²).
    b_count = 0
    b_mean  = np.zeros(3, dtype=np.float64)
    b_M2    = np.zeros(3, dtype=np.float64)

    with h5py.File(h5_path, "r") as f:
        for start in range(0, len(train_idx), chunk):
            idx = train_idx[start:start + chunk]
            idx_sorted = np.sort(idx)

            xs = f["B_sens"][idx_sorted].reshape(len(idx_sorted), -1).astype(np.float64)
            x_scaler.partial_fit(xs)

            bs = f["B_grid"][idx_sorted].astype(np.float64)   # (chunk, J, 3)
            bs_flat = bs.reshape(-1, 3)                        # (chunk·J, 3)

            # Welford batched: combinar stats acumulados con stats del batch
            n_b = bs_flat.shape[0]
            mean_b = bs_flat.mean(axis=0)
            M2_b   = ((bs_flat - mean_b) ** 2).sum(axis=0)

            n_total = b_count + n_b
            delta   = mean_b - b_mean
            b_mean  = b_mean + delta * (n_b / n_total)
            b_M2    = b_M2 + M2_b + (delta ** 2) * (b_count * n_b / n_total)
            b_count = n_total

        # Coords: el grid es el mismo para todos los samples, no necesita stream
        R_grid = f["R_grid_xyz_mm"][:].astype(np.float64)      # (J, 3)

    b_var = b_M2 / max(b_count, 1)
    b_std = np.sqrt(b_var)
    # Guarda contra std≈0 (componente trivialmente constante): cae a 1.0.
    b_std = np.where(b_std > 1e-12, b_std, 1.0).astype(np.float32)
    b_mean = b_mean.astype(np.float32)

    pts_mean = R_grid.mean(axis=0).astype(np.float32)
    pts_std  = R_grid.std(axis=0)
    pts_std  = np.where(pts_std > 1e-12, pts_std, 1.0).astype(np.float32)

    return {
        "x_scaler": x_scaler,
        "b_mean":   b_mean,
        "b_std":    b_std,
        "pts_mean": pts_mean,
        "pts_std":  pts_std,
    }


class PointDataset(Dataset):
    """Yields `(sensores_K, puntos_K_norm, B_K_norm)` desde tensores in-memory.

    Carga el split entero a RAM al inicializar y pre-aplica las normalizaciones:
      - sensores : `x_scaler.transform`
      - puntos   : `(xyz - pts_mean) / pts_std`, vector (3,)
      - B target : `(B - b_mean) / b_std`, vector (3,)

    `__getitem__` es indexing torch puro — sin syscalls, sin parsing HDF5.
    El DataLoader debe usarse con `batch_size=None`: cada item ya es un
    mini-batch de K puntos de un único sample.
    """

    def __init__(self, h5_path, sample_indices, *,
                 points_per_sample,
                 x_scaler=None,
                 b_mean=None, b_std=None,
                 pts_mean=None, pts_std=None,
                 seed=0):
        self.K         = int(points_per_sample)
        self.base_seed = int(seed)
        self._rng      = np.random.default_rng(self.base_seed)

        b_mean_a   = _as_f32_3(b_mean,   default=0.0)
        b_std_a    = _as_f32_3(b_std,    default=1.0)
        pts_mean_a = _as_f32_3(pts_mean, default=0.0)
        pts_std_a  = _as_f32_3(pts_std,  default=1.0)

        sorted_idx = np.sort(np.asarray(sample_indices, dtype=np.int64))
        with h5py.File(h5_path, "r") as f:
            B_sens_arr = f["B_sens"][sorted_idx].reshape(len(sorted_idx), -1).astype(np.float32)
            B_grid_arr = f["B_grid"][sorted_idx].astype(np.float32)            # [n, J, 3]
            R_grid_arr = f["R_grid_xyz_mm"][:].astype(np.float32)              # [J, 3]

        if x_scaler is not None:
            B_sens_arr = x_scaler.transform(B_sens_arr).astype(np.float32)
        B_grid_arr = ((B_grid_arr - b_mean_a) / b_std_a).astype(np.float32)
        R_grid_arr = ((R_grid_arr - pts_mean_a) / pts_std_a).astype(np.float32)

        self.B_sens      = torch.from_numpy(B_sens_arr)
        self.B_grid_norm = torch.from_numpy(B_grid_arr)
        self.R_grid_norm = torch.from_numpy(R_grid_arr)
        self.J           = self.R_grid_norm.shape[0]

    def __len__(self):
        return self.B_sens.shape[0]

    def __getitem__(self, k):
        sensors = self.B_sens[k]
        K_eff = min(self.K, self.J)
        point_idx = torch.from_numpy(
            self._rng.choice(self.J, size=K_eff, replace=False)
        )
        points_norm = self.R_grid_norm[point_idx]
        b_norm      = self.B_grid_norm[k, point_idx]
        sensors_rep = sensors.unsqueeze(0).expand(K_eff, -1).contiguous()
        return sensors_rep, points_norm, b_norm


def _as_f32_3(arr, *, default):
    """Coerce a `np.float32` shape (3,). Si arr es None, llena con `default`."""
    if arr is None:
        return np.full(3, default, dtype=np.float32)
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    if a.shape != (3,):
        raise ValueError(f"esperaba shape (3,), recibido {a.shape}")
    return a


def make_loader(h5_path, sample_indices, *,
                points_per_sample, shuffle,
                x_scaler=None,
                b_mean=None, b_std=None,
                pts_mean=None, pts_std=None,
                num_workers=0, pin_memory=False, seed=0):
    """DataLoader sobre `PointDataset` con `batch_size=None`."""
    ds = PointDataset(h5_path, sample_indices,
                      points_per_sample=points_per_sample,
                      x_scaler=x_scaler,
                      b_mean=b_mean, b_std=b_std,
                      pts_mean=pts_mean, pts_std=pts_std,
                      seed=seed)
    return DataLoader(
        ds,
        batch_size=None,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def load_full_grid_for_sample(h5_path, sample_idx, *,
                              x_scaler=None,
                              pts_mean=None, pts_std=None):
    """Helper para evaluación: carga TODO el grid de una sample (J puntos).

    Aplica `x_scaler` a los sensores y normaliza coords con `(pts - mean)/std`
    (igual que en training). **NO normaliza el B_target** — devuelve raw mT
    para que `metrics.evaluate(...)` compare contra física directa.
    """
    pts_mean_a = _as_f32_3(pts_mean, default=0.0)
    pts_std_a  = _as_f32_3(pts_std,  default=1.0)

    h5_path = Path(h5_path)
    sample_idx = int(sample_idx)
    with h5py.File(h5_path, "r") as f:
        sensors = f["B_sens"][sample_idx].reshape(-1).astype(np.float32)
        if x_scaler is not None:
            sensors = x_scaler.transform(sensors[None, :]).reshape(-1).astype(np.float32)
        R_grid  = f["R_grid_xyz_mm"][:].astype(np.float32)              # (J, 3) mm
        b_full  = f["B_grid"][sample_idx].astype(np.float32)            # (J, 3) mT raw
    points_norm = ((R_grid - pts_mean_a) / pts_std_a).astype(np.float32)
    sensors_rep = np.broadcast_to(sensors, (R_grid.shape[0], sensors.shape[0])).copy()
    return (torch.from_numpy(sensors_rep),
            torch.from_numpy(points_norm),
            torch.from_numpy(b_full))
