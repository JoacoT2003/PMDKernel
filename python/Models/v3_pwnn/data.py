"""HDF5 → GPU + scalers para v3_pwnn (point-wise, solo By).

Diferencias con `v2_1_pinn/data.py`:
- Input sensorial es **solo By** `(I,)` (no las 3 componentes).
- Target es **solo By** `(K, 1)` (no las 3 componentes).
- `B_grid_norm` y `B_sens` se pueden subir a GPU una vez (flag `data_device`)
  para eliminar el host→device transfer por step.
- Sampleo de puntos con `torch.randperm` en device (mucho más rápido que
  `np.random.choice(replace=False)` con J grande).
- `b_mean`/`b_std` son escalares (no vector de 3).

Convención de dimensiones:
- N = nº de muestras
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


def load_dataset_metadata(h5_path):
    """Lee SÓLO metadatos del HDF5 (layout v2). NO carga las matrices B grandes."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        N = int(f["geometria/grid/B"].shape[0])
        J = int(f["geometria/grid/B"].shape[1])
        I = int(f["geometria/sens/B"].shape[1])
        R_grid_xyz = f["geometria/grid/R"][:].astype(np.float32)
        sens_xyz   = f["geometria/sens/R"][:].astype(np.float32)
        grid_x     = f["geometria/grid/meta/x"][:].astype(np.float32)
        grid_y     = f["geometria/grid/meta/y"][:].astype(np.float32)
        grid_z     = f["geometria/grid/meta/z"][:].astype(np.float32)
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
    """Stats para normalización (solo By).

    - `x_scaler`  : `StandardScaler` per-feature sobre `B_sens[..., By]` shape (I,).
    - `b_mean`    : float — media escalar de By sobre todo el grid.
    - `b_std`     : float — std escalar de By sobre todo el grid.
    - `pts_mean`  : np.array shape (3,) — media de x, y, z.
    - `pts_std`   : np.array shape (3,) — std de x, y, z.

    Streaming sobre HDF5 con Welford incremental para By (numéricamente estable).
    """
    if chunk < 2:
        raise ValueError(f"chunk debe ser >= 2 (dado: {chunk})")

    h5_path = Path(h5_path)
    train_idx = np.asarray(train_indices, dtype=np.int64)

    x_scaler = StandardScaler()
    # Welford incremental escalar
    b_count = 0
    b_mean  = 0.0
    b_M2    = 0.0

    with h5py.File(h5_path, "r") as f:
        for start in range(0, len(train_idx), chunk):
            idx = train_idx[start:start + chunk]
            idx_sorted = np.sort(idx)

            # Solo componente By (índice 1) — evita cargar Bx/Bz que no usamos.
            xs = f["geometria/sens/B"][idx_sorted][..., 1].astype(np.float64)   # (chunk, I)
            x_scaler.partial_fit(xs)

            bs = f["geometria/grid/B"][idx_sorted][..., 1].astype(np.float64)   # (chunk, J)
            bs_flat = bs.ravel()

            n_b = bs_flat.size
            mean_b = bs_flat.mean()
            M2_b   = ((bs_flat - mean_b) ** 2).sum()

            n_total = b_count + n_b
            delta   = mean_b - b_mean
            b_mean  = b_mean + delta * (n_b / n_total)
            b_M2    = b_M2 + M2_b + (delta ** 2) * (b_count * n_b / n_total)
            b_count = n_total

        R_grid = f["geometria/grid/R"][:].astype(np.float64)   # (J, 3)

    b_std = float(np.sqrt(b_M2 / max(b_count, 1)))
    b_std = b_std if b_std > 1e-12 else 1.0
    b_mean = float(b_mean)

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
    """Yields `(sensores_K (K, I), puntos_K (K, 3), By_K (K, 1))` desde tensores in-device.

    Carga el split entero a `data_device` al inicializar y pre-aplica las
    normalizaciones. `__getitem__` queda como indexing torch puro on-device —
    sin syscalls, sin parsing HDF5, sin host→GPU transfer por step.

    Normalización:
      - sensores : `x_scaler.transform`  (per-feature sobre las I lecturas de By)
      - puntos   : `(xyz - pts_mean) / pts_std`
      - By target: `(By - b_mean) / b_std` (escalar)

    El DataLoader debe usarse con `batch_size=None`: cada item ya es un
    mini-batch de K puntos de un único sample.

    `data_device='cuda'` requiere que el DataLoader corra con `num_workers=0`
    (los workers no pueden compartir tensores GPU vía multiprocessing).
    """

    def __init__(self, h5_path, sample_indices, *,
                 points_per_sample,
                 x_scaler=None,
                 b_mean=0.0, b_std=1.0,
                 pts_mean=None, pts_std=None,
                 data_device="cpu",
                 seed=0):
        self.K         = int(points_per_sample)
        self.device    = torch.device(data_device)
        self.base_seed = int(seed)
        # Generator on-device para `torch.randperm`. Para CPU el generator es CPU.
        self._gen = torch.Generator(device=self.device).manual_seed(self.base_seed)

        b_mean_s   = float(b_mean)
        b_std_s    = float(b_std)
        pts_mean_a = _as_f32_3(pts_mean, default=0.0)
        pts_std_a  = _as_f32_3(pts_std,  default=1.0)

        sorted_idx = np.sort(np.asarray(sample_indices, dtype=np.int64))
        with h5py.File(h5_path, "r") as f:
            # Slice solo By (componente 1) → ahorra 3× memoria respecto a v2.1.
            B_sens_arr = f["geometria/sens/B"][sorted_idx][..., 1].astype(np.float32)  # (n, I)
            B_grid_arr = f["geometria/grid/B"][sorted_idx][..., 1].astype(np.float32)  # (n, J)
            R_grid_arr = f["geometria/grid/R"][:].astype(np.float32)                   # (J, 3)

        if x_scaler is not None:
            B_sens_arr = x_scaler.transform(B_sens_arr).astype(np.float32)
        B_grid_arr = ((B_grid_arr - b_mean_s) / b_std_s).astype(np.float32)
        R_grid_arr = ((R_grid_arr - pts_mean_a) / pts_std_a).astype(np.float32)

        # Subimos todo al device elegido. Para data_device='cuda' esto elimina
        # el host→GPU transfer per step (el bottleneck principal de v2.1).
        self.B_sens      = torch.from_numpy(B_sens_arr).to(self.device)        # (n, I)
        self.By_grid_n   = torch.from_numpy(B_grid_arr).to(self.device)        # (n, J)
        self.R_grid_norm = torch.from_numpy(R_grid_arr).to(self.device)        # (J, 3)
        self.J           = self.R_grid_norm.shape[0]
        self.I           = self.B_sens.shape[1]

    def __len__(self):
        return self.B_sens.shape[0]

    def __getitem__(self, k):
        sensors = self.B_sens[k]                                          # (I,)
        K_eff = min(self.K, self.J)
        # torch.randperm on device — bypassa numpy y el CPU→GPU transfer del v2.1.
        point_idx = torch.randperm(self.J, generator=self._gen,
                                   device=self.device)[:K_eff]
        points_norm = self.R_grid_norm[point_idx]                          # (K, 3)
        by_norm     = self.By_grid_n[k, point_idx]                         # (K,)
        sensors_rep = sensors.unsqueeze(0).expand(K_eff, -1).contiguous()  # (K, I)
        return sensors_rep, points_norm, by_norm.unsqueeze(-1)             # (K, 1)


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
                b_mean=0.0, b_std=1.0,
                pts_mean=None, pts_std=None,
                data_device="cpu",
                num_workers=0, pin_memory=False, seed=0):
    """DataLoader sobre `PointDataset` con `batch_size=None`.

    Si `data_device='cuda'`, `num_workers` se fuerza a 0 y `pin_memory` a False
    (los tensores ya viven en GPU; multiprocessing no funciona con CUDA tensors).
    """
    ds = PointDataset(h5_path, sample_indices,
                      points_per_sample=points_per_sample,
                      x_scaler=x_scaler,
                      b_mean=b_mean, b_std=b_std,
                      pts_mean=pts_mean, pts_std=pts_std,
                      data_device=data_device,
                      seed=seed)
    if torch.device(data_device).type == "cuda":
        num_workers = 0
        pin_memory  = False
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
    """Helper para evaluación: carga TODO el grid de una sample (J puntos), solo By.

    Devuelve `(sensors_J, points_J_norm, by_full_mt)`:
      - `sensors_J` (J, I) ya normalizado por `x_scaler`.
      - `points_J_norm` (J, 3) normalizado por `(pts - mean)/std`.
      - `by_full_mt` (J,) raw mT (sin normalizar) para comparar contra física.
    """
    pts_mean_a = _as_f32_3(pts_mean, default=0.0)
    pts_std_a  = _as_f32_3(pts_std,  default=1.0)

    h5_path = Path(h5_path)
    sample_idx = int(sample_idx)
    with h5py.File(h5_path, "r") as f:
        sensors = f["geometria/sens/B"][sample_idx][..., 1].astype(np.float32)     # (I,)
        if x_scaler is not None:
            sensors = x_scaler.transform(sensors[None, :]).reshape(-1).astype(np.float32)
        R_grid  = f["geometria/grid/R"][:].astype(np.float32)                       # (J, 3) mm
        by_full = f["geometria/grid/B"][sample_idx][..., 1].astype(np.float32)     # (J,) mT raw

    points_norm = ((R_grid - pts_mean_a) / pts_std_a).astype(np.float32)
    sensors_rep = np.broadcast_to(sensors, (R_grid.shape[0], sensors.shape[0])).copy()
    return (torch.from_numpy(sensors_rep),
            torch.from_numpy(points_norm),
            torch.from_numpy(by_full))
