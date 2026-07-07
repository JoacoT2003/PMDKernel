"""HDF5 → GPU + scalers para v4_deepsets (DeepSets, point-wise, solo By).

Diferencias con `v3_pwnn/data.py`:
- El dataset NO normaliza sensores ni coords — la red recibe `sensors_by` y
  `query_xyz` raw (en mm). La normalización ocurre **dentro del modelo** vía
  los buffers `feat_mean`, `feat_std` (5,), porque las features per-sensor
  `(dx, dy, dz, r, By_sens)` se construyen on-the-fly en el `forward`.
- Las stats de feature (5 canales) se calculan en `compute_train_stats` con un
  subsampleo (no necesitan pasar por todo el dataset).
- El dataset también expone `sensor_xyz` (I, 3) — la red lo registra como
  buffer para construir las coords relativas en el forward.

Convención de dimensiones:
- N = nº de muestras
- I = nº de sensores
- J = nº de puntos de grilla
- K = nº de puntos por step
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
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

    return {
        "h5_path": str(h5_path),
        "N": N, "I": I, "J": J,
        "Nx": Nx, "Ny": Ny, "Nz": Nz,
        "grid_x": grid_x, "grid_y": grid_y, "grid_z": grid_z,
        "R_grid_xyz": R_grid_xyz,
        "sens_xyz": sens_xyz,
        "attrs": attrs,
    }


def split_indices(N, *, val_frac, test_frac, seed):
    idx = np.arange(N, dtype=np.int64)
    idx_trainval, idx_test = train_test_split(idx, test_size=test_frac, random_state=seed)
    val_size_rel = val_frac / (1.0 - test_frac)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=val_size_rel, random_state=seed)
    return {
        "train":   idx_train,
        "val":     idx_val,
        "test":    idx_test,
        "n_train": len(idx_train),
        "n_val":   len(idx_val),
        "n_test":  len(idx_test),
    }


def compute_train_stats(h5_path, train_indices, *,
                        chunk=64, n_samples_for_feat=64, n_queries_for_feat=1024,
                        seed=0):
    """Stats para normalización.

    - `b_mean`, `b_std`         : escalares (target By).
    - `feat_mean`, `feat_std`   : (5,) — para (dx, dy, dz, r, By_sens) del input
                                  per-sensor. Se calcula con subsample
                                  `n_samples_for_feat × n_queries_for_feat × I`
                                  porque el espacio de features no necesita
                                  full statistics para una normalización razonable.

    Welford incremental sobre todos los puntos del train para b_mean/std
    (numéricamente estable), subsample para feat.
    """
    h5_path = Path(h5_path)
    train_idx = np.asarray(train_indices, dtype=np.int64)

    # --- 1. b_mean, b_std (escalar, By target) sobre TODO el train -----------
    b_count = 0
    b_mean  = 0.0
    b_M2    = 0.0

    with h5py.File(h5_path, "r") as f:
        for start in range(0, len(train_idx), chunk):
            idx_sorted = np.sort(train_idx[start:start + chunk])
            bs = f["geometria/grid/B"][idx_sorted][..., 1].astype(np.float64)   # (chunk, J)
            bs_flat = bs.ravel()
            n_b = bs_flat.size
            mean_b = bs_flat.mean()
            M2_b   = ((bs_flat - mean_b) ** 2).sum()
            n_total = b_count + n_b
            delta = mean_b - b_mean
            b_mean = b_mean + delta * (n_b / n_total)
            b_M2   = b_M2 + M2_b + (delta ** 2) * (b_count * n_b / n_total)
            b_count = n_total

        R_grid_arr = f["geometria/grid/R"][:].astype(np.float32)
        R_sens_arr = f["geometria/sens/R"][:].astype(np.float32)

        # --- 2. feat stats con subsample ---------------------------------
        n_sub = min(n_samples_for_feat, len(train_idx))
        sub_idx = np.sort(train_idx[:n_sub])
        B_sens_sub = f["geometria/sens/B"][sub_idx][..., 1].astype(np.float32)   # (n_sub, I)

    b_std = float(np.sqrt(b_M2 / max(b_count, 1)))
    b_std = b_std if b_std > 1e-12 else 1.0
    b_mean = float(b_mean)

    # Subsample de query points
    K_sub = min(n_queries_for_feat, R_grid_arr.shape[0])
    rng = np.random.default_rng(int(seed))
    q_idx = rng.choice(R_grid_arr.shape[0], size=K_sub, replace=False)
    queries = R_grid_arr[q_idx]                                                   # (K_sub, 3)

    # Build features (n_sub, K_sub, I, 5) → flatten
    # diff[s, k, i] = R_sens[i] - queries[k]    (independiente de s)
    diff = R_sens_arr[None, :, :] - queries[:, None, :]                          # (K_sub, I, 3)
    r    = np.linalg.norm(diff, axis=-1, keepdims=True)                          # (K_sub, I, 1)
    diff_e = np.broadcast_to(diff, (n_sub, K_sub, R_sens_arr.shape[0], 3))
    r_e    = np.broadcast_to(r,    (n_sub, K_sub, R_sens_arr.shape[0], 1))
    by_e   = np.broadcast_to(B_sens_sub[:, None, :, None],
                             (n_sub, K_sub, R_sens_arr.shape[0], 1))
    feats  = np.concatenate([diff_e, r_e, by_e], axis=-1).reshape(-1, 5)         # (n_sub*K_sub*I, 5)

    feat_mean = feats.mean(axis=0).astype(np.float32)
    feat_std  = feats.std(axis=0)
    feat_std  = np.where(feat_std > 1e-12, feat_std, 1.0).astype(np.float32)

    return {
        "b_mean":    b_mean,
        "b_std":     b_std,
        "feat_mean": feat_mean,
        "feat_std":  feat_std,
        "sensor_xyz": R_sens_arr,            # (I, 3) — se inyecta como buffer al modelo
    }


class PointDataset(Dataset):
    """Yields `(sensores_by_K (K, I), query_xyz_K (K, 3), by_K (K, 1))`.

    `sensores_by_K` y `query_xyz_K` se devuelven **raw** (en mT y mm
    respectivamente). La red los normaliza on-the-fly al construir las features
    per-sensor `(dx, dy, dz, r, By_sens)` con sus buffers `feat_mean/feat_std`.
    `by_K` ya viene normalizado con `b_mean/b_std` (el target del MSE).
    """

    def __init__(self, h5_path, sample_indices, *,
                 points_per_sample,
                 batch_size=1,
                 b_mean=0.0, b_std=1.0,
                 data_device="cpu",
                 seed=0):
        self.K          = int(points_per_sample)
        self.batch_size = int(batch_size)
        self.device     = torch.device(data_device)
        self.base_seed  = int(seed)
        self._gen       = torch.Generator(device=self.device).manual_seed(self.base_seed)

        b_mean_s = float(b_mean)
        b_std_s  = float(b_std)

        sorted_idx = np.sort(np.asarray(sample_indices, dtype=np.int64))
        with h5py.File(h5_path, "r") as f:
            # Solo componente By en sensores y grilla
            B_sens_arr = f["geometria/sens/B"][sorted_idx][..., 1].astype(np.float32)  # (n, I)
            B_grid_arr = f["geometria/grid/B"][sorted_idx][..., 1].astype(np.float32)  # (n, J)
            R_grid_arr = f["geometria/grid/R"][:].astype(np.float32)                   # (J, 3) mm

        B_grid_norm = (B_grid_arr - b_mean_s) / b_std_s                                # target normalizado

        self.B_sens    = torch.from_numpy(B_sens_arr).to(self.device)        # (n, I) raw mT
        self.By_grid_n = torch.from_numpy(B_grid_norm).to(self.device)       # (n, J)
        self.R_grid    = torch.from_numpy(R_grid_arr).to(self.device)        # (J, 3) raw mm
        self.J = self.R_grid.shape[0]
        self.I = self.B_sens.shape[1]
        self.N = self.B_sens.shape[0]

    def __len__(self):
        return (self.N + self.batch_size - 1) // self.batch_size

    def __getitem__(self, batch_idx):
        start = batch_idx * self.batch_size
        end   = min(start + self.batch_size, self.N)
        B     = end - start
        K_eff = min(self.K, self.J)

        sample_idx = torch.arange(start, end, device=self.device)             # (B,)
        point_idx  = torch.stack([
            torch.randperm(self.J, generator=self._gen, device=self.device)[:K_eff]
            for _ in range(B)
        ])                                                                    # (B, K_eff)

        sensors     = self.B_sens[sample_idx]                                 # (B, I)
        query_xyz   = self.R_grid[point_idx]                                  # (B, K_eff, 3)
        by_norm     = torch.gather(self.By_grid_n[sample_idx], 1, point_idx)  # (B, K_eff)
        sensors_rep = sensors.unsqueeze(1).expand(-1, K_eff, -1)              # (B, K_eff, I)

        # Aplanar batch y K → (B*K_eff, ...). El modelo es point-wise: no le
        # importa qué muestra originó cada punto. Forward único por step.
        BK = B * K_eff
        return (
            sensors_rep.reshape(BK, -1).contiguous(),                          # (B*K, I)
            query_xyz.reshape(BK, 3),                                          # (B*K, 3)
            by_norm.reshape(BK, 1),                                            # (B*K, 1)
        )


def make_loader(h5_path, sample_indices, *,
                points_per_sample, shuffle,
                batch_size=1,
                b_mean=0.0, b_std=1.0,
                data_device="cpu",
                num_workers=0, pin_memory=False, seed=0):
    """DataLoader sobre `PointDataset` con `batch_size=None`."""
    ds = PointDataset(h5_path, sample_indices,
                      points_per_sample=points_per_sample,
                      batch_size=batch_size,
                      b_mean=b_mean, b_std=b_std,
                      data_device=data_device,
                      seed=seed)
    if torch.device(data_device).type == "cuda":
        num_workers = 0
        pin_memory  = False
    return DataLoader(
        ds, batch_size=None, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def load_full_grid_for_sample(h5_path, sample_idx):
    """Devuelve `(sensors_J (J, I), query_xyz (J, 3), by_full_mt (J,))` raw mT/mm.

    No normaliza nada — la red se encarga via sus buffers. `by_full_mt`
    queda en mT físicas para comparar contra el ground truth.
    """
    h5_path = Path(h5_path)
    sample_idx = int(sample_idx)
    with h5py.File(h5_path, "r") as f:
        sensors = f["geometria/sens/B"][sample_idx][..., 1].astype(np.float32)   # (I,)
        R_grid  = f["geometria/grid/R"][:].astype(np.float32)                     # (J, 3)
        by_full = f["geometria/grid/B"][sample_idx][..., 1].astype(np.float32)   # (J,)
    sensors_rep = np.broadcast_to(sensors, (R_grid.shape[0], sensors.shape[0])).copy()
    return (torch.from_numpy(sensors_rep),
            torch.from_numpy(R_grid),
            torch.from_numpy(by_full))
