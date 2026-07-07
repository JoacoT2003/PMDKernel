"""HDF5 → GPU + scalers para v5_deepsets_pinn (DeepSets + PINN, output 3-vector).

Diferencias con `v4_deepsets/data.py`:
- Target es `(Bx, By, Bz)` en lugar de solo `By` → la red predice el campo
  vectorial completo, necesario para imponer ∇·B = 0 y ∇×B = 0.
- `b_mean, b_std` son vectores `(3,)` en lugar de escalares.
- El target en GPU es `(n, J, 3)` en lugar de `(n, J)` → 3× más memoria.
  Para un dataset N=5000 con J~50k esto es ~3 GB en train split (fp32).
  Si no entra, bajá la cantidad de muestras o usá `data_device='cpu'`.
- Sensores siguen siendo solo By (`(n, I)`) — el setup físico no cambia.

Convención de dimensiones: idéntica a v4.
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def load_dataset_metadata(h5_path):
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
    assert J == Nx * Ny * Nz
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
    """Stats per-componente para target B (3,) + 5-channel para features."""
    h5_path = Path(h5_path)
    train_idx = np.asarray(train_indices, dtype=np.int64)

    # --- 1. b_mean, b_std per-componente (3,) sobre TODO el train ------------
    # Welford vectorial sobre las 3 componentes simultáneamente.
    b_count = 0
    b_mean  = np.zeros(3, dtype=np.float64)
    b_M2    = np.zeros(3, dtype=np.float64)

    with h5py.File(h5_path, "r") as f:
        for start in range(0, len(train_idx), chunk):
            idx_sorted = np.sort(train_idx[start:start + chunk])
            bs = f["geometria/grid/B"][idx_sorted].astype(np.float64)        # (chunk, J, 3)
            bs_flat = bs.reshape(-1, 3)                                      # (chunk*J, 3)
            n_b = bs_flat.shape[0]
            mean_b = bs_flat.mean(axis=0)
            M2_b   = ((bs_flat - mean_b) ** 2).sum(axis=0)
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

    b_var = b_M2 / max(b_count, 1)
    b_std = np.sqrt(b_var)
    b_std = np.where(b_std > 1e-12, b_std, 1.0).astype(np.float32)
    b_mean = b_mean.astype(np.float32)

    # feat: (dx, dy, dz, r, By_sens) — mismas que v4
    K_sub = min(n_queries_for_feat, R_grid_arr.shape[0])
    rng = np.random.default_rng(int(seed))
    q_idx = rng.choice(R_grid_arr.shape[0], size=K_sub, replace=False)
    queries = R_grid_arr[q_idx]
    diff = R_sens_arr[None, :, :] - queries[:, None, :]
    r    = np.linalg.norm(diff, axis=-1, keepdims=True)
    diff_e = np.broadcast_to(diff, (n_sub, K_sub, R_sens_arr.shape[0], 3))
    r_e    = np.broadcast_to(r,    (n_sub, K_sub, R_sens_arr.shape[0], 1))
    by_e   = np.broadcast_to(B_sens_sub[:, None, :, None],
                             (n_sub, K_sub, R_sens_arr.shape[0], 1))
    feats  = np.concatenate([diff_e, r_e, by_e], axis=-1).reshape(-1, 5)

    feat_mean = feats.mean(axis=0).astype(np.float32)
    feat_std  = feats.std(axis=0)
    feat_std  = np.where(feat_std > 1e-12, feat_std, 1.0).astype(np.float32)

    return {
        "b_mean":     b_mean,           # (3,)
        "b_std":      b_std,            # (3,)
        "feat_mean":  feat_mean,        # (5,)
        "feat_std":   feat_std,         # (5,)
        "sensor_xyz": R_sens_arr,       # (I, 3)
    }


class PointDataset(Dataset):
    """Yields `(sensores_by (K, I), query_xyz (K, 3), B_target (K, 3))`.

    Target es el campo vectorial completo normalizado per-componente:
    `(B - b_mean) / b_std`, con `b_mean`, `b_std` de shape `(3,)`.

    `query_xyz` queda **raw mm** — la red se encarga de la normalización
    interna de features y, en `training_step`, lo envuelve con
    `requires_grad_(True)` para que autograd compute las derivadas espaciales
    de las physics losses.
    """

    def __init__(self, h5_path, sample_indices, *,
                 points_per_sample,
                 batch_size=1,
                 b_mean, b_std,
                 data_device="cpu", seed=0):
        self.K          = int(points_per_sample)
        self.batch_size = int(batch_size)
        self.device     = torch.device(data_device)
        self.base_seed  = int(seed)
        self._gen       = torch.Generator(device=self.device).manual_seed(self.base_seed)

        b_mean_a = np.asarray(b_mean, dtype=np.float32).reshape(3)
        b_std_a  = np.asarray(b_std,  dtype=np.float32).reshape(3)

        sorted_idx = np.sort(np.asarray(sample_indices, dtype=np.int64))
        with h5py.File(h5_path, "r") as f:
            B_sens_arr = f["geometria/sens/B"][sorted_idx][..., 1].astype(np.float32)  # (n, I) — solo By
            B_grid_arr = f["geometria/grid/B"][sorted_idx].astype(np.float32)          # (n, J, 3) — full B
            R_grid_arr = f["geometria/grid/R"][:].astype(np.float32)                    # (J, 3)

        B_grid_norm = ((B_grid_arr - b_mean_a) / b_std_a).astype(np.float32)            # (n, J, 3)

        self.B_sens     = torch.from_numpy(B_sens_arr).to(self.device)                  # (n, I) raw
        self.B_grid_n   = torch.from_numpy(B_grid_norm).to(self.device)                 # (n, J, 3)
        self.R_grid     = torch.from_numpy(R_grid_arr).to(self.device)                  # (J, 3) raw
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

        sensors   = self.B_sens[sample_idx]                                   # (B, I)
        query_xyz = self.R_grid[point_idx]                                    # (B, K_eff, 3)

        # Gather de B_grid_n: shape (n, J, 3) → (B, K_eff, 3)
        B_sel = self.B_grid_n[sample_idx]                                     # (B, J, 3)
        b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, K_eff)
        b_target = B_sel[b_idx, point_idx]                                    # (B, K_eff, 3)

        sensors_rep = sensors.unsqueeze(1).expand(-1, K_eff, -1)              # (B, K_eff, I)

        # Aplanar batch y K → (B*K, ...). El modelo es point-wise: cada item
        # tiene su propio (sensors_by, query_xyz) y por lo tanto su propio
        # residuo físico independiente (ver model.py:_step).
        BK = B * K_eff
        return (
            sensors_rep.reshape(BK, -1).contiguous(),                          # (B*K, I)
            query_xyz.reshape(BK, 3),                                          # (B*K, 3)
            b_target.reshape(BK, 3),                                           # (B*K, 3)
        )


def make_loader(h5_path, sample_indices, *,
                points_per_sample, shuffle,
                b_mean, b_std,
                batch_size=1,
                data_device="cpu",
                num_workers=0, pin_memory=False, seed=0):
    ds = PointDataset(h5_path, sample_indices,
                      points_per_sample=points_per_sample,
                      batch_size=batch_size,
                      b_mean=b_mean, b_std=b_std,
                      data_device=data_device, seed=seed)
    if torch.device(data_device).type == "cuda":
        num_workers = 0
        pin_memory  = False
    return DataLoader(
        ds, batch_size=None, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def load_full_grid_for_sample(h5_path, sample_idx):
    """Devuelve `(sensors_J (J, I), query_xyz (J, 3), b_full_mt (J, 3))` raw."""
    h5_path = Path(h5_path)
    sample_idx = int(sample_idx)
    with h5py.File(h5_path, "r") as f:
        sensors = f["geometria/sens/B"][sample_idx][..., 1].astype(np.float32)    # (I,)
        R_grid  = f["geometria/grid/R"][:].astype(np.float32)                      # (J, 3)
        b_full  = f["geometria/grid/B"][sample_idx].astype(np.float32)            # (J, 3)
    sensors_rep = np.broadcast_to(sensors, (R_grid.shape[0], sensors.shape[0])).copy()
    return (torch.from_numpy(sensors_rep),
            torch.from_numpy(R_grid),
            torch.from_numpy(b_full))
