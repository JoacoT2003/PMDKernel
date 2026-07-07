"""HDF5 → GPU + scalers para v6_deepxde (PI-DeepONet, formato CartesianProd).

en v5 cada punto query re-encodeaba los I sensores (O(J·I)). DeepONet CartesianProd evalúa el
**branch 1× por muestra** sobre los I sensores y el **trunk 1× por punto** sobre
la grilla compartida

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


def compute_train_stats(h5_path, train_indices, *, chunk=64):
    """Stats sobre el split de train:

    - `b_mean`, `b_std` (3,)  — target B per-componente (Welford vectorial).
    - `branch_mean`, `branch_std` (I,) — By de cada sensor (entrada del branch).
    - `coord_mean`, `coord_std` (3,) — coords de la grilla (entrada del trunk).
    """
    h5_path = Path(h5_path)
    train_idx = np.asarray(train_indices, dtype=np.int64)

    # --- b_mean, b_std per-componente (3,) sobre todo el train (Welford) -----
    b_count = 0
    b_mean  = np.zeros(3, dtype=np.float64)
    b_M2    = np.zeros(3, dtype=np.float64)

    with h5py.File(h5_path, "r") as f:
        for start in range(0, len(train_idx), chunk):
            idx_sorted = np.sort(train_idx[start:start + chunk])
            bs = f["geometria/grid/B"][idx_sorted].astype(np.float64)        # (chunk, J, 3)
            bs_flat = bs.reshape(-1, 3)
            n_b = bs_flat.shape[0]
            mean_b = bs_flat.mean(axis=0)
            M2_b   = ((bs_flat - mean_b) ** 2).sum(axis=0)
            n_total = b_count + n_b
            delta = mean_b - b_mean
            b_mean = b_mean + delta * (n_b / n_total)
            b_M2   = b_M2 + M2_b + (delta ** 2) * (b_count * n_b / n_total)
            b_count = n_total

        # branch: By de cada sensor, stats per-sensor (I,) sobre todo el train.
        sens_by = f["geometria/sens/B"][np.sort(train_idx)][..., 1].astype(np.float64)  # (n, I)
        R_grid  = f["geometria/grid/R"][:].astype(np.float64)                            # (J, 3)
        R_sens  = f["geometria/sens/R"][:].astype(np.float32)

    b_std = np.sqrt(b_M2 / max(b_count, 1))
    b_std = np.where(b_std > 1e-12, b_std, 1.0).astype(np.float32)
    b_mean = b_mean.astype(np.float32)

    branch_mean = sens_by.mean(axis=0).astype(np.float32)              # (I,)
    branch_std  = sens_by.std(axis=0)
    branch_std  = np.where(branch_std > 1e-12, branch_std, 1.0).astype(np.float32)

    coord_mean = R_grid.mean(axis=0).astype(np.float32)                # (3,)
    coord_std  = R_grid.std(axis=0)
    coord_std  = np.where(coord_std > 1e-12, coord_std, 1.0).astype(np.float32)

    return {
        "b_mean": b_mean, "b_std": b_std,                 # (3,)
        "branch_mean": branch_mean, "branch_std": branch_std,  # (I,)
        "coord_mean": coord_mean, "coord_std": coord_std,      # (3,)
        "sensor_xyz": R_sens,                                  # (I, 3)
    }


class SampleDataset(Dataset):
    """Yields batches `(branch (B, I), target (B, J, 3))` — formato CartesianProd.

    El trunk (coords de grilla) NO va acá: es compartido entre todas las
    muestras y vive como buffer del modelo. El batching es directo sobre
    muestras (configs de imanes), sin el gather sample×punto de v5.
    """

    def __init__(self, h5_path, sample_indices, *,
                 batch_size,
                 b_mean, b_std, branch_mean, branch_std,
                 data_device="cpu"):
        self.batch_size = int(batch_size)
        self.device     = torch.device(data_device)

        b_mean_a = np.asarray(b_mean, dtype=np.float32).reshape(3)
        b_std_a  = np.asarray(b_std,  dtype=np.float32).reshape(3)
        br_mean  = np.asarray(branch_mean, dtype=np.float32).reshape(1, -1)
        br_std   = np.asarray(branch_std,  dtype=np.float32).reshape(1, -1)

        sorted_idx = np.sort(np.asarray(sample_indices, dtype=np.int64))
        with h5py.File(h5_path, "r") as f:
            B_sens_arr = f["geometria/sens/B"][sorted_idx][..., 1].astype(np.float32)  # (n, I) By
            B_grid_arr = f["geometria/grid/B"][sorted_idx].astype(np.float32)          # (n, J, 3)

        branch_norm = (B_sens_arr - br_mean) / br_std                                  # (n, I)
        B_grid_norm = ((B_grid_arr - b_mean_a) / b_std_a).astype(np.float32)           # (n, J, 3)

        self.branch   = torch.from_numpy(branch_norm).to(self.device)                  # (n, I)
        self.target_n = torch.from_numpy(B_grid_norm).to(self.device)                  # (n, J, 3)
        self.N = self.branch.shape[0]
        self.I = self.branch.shape[1]
        self.J = self.target_n.shape[1]

    def __len__(self):
        return (self.N + self.batch_size - 1) // self.batch_size

    def __getitem__(self, batch_idx):
        start = batch_idx * self.batch_size
        end   = min(start + self.batch_size, self.N)
        return self.branch[start:end], self.target_n[start:end]      # (B,I), (B,J,3)


def make_loader(h5_path, sample_indices, *,
                shuffle, batch_size,
                b_mean, b_std, branch_mean, branch_std,
                data_device="cpu", num_workers=0, pin_memory=False):
    ds = SampleDataset(h5_path, sample_indices,
                       batch_size=batch_size,
                       b_mean=b_mean, b_std=b_std,
                       branch_mean=branch_mean, branch_std=branch_std,
                       data_device=data_device)
    # DeepXDE (backend pytorch + cuda) fija el default device en cuda; el
    # RandomSampler de shuffle hace torch.randperm con generator CPU y choca.
    # Pasamos un generator explícito que matchee el device de los datos.
    gen = None
    if torch.device(data_device).type == "cuda":
        num_workers = 0
        pin_memory  = False
        gen = torch.Generator(device="cuda")
    return DataLoader(
        ds, batch_size=None, shuffle=shuffle, generator=gen,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def load_full_grid_for_sample(h5_path, sample_idx):
    """Devuelve `(branch_by (I,), R_grid (J, 3), b_full_mt (J, 3))` raw (sin normalizar)."""
    h5_path = Path(h5_path)
    sample_idx = int(sample_idx)
    with h5py.File(h5_path, "r") as f:
        sensors = f["geometria/sens/B"][sample_idx][..., 1].astype(np.float32)    # (I,)
        R_grid  = f["geometria/grid/R"][:].astype(np.float32)                      # (J, 3)
        b_full  = f["geometria/grid/B"][sample_idx].astype(np.float32)            # (J, 3)
    return (torch.from_numpy(sensors), torch.from_numpy(R_grid), torch.from_numpy(b_full))
