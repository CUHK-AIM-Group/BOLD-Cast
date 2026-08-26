import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class SubjectNPZDataset(Dataset):
    """Stage-I subject-level dataset backed directly by Stage-II NPZ files.

    Each NPZ is expected to contain at least ``fMRI``, ``text`` and ``corr``.
    Stage I only consumes ``fMRI`` and ``corr``. Other keys are left untouched
    and are preserved when ``com`` and ``priv`` are written back later.
    """

    def __init__(self, split_dir, time_len, num_rois, normalize=True):
        self.split_dir = Path(split_dir)
        self.time_len = int(time_len)
        self.num_rois = int(num_rois)
        self.normalize = normalize

        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Stage-I split directory not found: {self.split_dir}")

        self.files = sorted(self.split_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz subject files found in: {self.split_dir}")

    @staticmethod
    def _zscore_per_roi(fmri):
        epsilon = 1e-8
        mean_vals = np.mean(fmri, axis=1, keepdims=True)
        std_vals = np.std(fmri, axis=1, keepdims=True)
        return (fmri - mean_vals) / (std_vals + epsilon)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        with np.load(file_path, allow_pickle=True) as data:
            missing = [key for key in ("fMRI", "corr") if key not in data.files]
            if missing:
                raise KeyError(f"{file_path} is missing required key(s): {missing}")

            fmri = np.asarray(data["fMRI"], dtype=np.float32)
            corr = np.asarray(data["corr"], dtype=np.float32)

        if fmri.ndim != 2:
            raise ValueError(f"fMRI in {file_path} must be 2-D [ROI, time], got {fmri.shape}")
        if fmri.shape[0] != self.num_rois:
            raise ValueError(
                f"fMRI ROI mismatch in {file_path}: expected {self.num_rois}, got {fmri.shape[0]}"
            )
        if fmri.shape[1] < self.time_len:
            raise ValueError(
                f"fMRI in {file_path} has only {fmri.shape[1]} time points; "
                f"Stage I requires at least {self.time_len}."
            )
        if corr.shape != (self.num_rois, self.num_rois):
            raise ValueError(
                f"corr shape mismatch in {file_path}: expected "
                f"({self.num_rois}, {self.num_rois}), got {corr.shape}"
            )

        if self.normalize:
            fmri = self._zscore_per_roi(fmri)

        # Self-reconstruction only. No following/future segment is used.
        x = fmri[:, : self.time_len].astype(np.float32, copy=False)
        y = x.copy()

        # Preserve the original Stage-I behavior of treating NaNs in adjacency
        # as invalid values handled later by the model.
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(corr),
            str(file_path),
        )


def _pad_subject_batch(batch, batch_size):
    """Pad the final batch by repeating its last subject.

    The original GNNDAE creates one encoder/decoder per batch position and
    therefore requires a fixed number of views. Padding keeps this architecture
    unchanged while ensuring no real subject is dropped.
    """
    real_count = len(batch)
    if real_count == 0:
        raise RuntimeError("Received an empty Stage-I batch.")
    if real_count > batch_size:
        raise ValueError(f"Batch contains {real_count} items, expected <= {batch_size}.")

    padded = list(batch)
    while len(padded) < batch_size:
        padded.append(batch[-1])

    xs, ys, adjs, paths = zip(*padded)
    return (
        torch.stack(xs, dim=0),
        torch.stack(ys, dim=0),
        torch.stack(adjs, dim=0),
        list(paths),
        real_count,
    )


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=lambda batch: _pad_subject_batch(batch, batch_size),
    )


class embedder:
    """Data/device setup shared by Stage-I GDA.

    Training uses the union of the original ``train`` and ``test`` splits.
    The original ``val`` split is kept exclusively for model selection.
    Latent extraction later revisits train/val/test separately.
    """

    def __init__(self, args):
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == -1:
            args.device = torch.device("cpu")
        else:
            args.device = torch.device(
                f"cuda:{args.gpu_num_}" if torch.cuda.is_available() else "cpu"
            )

        args.root_path = os.path.abspath(args.root_path)
        train_dir = os.path.join(args.root_path, "train")
        val_dir = os.path.join(args.root_path, "val")
        test_dir = os.path.join(args.root_path, "test")

        self.train_split_dataset = SubjectNPZDataset(
            train_dir, args.time_len, args.num_rois, normalize=args.normalize
        )
        self.val_split_dataset = SubjectNPZDataset(
            val_dir, args.time_len, args.num_rois, normalize=args.normalize
        )
        self.test_split_dataset = SubjectNPZDataset(
            test_dir, args.time_len, args.num_rois, normalize=args.normalize
        )

        # Stage I is unsupervised/self-reconstructive: use train + test for
        # representation learning, while val remains untouched for selection.
        train_dataset = ConcatDataset([self.train_split_dataset, self.test_split_dataset])

        self.data_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
        self.val_data_loader = make_loader(
            self.val_split_dataset, args.batch_size, shuffle=False
        )

        # Separate deterministic loaders are used only after training to write
        # latent representations back to the original NPZ files.
        self.latent_loaders = {
            "train": make_loader(self.train_split_dataset, args.batch_size, shuffle=False),
            "val": make_loader(self.val_split_dataset, args.batch_size, shuffle=False),
            "test": make_loader(self.test_split_dataset, args.batch_size, shuffle=False),
        }

        args.ft_size = args.time_len
        args.nb_nodes = args.num_rois
        # Neighbor sampling spans ROI nodes. Keep it tied to the atlas size.
        args.neighbor_num = args.num_rois

        print(
            f"Stage I data | dataset={args.dataset} atlas={args.atlas} "
            f"ROIs={args.num_rois} time_len={args.time_len}"
        )
        print(
            f"Subjects | train={len(self.train_split_dataset)} "
            f"test={len(self.test_split_dataset)} "
            f"(training total={len(train_dataset)}), val={len(self.val_split_dataset)}"
        )
