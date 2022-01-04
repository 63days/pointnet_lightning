from typing import Optional, Tuple
import os
import os.path as osp
import h5py
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms


def pc_normalize(pc):
    l = pc.shape[0]
    c = np.mean(pc, axis=0)
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class ModelNet40Dataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        super().__init__()
        # Load data
        modelnet_dir = osp.join(data_dir, "modelnet40_ply_hdf5_2048")
        with open(osp.join(modelnet_dir, f"{split}_files.txt")) as f:
            hdf5_list = [
                osp.join(modelnet_dir, line.rstrip()) for line in f.readlines()
            ]

        Data, Label = [], []
        for hdf5_path in hdf5_list:
            hdf5 = h5py.File(hdf5_path, "r")
            Data.append(hdf5["data"][:])
            Label.append(hdf5["label"][:])

        self.data = np.concatenate(Data, axis=0).astype(np.float32)
        self.label = np.concatenate(Label, axis=0).astype(np.int64).reshape(-1)

    def __getitem__(self, idx):
        return pc_normalize(self.data[idx]), self.label[idx]

    def __len__(self):
        return len(self.data)


class ModelNet40DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        num_classes: int = 40
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed."""
        DATA_DIR = self.hparams.data_dir
        if not osp.exists(DATA_DIR):
            print(f"{DATA_DIR} does not exist")
            os.makedirs(DATA_DIR)

        if not osp.exists(osp.join(DATA_DIR, "modelnet40_ply_hdf5_2048")):
            www = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
            zipfile = osp.basename(www)
            os.system(f"wget --no-check-certificate {www}; unzip {zipfile}")
            os.system(f"mv {zipfile[:-4]} DATA_DIR")
            os.system(f"rm {zipfile}")

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ModelNet40Dataset(self.hparams.data_dir, split="train")
            self.data_val = ModelNet40Dataset(self.hparams.data_dir, split="test")
            self.data_test = ModelNet40Dataset(self.hparams.data_dir, split="test")

    def _build_dataloader(self, split: str):
        return DataLoader(
            dataset=getattr(self, "data_" + split),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=False,
            shuffle=split == "train",
        )

    def train_dataloader(self):
        return self._build_dataloader("train")

    def val_dataloader(self):
        return self._build_dataloader("val")

    def test_dataloader(self):
        return self._build_dataloader("test")
