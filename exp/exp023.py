"""exp023

exp022のバックアップ

* Self-Supervised Learning by lightly
* Simsiam
* epoch 400
* batch size 320
* backbone is vit small

* https://arxiv.org/abs/2011.10566
* https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simsiam/simsiam_module.py#L19-L268
* https://www.guruguru.science/competitions/17/discussions/a39d588e-aff2-4728-8323-b07f15563552/
* https://speakerdeck.com/sansandsoc/simsiam-exploring-simple-siamese-representation-learning

"""
# ====================
# package
# ====================
import gc
import os
import sys
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import albumentations as A
import cv2
import lightly
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from albumentations.pytorch import ToTensorV2
from loguru import logger
from madgrad import MADGRAD
from PIL import Image
from pl_bolts.models.self_supervised import SimSiam
from pl_bolts.models.self_supervised.simclr import (SimCLREvalDataTransform,
                                                    SimCLRTrainDataTransform)
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.callback_hook import TrainerCallbackHookMixin
from sklearn import model_selection
from torch._C import dtype
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet

sys.path.append(os.getcwd())
from src import utils

# from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    is_logging: bool = True
    debug: bool = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers: int = 4
    seed: int = 42
    base_dir: Path = Path(".")
    data_dir: Path = base_dir / "input" / "atmacup-11"
    train_csv_path: Path = data_dir / "train.csv"
    test_csv_path: Path = data_dir / "test.csv"
    output_dir: Path = base_dir

    # training strategy
    target_col: str = "target"
    fp16: bool = True
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 64
    epochs: int = 400
    fold: int = 0
    n_fold: int = 5
    lr: float = 0.05 * batch_size / 256
    T_max: int = epochs

    # model
    # ssl by lightly
    # dimension of the embedding
    outdim: int = 512
    backbone_model_name: str = "vit_small_patch16_224"
    num_ftrs = 75264
    num_mlp_layers = 2


class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.test_path = cfg.test_csv_path
        self.train_path = cfg.train_csv_path
        self.target_col = cfg.target_col
        self.data_dir = cfg.data_dir

        self.fold = cfg.fold
        self.n_fold = cfg.n_fold
        self.epochs = cfg.epochs
        self.seed = cfg.seed
        self.img_size = cfg.img_size
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers

        # define only
        self.train_df = None
        self.valid_df = None
        self.test_df = None

    def get_test_df(self) -> pd.core.frame.DataFrame:
        test_df = pd.read_csv(self.test_path)
        return test_df

    def create_data(self):
        train = pd.read_csv(self.train_path)
        train["fold"] = -1
        # split
        kf = model_selection.StratifiedGroupKFold(
            n_splits=self.n_fold, shuffle=True, random_state=self.seed
        )
        groups = train["art_series_id"].values
        for fold_i, (train_idx, valid_idx) in enumerate(
            kf.split(train, train[self.target_col], groups)
        ):
            train.loc[valid_idx, "fold"] = int(fold_i)
        train["fold"] = train["fold"].astype(int)
        return train

    def split_train_valid_df(self, train: pd.core.frame.DataFrame, fold: int):
        fold = int(self.fold)
        train_df = train[train["fold"] != fold].reset_index(drop=True)
        valid_df = train[train["fold"] == fold].reset_index(drop=True)
        return train_df, valid_df

    def setup(self, stage):
        self.test_df = self.get_test_df()
        train = self.create_data()
        # train_df, valid_df = self.split_train_valid_df(train, self.fold)
        self.train_df = train
        self.valid_df = None

    def get_df(self, phase: str) -> pd.core.frame.DataFrame:
        assert phase in {"train", "valid", "test"}
        if phase == "train":
            return self.train_df
        elif phase == "valid":
            return self.valid_df
        elif phase == "test":
            return self.test_df

    def get_dataset(self, phase: str):
        if phase == "train":
            dataset_simsiam = lightly.data.LightlyDataset(
                input_dir=self.data_dir / "image",
            )
        else:
            dataset_simsiam = lightly.data.LightlyDataset(
                input_dir=self.data_dir / "image", transform=self.get_test_transforms()
            )
        return dataset_simsiam

    def get_loader(self, phase: str):
        dataset = self.get_dataset(phase=phase)
        if phase == "train":
            return DataLoader(
                dataset,
                collate_fn=self.get_train_collate_fn(),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
            )

    def train_dataloader(self):
        return self.get_loader(phase="train")

    def val_dataloader(self):
        return self.get_loader(phase="valid")

    def test_dataloader(self):
        return self.get_loader(phase="test")

    def get_test_transforms(self):
        test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size[0], self.img_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=lightly.data.collate.imagenet_normalize["mean"],
                    std=lightly.data.collate.imagenet_normalize["std"],
                ),
            ]
        )
        return test_transforms

    def get_train_collate_fn(self):
        # define the augmentations for self-supervised learning
        collate_fn = lightly.data.ImageCollateFunction(
            input_size=self.img_size[0],
            # require invariance to flips and rotations
            hf_prob=0.5,
            vf_prob=0.5,
            rr_prob=0.5,
            # satellite images are all taken from the same height
            # so we use only slight random cropping
            min_scale=0.5,
            # use a weak color jitter for invariance w.r.t small color changes
            cj_prob=0.2,
            cj_bright=0.1,
            cj_contrast=0.1,
            cj_hue=0.1,
            cj_sat=0.1,
        )
        return collate_fn


class MyLitModel(pl.LightningModule):
    def __init__(self, cfg: Config) -> None:
        super(MyLitModel, self).__init__()
        # set hyperparams
        self.backbone_model_name = cfg.backbone_model_name
        self.num_ftrs = cfg.num_ftrs
        self.lr = cfg.lr
        self.T_max = cfg.T_max

        # prepare train
        self.avg_loss = 0.0
        self.avg_output_std = 0.0
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
        # prepare network
        self.out_dim = cfg.outdim
        backbone_model = timm.create_model(self.backbone_model_name, pretrained=False)
        self.backbone = nn.Sequential(
            *list(backbone_model.children())[:-1]
        )  # drop last fc layer
        self.model = lightly.models.SimSiam(
            self.backbone,
            num_ftrs=self.num_ftrs,
        )

        self.save_hyperparameters()

    def forward(self, x0, x1):
        return self.model(x0, x1)

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_max
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        sample, _, _ = batch
        # x0: (batch_size, n_channels, w, h)
        # x1: (batch_size, n_channels, w, h)
        x0, x1 = sample

        # y0 : ((batch_size, 2048),(batch_size, 2048))
        # y1 : ((batch_size, 2048),(batch_size, 2048))
        y0, y1 = self.model(x0, x1)
        loss = self.criterion(y0, y1)

        output, _ = y0
        output = output.detach()
        output = torch.nn.functional.normalize(output, dim=1)

        output_std = torch.std(output, 0)
        output_std = output_std.mean()

        # moving average
        w = 0.9
        self.avg_loss = w * self.avg_loss + (1 - w) * loss.item()
        self.avg_output_std = w * self.avg_output_std + (1 - w) * output_std.item()

        self.log_dict({"train_loss": loss}, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs) -> None:
        collapse_level = max(0.0, 1 - np.sqrt(self.out_dim) * self.avg_output_std)
        self.log_dict({"Collapse Level": collapse_level}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        return None


def main():
    config = Config()
    config.is_logging = True
    config.debug = False

    seed_everything(config.seed)
    secret_cfg = utils.get_config()
    runfilepath = Path(__file__)
    expname = runfilepath.name.split(".")[0]
    base_dir = Path(".")
    output_dir = base_dir / "output" / f"{expname}"
    output_dir.mkdir(exist_ok=True)
    config.output_dir = output_dir
    logger.info(f"project: {secret_cfg.prj_name}, name: {secret_cfg.entity}")
    if config.is_logging:
        # print("login skipped")
        wandb.login(key=secret_cfg.wandb_token)

    logger.info(f"Settingns:\n\t{asdict(config)}")
    wandb_logger = WandbLogger(
        name=expname,
        save_dir=str(config.output_dir),
        project=secret_cfg.prj_name,
        entity=secret_cfg.entity,
        log_model=False,
    )
    loggers = [wandb_logger]
    trainer = Trainer(
        max_epochs=config.epochs if not config.debug else 1,
        precision=16 if config.fp16 else 32,
        accumulate_grad_batches=8,
        amp_backend="native",
        gpus=1,
        benchmark=False,
        default_root_dir=os.getcwd(),
        deterministic=True,
        limit_train_batches=0.1 if config.debug else 1.0,
        # limit_val_batches=0.0,
        callbacks=None,
        logger=loggers,
        # check_val_every_n_epoch=0,
    )
    lit_module = MyLitModel(cfg=config)
    data_module = MyDataModule(cfg=Config)
    trainer.fit(model=lit_module, datamodule=data_module)
    trainer.save_checkpoint(f"./output/{expname}/{expname}_epoch={config.epochs}.ckpt")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
