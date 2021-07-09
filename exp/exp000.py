""" exp000

"""
# ====================
# package
# ====================
import gc
import os
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn import model_selection
from torch.utils.data import DataLoader, Dataset

from exp import utils

# from torch.utils.data import DataLoader, Dataset


@dataclass
class OptimizerCfg:
    name: str = "AdamW"
    params: Dict[str, Any] = {}


@dataclass
class SchedulerCfg:
    name: str = "CosinAnnealingLR"
    params: Dict[str, Any] = {}


@dataclass
class OptimConfig:
    opmizer = OptimizerCfg()
    scheduler = SchedulerCfg()


@dataclass
class Config:
    # Meta
    is_logging: bool = False
    debug: bool = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers: int = os.cpu_count()
    seed: int = 42
    base_dir: Path = Path(".")
    data_dir: Path = base_dir / "input" / "atmacup-11"
    train_csv_path = data_dir / "train.csv"
    test_csv_path = data_dir / "test.csv"

    # Train strategy
    target_col: str = 
    fold: int = 0
    n_fold: int = 5
    epochs: int = 5
    batch_size: int = 8
    fp16: bool = True

    optim = OptimConfig()


# =====================
# Dataset
# =====================
class Atma11Dataset(Dataset):
    def __init__(self, df: pd.core.frame.DataFrame, is_test: bool = False) -> None:
        self.df = df
        self.is_test = is_test

    def __len__(self) -> None:
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, Any]:
        pass


class MyDataModule(pl.LightingDataModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.train_df = None
        self.valid_df = None
        self.test_df = None

    def get_test_df(self) -> pd.core.frame.DataFrame:
        return pd.read_csv(self.test_csv_path)

    def create_data(self):
        train = pd.read_csv(self.cfg.train_csv_path)
        train["fold"] = -1
        # split
        kf = model_selection.StraitifiedKfold(n_splits=self.n_fold, shuffle=True, random_state=self.cfg.seed)
        for fold_i, (train_idx, valid_idx) in enumerate(kf.split(train, train[self.cfg.target_col])):
            train.loc[valid_idx, "fold"] = int(fold_i)
        train["fold"] = train["fold"].astype(int)
        return train

    def split_train_valid_df(self, train: pd.core.frame.DataFrame):
        fold = int(self.cfg.fold)
        train_df = train[train["fold"] != fold].reset_index(drop=True)
        valid_df = train[train["fold"] == fold].reset_index(drop=True)
        return train_df, valid_df

    def setup(self, stage):
        self.test_df = self.get_test_df()
        train = self.create_data()
        train_df, valid_df = self.split_train_valid_df(train, self.cfg.fold)
        self.train_df = train_df
        self.valid_df = valid_df

    def get_df(self, phase: str) -> pd.core.frame.DataFrame:
        assert phase in {"train", "valid", "test"}
        if phase == "train":
            return self.train_df
        elif phase == "valid":
            return self.valid_df
        elif phase == "test":
            return self.test_df

    def get_dataset(self, phase: str):
        dataset = Atma11Dataset(
            df=self.get_df(phase=phase),
            is_test=(phase == "test"),
        )
        return dataset

    def get_loader(self, phase: str):
        dataset = self.get_dataset(phase=phase)
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True if phase == "train" else False,
            num_workers=self.cfg.num_workers,
            drop_last=True if phase == "train" else False
        )

    def train_dataloader(self):
        return self.get_loader(phase="train")

    def valid_dataloader(self):
        return self.get_loader(phase="valid")

    def test_dataloader(self):
        return self.get_loader(phase="test")


# =====================
# Model
# =====================
class Atma11Model(nn.Module):
    def __init__(self, config: Config) -> None:
        self.config = config

    def forward(self) -> torch.Tensor:
        pass


class MyLightningModule(pl.LightingModule):
    def __init__(self, cfg: Config, optimizer: optim.Optimizer) -> None:
        super(MyLightningModule, self).__init__()
        self.cfg = cfg
        self.net = Atma11Model(config=cfg)

    def forward(self, x) -> torch.Tensor:
        logits = self.net(x).squueze(1)
        return logits

    def training_step(self, batch: Dict[str, torch.Tensor]):
        data, targets = batch["data"], batch["targets"]
        logits = self.forward(data)
        loss = self.loss(logits, targets)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor]):
        # data = images
        data, targets = batch["data"], batch["targets"]
        logits = self.forward(data)
        loss = self.loss(logits, targets)
        self.log("val_loss", loss)
        output = OrderedDict(
            {
                "targets": targets.detach(),
                "preds": logits.detach(),
                "loss": loss.detach(),
            }
        )
        return output

    def validation_epoch_end(self, outputs):
        result_dict = dict()
        result_dict["epoch"] = int(self.current_epoch)
        result_dict["v_loss"] = (
            torch.stack([out["loss"] for out in outputs]).mead().item()
        )

        targets = torch.cat([out["targets"].view(-1) for out in outputs]).cpu().numpy()
        preds = torch.cat([out["pred"].view(-1) for out in outputs]).cpu().numpy()

        result_dict["v_score"] = torch.sqrt(self.loss(preds, targets))
        self.log_dict(result_dict, prog_bar=True)

    def configure_optimizers(self):
        cfg_optim = self.config.optim
        optimizer = make_optimizer(
            params=self.parameters(),
            name=cfg_optim.optimizer.name,
            **cfg_optim.optimizer.params,
        )
        scheduler = make_scheduler(
            optimizer, name=cfg_optim.scheduler.name, **cfg_optim.scheduler.params
        )
        return [optimizer], [scheduler]


# =====================
# Utils
# =====================
def make_loader(batch_size: int, fold: int) -> Tuple[DataLoader, DataLoader]:
    num_workers: int = os.cpu_count()
    train_dataset = Atma11Dataset()
    valid_dataset = Atma11Dataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, valid_loader


def make_model(config) -> Atma11Model:
    model = Atma11Model(config=config)
    return model


def make_optimizer(name: str, model_params, params: Dict[str, Any]) -> optim.Optimizer:
    optimizer = getattr(optim, name)(params=model_params, **params)
    return optimizer


def make_scheduler(
    optimizer: optim.Optimizer, name: str, params: Dict[str, Any]
) -> lr_scheduler.Scheduler:
    schduler = getattr(lr_scheduler, name)(optimizer, **params)
    return schduler


def run(
    expname: str,
    fold: int,
    secret_cfg: Dict[str, Any],
    config: Config,
    output_dir: Path,
) -> Dict[str, Any]:
    seed_everything(config.seed)
    config.fold = fold

    loggers = None
    if config.is_logging:
        wandb.login(key=secret_cfg.wandb_token)
        wandb_logger = WandbLogger(
            name=expname,
            save_dir=config.output_dir,
            project=secret_cfg.prj_name,
            entity=secret_cfg.entity,
            log_model=False,
        )
        loggers = [wandb_logger]

    checkpoint = ModelCheckpoint(
        dirpath=str(output_dir),
        filename=f"{expname}_fold{fold}" + "{epoch:02d}",
        save_weights_only=True,
        save_top_k=None,
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(
        max_epochs=config.epochs if config.debug else 1,
        precision=16 if config.fp16 else 32,
        amp_backend=True,
        benchmark=False,
        default_root_dir=os.getcwd(),
        deterministic=True,
        limit_train_batches=0.02 if config.debug else 1.0,
        limit_val_batches=0.05 if config.debug else 1.0,
        callbacks=[checkpoint],
        logger=loggers,
    )

    # training
    lightning_module = MyLightningModule(config)
    data_module = MyDataModule(config)
    trainer.fit(lightning_module, datamodule=data_module)


def main():
    config = Config()
    secret_cfg = utils.get_config()
    runfilepath = Path(__file__)
    expname = runfilepath.name.split(".")[0]
    base_dir = Path(".")
    output_dir = base_dir / "output" / "expname"
    output_dir.mkdir(exist_ok=True)
    print(secret_cfg.prj_name, secret_cfg.entity)
    logger.info(f"Settingns:\n\t{asdict(config)}")

    for fold in config.n_fold:
        print(f" ==== Fold {fold} start ==== ")
        run(
            expname=expname,
            secret_cfg=secret_cfg,
            output_dir=output_dir,
            config=config,
            fold=fold
        )


if __name__ == "__main__":
    main()
