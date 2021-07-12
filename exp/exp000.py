""" exp000

* base line
"""
# ====================
# package
# ====================
import gc
import os
import sys
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
from madgrad import MADGRAD
from PIL import Image
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn import model_selection
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet

sys.path.append(os.getcwd())
import utils

# from torch.utils.data import DataLoader, Dataset


@dataclass
class OptimizerCfg:
    name: str
    params: Dict[str, Any]


@dataclass
class SchedulerCfg:
    name: str
    params: Dict[str, Any]


@dataclass
class OptimConfig:
    optimizer: OptimizerCfg
    scheduler: SchedulerCfg


@dataclass
class Config:
    # Meta
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

    # model params
    model_name = "resnet18"
    models_params = None

    # Train strategy
    target_col: str = "target"
    num_labels: int = 1
    fold: int = 0
    n_fold: int = 5
    epochs: int = 20
    batch_size: int = 8 * 8 * 2
    fp16: bool = False
    img_size: Tuple[int, int] = (224, 224)

    optim = OptimConfig(
        optimizer=OptimizerCfg(
            name="AdamW",
            params={"lr": 1e-3},
            # name="MADGRAD",
            # params={"lr": 1e-3, "eps": 1e-6, "weight_decay": 5e-4},
        ),
        scheduler=SchedulerCfg(
            name="CosineAnnealingLR",
            params={
                "T_max": epochs,
            },
        ),
    )


# =====================
# Dataset
# =====================
class Atma11Dataset(Dataset):
    def __init__(
        self,
        cfg: Config,
        df: pd.core.frame.DataFrame,
        img_size: Tuple[int, int],
        is_test: bool = False,
    ) -> None:
        self.data_dir = Path(".") / "input" / "atmacup-11"
        self.cfg = cfg
        self.size = img_size
        self.df = df
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, Any]:
        img_id = self.df["object_id"][idx]
        image = self.get_image(img_id + ".jpg").to(dtype=torch.float32)
        """
        logger.debug(
            f"shape of labels : {labels.detach().cpu().shape}\n"
            + f"shape of image {image.detach().cpu().shape}"
        )
        """
        if self.is_test:
            return {"object_id": img_id, "image": image}
        else:
            labels = torch.tensor(self.df["target"][idx], dtype=torch.float32)
            return {
                "object_id": img_id,
                "target": labels,
                "image": image,
            }

    def get_image(self, img_path):
        img = Image.open(self.data_dir / "image" / img_path)
        img = img.convert("RGB")
        # img = img.resize(self.size)
        # img = np.array(img) / 255.0
        img: torch.Tensor = self.get_preprocess()(img)
        return img

    def get_preprocess(self):
        return transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )


class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.train_df = None
        self.valid_df = None
        self.test_df = None

    def get_test_df(self) -> pd.core.frame.DataFrame:
        test_df = pd.read_csv(self.cfg.test_csv_path)
        return test_df

    def create_data(self):
        train = pd.read_csv(self.cfg.train_csv_path)
        train["fold"] = -1
        # split
        kf = model_selection.StratifiedKFold(
            n_splits=self.cfg.n_fold, shuffle=True, random_state=self.cfg.seed
        )
        for fold_i, (train_idx, valid_idx) in enumerate(
            kf.split(train, train[self.cfg.target_col])
        ):
            train.loc[valid_idx, "fold"] = int(fold_i)
        train["fold"] = train["fold"].astype(int)
        return train

    def split_train_valid_df(self, train: pd.core.frame.DataFrame, fold: int):
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
            cfg=self.cfg,
            is_test=(phase == "test"),
            img_size=self.cfg.img_size,
        )
        return dataset

    def get_loader(self, phase: str):
        dataset = self.get_dataset(phase=phase)
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True if phase == "train" else False,
            num_workers=self.cfg.num_workers,
            drop_last=True if phase == "train" else False,
        )

    def train_dataloader(self):
        return self.get_loader(phase="train")

    def val_dataloader(self):
        return self.get_loader(phase="valid")

    def test_dataloader(self):
        return self.get_loader(phase="test")


# =====================
# Model
# =====================
class Atma11Model(nn.Module):
    def __init__(self, config: Config) -> None:
        super(Atma11Model, self).__init__()
        self.config = config
        logger.info(f"Loading ... {config.model_name}")
        model = getattr(resnet, config.model_name)(pretrained=False)
        model.fc = nn.Linear(512, out_features=config.num_labels, bias=True)
        for param in model.parameters():
            param.require_grad = True
        self.model = model

    def forward(self, x) -> torch.Tensor:
        # shape: (batch_size, channel, w, h)
        output = self.model(x)
        # logger.debug(f"{x.detach().cpu().shape}")
        return output.view(-1)


class MyLightningModule(pl.LightningModule):
    def __init__(self, cfg: Config) -> None:
        super(MyLightningModule, self).__init__()
        self.cfg = cfg
        self.net = Atma11Model(config=cfg)
        self.criterion = torch.nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        logits = self.net(x)
        return logits

    def training_step(self, batch: Dict[str, Any], batch_nb: int):
        # logger.debug(f"batch_nb is {batch_nb}")
        data = batch["image"]
        targets = batch["target"]
        logits = self(data)
        loss = torch.sqrt(self.criterion(logits, targets))
        self.log(f"fold{self.cfg.fold}_train_loss", loss)
        wandb.log({f"fold{self.cfg.fold}_train_loss": loss.detach().cpu()})
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_nb: int
    ) -> OrderedDict:
        # data = images
        data, targets = batch["image"], batch["target"]
        logits = self.forward(data)
        loss = torch.sqrt(self.criterion(logits, targets))
        self.log(f"fold{self.cfg.fold}_val_loss", loss)
        wandb.log({f"fold{self.cfg.fold}_valid_loss": loss.detach().cpu()})
        output = OrderedDict(
            {
                "targets": targets.detach(),
                "preds": logits.detach(),
                "loss": loss.detach(),
            }
        )
        return output

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int
    ) -> OrderedDict:
        data = batch["image"]
        object_id = batch["object_id"]
        output = self(data)
        return OrderedDict({"preds": output, "object_id": object_id})

    def validation_epoch_end(self, outputs):
        result_dict = dict()
        result_dict["epoch"] = int(self.current_epoch)
        result_dict[f"fold{self.cfg.fold}_val_loss"] = (
            torch.stack([out["loss"] for out in outputs]).mean().item()
        )

        targets = (
            torch.cat([out["targets"].view(-1) for out in outputs])
            .detach()
            .cpu()
            .numpy()
        )
        preds = (
            torch.cat([out["preds"].view(-1) for out in outputs]).detach().cpu().numpy()
        )

        # logger.debug(f"targets: {targets.shape}, preds: {preds.shape}")
        # calc score
        result_dict[f"fold{self.cfg.fold}_v_score"] = score_func(
            preds=preds, targets=targets
        )
        self.log_dict(result_dict, prog_bar=True)

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        cfg_optim = self.cfg.optim
        optimizer = make_optimizer(
            model_params=self.parameters(),
            name=cfg_optim.optimizer.name,
            params=cfg_optim.optimizer.params,
        )
        scheduler = make_scheduler(
            optimizer, name=cfg_optim.scheduler.name, params=cfg_optim.scheduler.params
        )
        return [optimizer], [scheduler]


# =====================
# Utils
# =====================
def score_func(preds: np.ndarray, targets: np.ndarray) -> np.float64:
    assert targets.shape == preds.shape
    return np.sqrt((np.sum((preds - targets) ** 2) / len(preds)))


def make_optimizer(name: str, model_params, params: Dict[str, Any]) -> optim.Optimizer:
    logger.info(f"{name}, with {params}")
    if name == "MADGRAD":
        optimizer = MADGRAD(model_params, **params)
    else:
        optimizer = getattr(optim, name)(params=model_params, **params)
    return optimizer


def make_scheduler(optimizer: optim.Optimizer, name: str, params: Dict[str, Any]):
    logger.info(f"{name} with {params}")
    schduler = getattr(lr_scheduler, name)(optimizer, **params)
    return schduler


def make_sub(preds_df: pd.core.frame.DataFrame, n_fold: int, output_dir: str) -> None:
    sample_path = "./input/atmacup-11/atmaCup#11_sample_submission.csv"
    sub = pd.read_csv(sample_path)

    # mean average
    sub["target"] = (
        np.sum(preds_df[f"fold{fold}_preds"].values for fold in range(n_fold)) / n_fold
    )

    logger.info(
        "\n\tSub Summary\n"
        + "\n\tHead\n\n"
        + f"{sub.head(5)}\n"
        + "\n\tDescribe\n\n"
        + f"{sub.describe()}\n"
        + "\n\tInfo\n\n"
        + f"{sub.info()}"
    )
    sub.to_csv(f"{output_dir}/submission.csv", index=False)


def make_predict(path: str, config: Config) -> Dict[str, np.ndarray]:
    """

    TODO:
        * hackしてる感じがあるので正しく設定して正しく使えるようにしておく(setupとか)
    """
    logger.info("predict start")
    logger.info(f"Loading... {path}")
    lightning_module = MyLightningModule.load_from_checkpoint(checkpoint_path=path)
    data_module = MyDataModule(config)
    data_module.setup(stage="predict")
    # predict for test dataset
    trainer = Trainer(
        max_epochs=config.epochs if not config.debug else 1,
        precision=16 if config.fp16 else 32,
        accumulate_grad_batches=1,
        amp_backend="native",
        gpus=1,
        benchmark=False,
        default_root_dir=os.getcwd(),
        deterministic=True,
    )

    test_loader = data_module.get_loader(phase="test")
    # shape: (num_batch, batch_size), type: List[torch.Tensor]
    outputs: List[OrderedDict] = trainer.predict(
        model=lightning_module,
        dataloaders=test_loader,
        # datamodule=data_module,
        return_predictions=True,
    )

    preds = np.concatenate(
        [output["preds"].detach().cpu().numpy() for output in outputs]
    )
    ids = np.concatenate([output["object_id"] for output in outputs])

    test_df_length = data_module.get_df(phase="test").shape[0]
    assert len(preds) == test_df_length, f"{len(preds)} != {test_df_length}"

    return {
        "object_id": ids,
        "preds": preds,
    }


def train(
    expname: str,
    fold: int,
    secret_cfg,
    config: Config,
    output_dir: Path,
) -> None:
    seed_everything(config.seed)
    config.fold = fold

    wandb_logger = WandbLogger(
        name=expname,
        save_dir=str(config.output_dir),
        project=secret_cfg.prj_name,
        entity=secret_cfg.entity,
        log_model=False,
    )
    loggers = [wandb_logger]

    checkpoint = ModelCheckpoint(
        dirpath=str(output_dir),
        filename=f"{expname}_fold{fold}" + "{epoch:02d}",
        save_weights_only=True,
        save_top_k=1,
        monitor=f"fold{fold}_v_score",
        mode="min",
    )

    trainer = Trainer(
        max_epochs=config.epochs if not config.debug else 1,
        precision=16 if config.fp16 else 32,
        accumulate_grad_batches=1,
        amp_backend="native",
        gpus=1,
        benchmark=False,
        default_root_dir=os.getcwd(),
        deterministic=True,
        limit_train_batches=0.1 if config.debug else 1.0,
        limit_val_batches=0.1 if config.debug else 1.0,
        callbacks=[checkpoint],
        logger=loggers,
    )

    # training
    lightning_module = MyLightningModule(config)
    data_module = MyDataModule(config)
    trainer.fit(lightning_module, datamodule=data_module)

    torch.cuda.empty_cache()
    gc.collect()


def main() -> None:
    config = Config()

    # TODO DEBUG
    config.debug = False

    secret_cfg = utils.get_config()
    runfilepath = Path(__file__)
    expname = runfilepath.name.split(".")[0]
    base_dir = Path(".")
    output_dir = base_dir / "output" / f"{expname}"
    output_dir.mkdir(exist_ok=True)
    config.output_dir = output_dir
    logger.info(f"project: {secret_cfg.prj_name}, name: {secret_cfg.entity}")
    if config.is_logging:
        wandb.login(key=secret_cfg.wandb_token)

    logger.info(f"Settingns:\n\t{asdict(config)}")

    n_fold = config.n_fold if not config.debug else 2
    test_df = pd.read_csv(base_dir / "input" / "atmacup-11" / "test.csv")
    for fold in range(n_fold):
        logger.info("\n\t\t" + "=" * 10 + f"  Fold {fold} start  " + "=" * 10)
        train(
            expname=expname,
            secret_cfg=secret_cfg,
            output_dir=output_dir,
            config=config,
            fold=fold,
        )
        ckpt = list(Path(".").glob(f"{output_dir}/{expname}_fold{fold}*.ckpt"))[0]
        result = make_predict(path=str(ckpt), config=config)
        result_df = pd.DataFrame(result)
        result_df = result_df.rename(columns={"preds": f"fold{fold}_preds"})
        test_df = pd.merge(left=test_df, right=result_df, on="object_id", how="left")
        logger.info("\n\t\t" + "=" * 10 + f"  Fold {fold} end  " + "=" * 10)
        torch.cuda.empty_cache()
        gc.collect()

    # make submission.csv
    make_sub(preds_df=test_df, n_fold=n_fold, output_dir=str(output_dir))


if __name__ == "__main__":
    main()
