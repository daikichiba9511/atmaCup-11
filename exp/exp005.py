""" exp005

* Self-Supervised Learning by lightning bolts
* Simsiam
* resnet18 finetuning
* epoch 60

Reference

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
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
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
from pl_bolts.models.self_supervised.simclr import (
    SimCLREvalDataTransform,
    SimCLRTrainDataTransform,
)
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
    expname: str = ""
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
    epochs: int = 60
    batch_size: int = 8 * 8 * 2
    fp16: bool = False
    img_size: Tuple[int, int] = (224, 224)
    train_batch_size: int = 512
    test_batch_size: int = 2048
    ckpt_path: Optional[Path] = None
    is_test: bool = False

    optim = OptimConfig(
        optimizer=OptimizerCfg(
            # name="AdamW",
            # params={"lr": 1e-3},
            name="MADGRAD",
            params={"lr": 1e-2, "eps": 1e-6, "weight_decay": 5e-4},
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
        df: pd.core.frame.DataFrame,
        img_size: Tuple[int, int],
        is_test: bool = False,
        is_valid: bool = False,
        fp16: bool = False,
    ) -> None:
        self.data_dir = Path(".") / "input" / "atmacup-11"
        self.size = img_size
        self.df = df
        self.dtype = torch.float32 if not fp16 else torch.float16
        self.is_test = is_test
        self.is_valid = is_valid

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df["object_id"][idx]
        image = self.get_image(img_id + ".jpg")

        if self.is_test:
            image = image.to(self.dtype)
            return {"object_id": img_id, "image": image}
        else:
            # image = [image[i].to(dtype=self.dtype) for i in range(len(image))] # for SimCLR
            image = image.to(self.dtype)
            labels = torch.tensor(self.df["target"][idx], dtype=self.dtype)
            # return image, labels # for SimCLR
            return {"object_id": img_id, "image": image, "target": labels}

    def get_image(self, img_path: str):
        # transform = self.get_transform_simclr()
        transform = self.get_preprocess()
        img = Image.open(str(self.data_dir / "image" / img_path))
        img = img.convert("RGB")
        img = np.array(img)
        img = transform(image=img)["image"]
        # img = transform(img) # for SimCLR
        return img

    def get_transform_simclr(self):
        if self.is_valid:
            return SimCLREvalDataTransform(input_height=self.size[0])
        elif self.is_test:
            return A.Compose(
                [
                    A.Resize(width=self.size[0], height=self.size[1], p=1.0),
                    ToTensorV2(),
                ]
            )

        else:
            return SimCLRTrainDataTransform(input_height=self.size[0])

    def get_preprocess(self):
        if self.is_test:
            return A.Compose(
                [
                    A.Resize(width=self.size[0], height=self.size[1], p=1.0),
                    ToTensorV2(),
                ]
            )
        return A.Compose(
            [
                A.Resize(width=self.size[0], height=self.size[1], p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                # A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, p=0.5),
                ToTensorV2(),
            ]
        )


class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.test_path = cfg.test_csv_path
        self.train_path = cfg.train_csv_path
        self.target_col = cfg.target_col
        self.fold = cfg.fold
        self.n_fold = cfg.n_fold
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
        kf = model_selection.StratifiedKFold(
            n_splits=self.n_fold, shuffle=True, random_state=self.seed
        )
        for fold_i, (train_idx, valid_idx) in enumerate(
            kf.split(train, train[self.target_col])
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
        train_df, valid_df = self.split_train_valid_df(train, self.fold)
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
            img_size=self.img_size,
            is_valid=(phase == "valid"),
        )
        return dataset

    def get_loader(self, phase: str):
        dataset = self.get_dataset(phase=phase)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if phase == "train" else False,
            num_workers=self.num_workers,
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
    def __init__(self, config: Config, model_state=None) -> None:
        super(Atma11Model, self).__init__()
        logger.info(f"Loading ... {config.model_name}")
        # make model by using Pretrainsed ssl
        self.model = resnet.resnet18(pretrained=False)
        if model_state is not None:
            self.model.load_state_dict(model_state)
        self.model.fc = nn.Sequential(
            # nn.Linear(in_features=512, out_features=32, bias=True),
            # nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=config.num_labels, bias=True),
        )
        if config.debug:
            logger.debug(f"\n{self.model}")

        # for param in self.model.parameters():
            # param.require_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape of x: (batch_size, channel, w, h)
        output = self.model(x).view(-1)
        # logger.debug(f"{x.detach().cpu().shape}, {output.shape}")
        return output


class MyLightningModule(pl.LightningModule):
    def __init__(self, cfg: Config, model_state=None) -> None:
        super(MyLightningModule, self).__init__()
        self.net = Atma11Model(config=cfg, model_state=model_state)
        self.criterion = torch.nn.MSELoss()
        self.save_hyperparameters()
        self.optim = cfg.optim
        self.fold = cfg.fold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # logger.debug(f" shape of x {x.shape}")
        logits = self.net(x)
        return logits

    def training_step(self, batch: Dict[str, Any], batch_nb: int):
        # logger.debug(f"batch_nb is {batch_nb}")
        data = batch["image"]
        targets = batch["target"]
        # logger.debug(f"shape of data {data.shape}, shape of target {targets.shape}")
        logits = self(data)
        loss = torch.sqrt(self.criterion(logits, targets))
        self.log(f"fold{self.fold}_train_loss", loss)
        wandb.log({f"fold{self.fold}_train_loss": loss.detach().cpu()})
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_nb: int
    ) -> OrderedDict:
        # data = images
        data, targets = batch["image"], batch["target"]
        logits = self.forward(data)
        loss = torch.sqrt(self.criterion(logits, targets))
        self.log(f"fold{self.fold}_val_loss", loss)
        wandb.log({f"fold{self.fold}_valid_loss": loss.detach().cpu()})
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
        output = self.forward(data)
        return OrderedDict({"preds": output, "object_id": object_id})

    def validation_epoch_end(self, outputs):
        result_dict = dict()
        result_dict["epoch"] = int(self.current_epoch)
        result_dict[f"fold{self.fold}_val_loss"] = (
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
        result_dict[f"fold{self.fold}_v_score"] = score_func(
            preds=preds, targets=targets
        )
        self.log_dict(result_dict, prog_bar=True)

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        cfg_optim = self.optim
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


def make_predict(config: Config) -> Dict[str, np.ndarray]:
    """ """
    debug = config.debug
    fp16 = config.fp16
    epochs = config.epochs
    batch_size = 2048
    logger.debug(f"fold{config.fold} test batch_size is loaded {batch_size}")

    ckpt_path = list(
        config.output_dir.glob(f"{config.expname}_fold{config.fold}epoch*.ckpt")
    )[0]
    state = torch.load(ckpt_path)["state_dict"]

    logger.info("predict start")
    lightning_module = MyLightningModule(config)
    lightning_module.load_state_dict(state)
    data_module = MyDataModule(config)
    data_module.setup(stage="predict")
    # predict for test dataset
    trainer = Trainer(
        max_epochs=epochs if not debug else 1,
        precision=16 if fp16 else 32,
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
    config.is_test = False

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
) -> Dict[str, Any]:
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
        save_weights_only=False,
        save_top_k=None,
        monitor=f"fold{fold}_val_loss",
        mode="max",
        verbose=False,
        every_n_train_steps=None,
        every_n_val_epochs=None,
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
        limit_val_batches=0.15 if config.debug else 1.0,
        callbacks=[checkpoint],
        logger=loggers,
    )

    # training
    # If you see more detail, please see this source below
    # https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simsiam/simsiam_module.py#L19-L268
    batch_size = 512
    logger.debug(f"fold{config.fold} train batch size is loaded {batch_size}")
    ckpt_path = list(
        Path("output/exp002").glob(f"exp002_fold{config.fold}epoch*.ckpt")
    )[0]
    ssl_model = SimSiam.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        gpus=1,
        dataset="stl10",
        num_samples=128,
        batch_size=batch_size,
        arch="resnet18",
        num_workers=4,
        max_epochs=200 if not config.debug else 1,
        first_conv=True,
        hidden_mlp=512,
    )
    state = ssl_model.online_network.encoder.state_dict()
    data_module = MyDataModule(config)
    lightning_module = MyLightningModule(config, model_state=state)

    trainer.fit(
        lightning_module,
        data_module,
        # train_dataloader=data_module.get_loader(phase="train"),
        # val_dataloaders=[data_module.get_loader(phase="valid")],
    )

    torch.cuda.empty_cache()
    gc.collect()


def main() -> None:
    config = Config()

    # TODO DEBUG
    # config.debug = True

    secret_cfg = utils.get_config()
    runfilepath = Path(__file__)
    expname = runfilepath.name.split(".")[0]
    config.expname = expname
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
        result = make_predict(config)
        result_df = pd.DataFrame(result)
        result_df = result_df.rename(columns={"preds": f"fold{fold}_preds"})
        test_df = pd.merge(left=test_df, right=result_df, on="object_id", how="left")
        logger.info("\n\t\t" + "=" * 10 + f"  Fold {fold} end  " + "=" * 10)

        torch.cuda.empty_cache()
        gc.collect()

    # make submission.csv
    make_sub(preds_df=test_df, n_fold=n_fold, output_dir=str(output_dir))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
