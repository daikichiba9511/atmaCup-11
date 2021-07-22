"""exp025

* Self-Supervised Learning by lightly
* Simsiam
* epoch 30
* batch size 320
* backbone is vit small patch 16 size 224 pretrained by ssl
* fine-tuning vit

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
    # is_logging: bool = False
    debug: bool = False
    # debug: bool = True
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
    epochs: int = 30
    fold: int = 0
    n_fold: int = 5
    # lr: float = 0.05 * batch_size / 256
    lr: float = 1e-5
    T_max: int = epochs
    freeze_embed: bool = True

    # model
    # ssl by lightly
    # dimension of the embedding
    outdim: int = 512
    backbone_model_name: str = "vit_small_patch16_224"
    num_ftrs = 75264
    num_mlp_layers = 2


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
        if self.is_test or self.is_valid:
            # logger.debug("for inference compose is called")
            return A.Compose(
                [
                    A.Resize(width=self.size[0], height=self.size[1], p=1.0),
                    ToTensorV2(),
                ]
            )
        return A.Compose(
            [
                # A.Resize(width=self.size[0], height=self.size[1], p=1.0),
                A.RandomResizedCrop(width=self.size[0], height=self.size[1], p=1.0),
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


class MyLitModel(pl.LightningModule):
    """for pretrain by lightly + pytorch lightning"""

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


class FineTuningModel(pl.LightningModule):
    def __init__(self, cfg: Config, model: torch.nn.Module) -> None:
        super(FineTuningModel, self).__init__()
        self.model = model
        self.criterion = torch.nn.MSELoss()
        self.save_hyperparameters()

        self.fold = cfg.fold
        self.lr = cfg.lr
        self.T_max = cfg.T_max

        # define only
        self.test_df_lengh = 0
        # logger.debug(f"\n{model}")

    def forward(self, data) -> Any:
        return self.model(data).mean(1)

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        # ids = batch["ids"]
        target = batch["target"]
        preds = self.forward(image)

        loss = torch.sqrt(self.criterion(target, preds))
        self.log(f"fold{self.fold}_train_loss", loss)
        wandb.log({f"fold{self.fold}_train_loss": loss.detach().cpu()})
        return loss

    def validation_step(self, batch, batch_idx):
        # data = images
        data, targets = batch["image"], batch["target"]
        logits = self.forward(data)
        loss = torch.sqrt(self.criterion(logits, targets))
        # self.log(f"fold{self.fold}_val_loss", loss)
        wandb.log({f"fold{self.fold}_valid_loss": loss.detach().cpu()})
        output = OrderedDict(
            {
                "targets": targets.detach(),
                "preds": logits.detach(),
                "loss": loss.detach(),
            }
        )
        return output

    def validation_epoch_end(self, outputs: OrderedDict) -> None:
        result_dict = dict()
        result_dict["epoch"] = int(self.current_epoch)
        result_dict[f"fold{self.fold}_val_loss"] = (
            torch.stack([out["loss"] for out in outputs]).mean().item()
        )
        self.log_dict(result_dict, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        image = batch["image"]
        ids = batch["object_id"]
        preds = self.forward(image)

        self.test_df_lengh += len(image)
        return OrderedDict(
            {
                "preds": preds,
                "object_id": ids,
            }
        )

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


class MLP(nn.Module):
    def __init__(
        self, input_dim: int = 2048, hidden_size: int = 4096, output_dim: int = 256
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


def main():
    config = Config()

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

    test_df = pd.read_csv(base_dir / "input" / "atmacup-11" / "test.csv")
    n_fold = config.n_fold if not config.debug else 1
    for fold in range(n_fold):
        config.fold = fold
        logger.info(f"Settingns:\n\t{asdict(config)}")
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
            mode="min",
            verbose=False,
            every_n_train_steps=None,
            every_n_val_epochs=None,
        )
        trainer = Trainer(
            max_epochs=config.epochs if not config.debug else 5,
            precision=16 if config.fp16 else 32,
            accumulate_grad_batches=8,
            amp_backend="native",
            gpus=1,
            benchmark=False,
            default_root_dir=os.getcwd(),
            deterministic=True,
            limit_train_batches=0.1 if config.debug else 1.0,
            # limit_val_batches=0.0,
            callbacks=[checkpoint],
            logger=loggers,
            # check_val_every_n_epoch=0,
        )

        # prepare model
        def prepare_model(config: Config):
            pretrained_ckpt_path = list(
                (config.base_dir / "output" / "exp023").glob("exp023_epoch=400.ckpt")
            )[0]
            pretrained_lit_module = MyLitModel.load_from_checkpoint(
                checkpoint_path=str(pretrained_ckpt_path)
            )
            # get the model pretrained by SimSiam
            model = pretrained_lit_module.model.backbone
            model.load_state_dict(pretrained_lit_module.model.backbone.state_dict())
            # model = timm.create_model(config.backbone_model_name, pretrained=False)
            if config.freeze_embed:
                for param in model.parameters():
                    param.requires_grad = False
            if config.debug:
                logger.info(f"\n{model}")
            custom_head = torch.nn.Sequential(
                # MLP(input_dim=384, hidden_size=196, output_dim=384 * 2),
                # MLP(input_dim=384 * 2, hidden_size=196, output_dim=384),
                MLP(input_dim=384, hidden_size=196, output_dim=1),
            )
            model.add_module("head", custom_head)
            del pretrained_lit_module
            return model

        # fine turning start
        fine_lit_model = FineTuningModel(cfg=config, model=prepare_model(config))
        data_module = MyDataModule(cfg=Config)
        trainer.fit(model=fine_lit_model, datamodule=data_module)

        # make predict
        ckpt_path = list(
            Path(f"./output/{expname}").glob(f"{expname}_fold{fold}epoch*.ckpt")
        )[0]
        fine_lit_model = FineTuningModel.load_from_checkpoint(ckpt_path)
        test_dataloader = data_module.get_loader(phase="test")
        outputs = trainer.predict(
            fine_lit_model, test_dataloader, return_predictions=True
        )
        preds = np.concatenate(
            [output["preds"].detach().cpu().numpy() for output in outputs]
        ).reshape(-1)
        ids = np.concatenate([output["object_id"] for output in outputs])
        result_df = pd.DataFrame({f"fold{fold}_preds": preds, "object_id": ids})
        test_df = pd.merge(left=test_df, right=result_df, on="object_id", how="left")

    # make submission.csv
    make_sub(preds_df=test_df, n_fold=config.n_fold, output_dir=str(output_dir))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
