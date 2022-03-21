import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import segmentation_models_pytorch as smp
import torchmetrics
from pytorch_lightning.utilities.cli import MODEL_REGISTRY


@MODEL_REGISTRY
class Segmenter(LightningModule):

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        patience: int = 3,
        learning_rate: float = 5e-3,
        min_learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(self.hparams.in_channels,
                                    track_running_stats=True),
            torch.nn.Conv2d(self.hparams.in_channels, 3, (1, 1)),
            torch.nn.InstanceNorm2d(3), self.get_model(), torch.nn.Sigmoid())

        self.loss = lambda y_hat, y: torch.nn.BCELoss()(
            y_hat, y) + smp.losses.TverskyLoss(mode=smp.losses.constants.
                                               MULTILABEL_MODE)(y_hat, y)

        self.train_iou = torchmetrics.JaccardIndex(self.hparams.num_classes)
        self.valid_iou = torchmetrics.JaccardIndex(self.hparams.num_classes)
        self.test_iou = torchmetrics.JaccardIndex(self.hparams.num_classes)

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            EarlyStopping(patience=2 * self.hparams.patience,
                          monitor='val_loss',
                          verbose=True,
                          mode='min'),
            ModelCheckpoint(monitor='val_loss',
                            save_top_k=1,
                            mode="min",
                            filename='{epoch}-{val_loss:.6f}'),
        ]

        try:
            callbacks.append(DeviceStatsMonitor())
        except MisconfigurationException:
            pass
        return callbacks

    def get_model(self):
        return smp.Unet(encoder_name='efficientnet-b0',
                        encoder_weights="imagenet",
                        classes=self.hparams.num_classes)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)

        torch.use_deterministic_algorithms(False)
        self.train_iou(torch.round(y_hat).int(), torch.round(y).int())
        torch.use_deterministic_algorithms(True)
        self.log('train_iou',
                 self.train_iou,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True)

        return self.loss(y_hat, y)

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, _batch_idx):
        x, y = batch

        y_hat = self(x)

        torch.use_deterministic_algorithms(False)
        self.valid_iou(torch.round(y_hat).int(), torch.round(y).int())
        torch.use_deterministic_algorithms(True)
        self.log('valid_iou',
                 self.valid_iou,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)

        torch.use_deterministic_algorithms(False)
        self.test_iou(torch.round(y_hat).int(), torch.round(y).int())
        torch.use_deterministic_algorithms(True)
        self.log('test_iou',
                 self.test_iou,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.hparams.patience,
            min_lr=self.hparams.min_learning_rate,
            verbose=True,
            mode="min")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
