import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint, GPUStatsMonitor
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import monai
import torchmetrics


class Classifier(LightningModule):

    def __init__(
        self,
        patience: int = 3,
        learning_rate: float = 5e-3,
        min_learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()

        self.patience = patience
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.weight_decay = weight_decay

        self.model = torch.nn.Sequential(torch.nn.Conv2d(1, 3, (1, 1)),
                                         torch.nn.InstanceNorm2d(3),
                                         self.get_model(), torch.nn.Sigmoid())

        self.loss = torch.nn.BCELoss()

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            EarlyStopping(patience=2 * self.patience,
                          monitor='val_loss',
                          verbose=True,
                          mode='min'),
            ModelCheckpoint(monitor='val_loss',
                            save_top_k=1,
                            mode="min",
                            filename='{epoch}-{val_loss:.6f}'),
        ]

        try:
            callbacks.append(GPUStatsMonitor())
        except MisconfigurationException:
            pass
        return callbacks

    def get_model(self):
        return monai.networks.nets.EfficientNetBN("efficientnet-b0",
                                                  spatial_dims=2,
                                                  pretrained=True,
                                                  num_classes=10)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)

        self.train_acc(torch.round(y_hat).int(), torch.round(y).int())
        self.log('train_acc_step',
                 self.train_acc,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True)

        return self.loss(y_hat, y)

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)

        self.valid_acc(torch.round(y_hat).int(), torch.round(y).int())
        self.log('valid_acc',
                 self.valid_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self(x)

        self.test_acc(torch.round(y_hat).int(), torch.round(y).int())
        self.log('test_acc',
                 self.test_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.patience,
            min_lr=self.min_learning_rate,
            verbose=True,
            mode="min")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
