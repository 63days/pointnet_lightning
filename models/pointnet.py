from typing import Any, List
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(pl.LightningModule):
    def __init__(self, num_classes, lr, weight_decay):
        super().__init__()
        # this line allows to acess init params with 'self.hparams' attribute
        # e.g., self.hparams.num_classes
        self.save_hyperparameters(logger=False)
        self._build_model()

    def _build_model(self):
        channels = [3, 64, 64, 64, 128, 1024]
        for i in range(1, len(channels)):
            in_ch, out_ch = channels[i - 1], channels[i]
            setattr(
                self,
                "conv" + str(i),
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                ),
            )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.hparams.num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B,C,N]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.max(2)[0]

        return self.fc(x)

    def step(self, batch: Any, reduction: str = "mean"):
        points, labels = batch
        logits = self(points)
        loss = F.cross_entropy(logits, labels, reduction=reduction)
        preds = torch.argmax(logits, dim=1)

        return loss, preds, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = (preds == targets).float().mean() * 100
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = (preds == targets).float().mean() * 100
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, reduction="sum")

        # return results of the batch to aggregate across whole test data.
        correct = (preds == targets).sum()
        total = targets.numel()

        return {"loss": loss, "correct": correct, "total": total}

    def test_epoch_end(self, test_step_outputs: List[Any]):
        loss_sum = 0
        total_correct = 0
        total_seen = 0

        for out in test_step_outputs:
            loss_sum += out["loss"]
            total_correct += out["correct"]
            total_seen += out["total"]

        self.log("test/loss", loss_sum / total_seen, on_epoch=True)
        self.log("test/acc", total_correct / total_seen * 100, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
                params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
                )
