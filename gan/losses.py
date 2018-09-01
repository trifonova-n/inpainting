import torch
import torch.nn as nn


class GeneratorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_wl_loss = nn.BCEWithLogitsLoss()

    def forward(self, D_fake_logits):
        target = torch.ones_like(D_fake_logits.data)
        loss = self.bce_wl_loss(D_fake_logits, target)
        return loss


class DiscriminatorLoss(torch.nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.bce_wl_loss = nn.BCEWithLogitsLoss()
        self.label_smoothing = label_smoothing

    def forward(self, D_logit_real, D_logit_fake):
        target = torch.ones_like(D_logit_real.data) - self.label_smoothing / 2.0
        loss_on_real = self.bce_wl_loss(D_logit_real, target)
        target = torch.zeros_like(D_logit_fake.data)
        loss_on_fake = self.bce_wl_loss(D_logit_fake, target)
        loss = 0.5*(loss_on_real + loss_on_fake)
        return loss
