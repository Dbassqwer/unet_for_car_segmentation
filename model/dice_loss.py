import torch
import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        eps = 1e-6
        N = output.shape[0]
        output_flatten = output.view(N, -1)
        output_flatten = torch.sigmoid(output_flatten)
        target_flatten = target.view(N, -1)
        target_flatten = torch.sigmoid(target_flatten)
        intersection = output_flatten * target_flatten

        dice_loss = 2 * (intersection.sum(1) + eps) / (output_flatten.sum(1) + target_flatten.sum(1) + eps)
        dice_loss = 1 - dice_loss.sum() / N

        BCE = nn.BCEWithLogitsLoss()
        BCE_loss = BCE(output, target)
        loss = 0.333 * dice_loss + 0.667 * BCE_loss
        return loss