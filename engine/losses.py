import torch
import torch.nn as nn


class CTCLossWithSmoothing(nn.Module):
    def __init__(self, blank_index=0, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.ctc = nn.CTCLoss(blank=blank_index, reduction="mean", zero_infinity=True)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        if self.smoothing > 0.0:
            smooth_loss = -log_probs.mean(dim=2).mean()
            return (1.0 - self.smoothing) * ctc_loss + self.smoothing * smooth_loss
        return ctc_loss


def get_ctc_loss(blank_index=0, smoothing=0.0):
    if smoothing > 0.0:
        return CTCLossWithSmoothing(blank_index=blank_index, smoothing=smoothing)
    return nn.CTCLoss(blank=blank_index, reduction="mean", zero_infinity=True)