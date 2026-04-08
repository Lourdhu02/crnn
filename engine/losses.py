import torch.nn as nn


def get_ctc_loss(blank_index=0):
    return nn.CTCLoss(
        blank=blank_index,
        reduction="mean",
        zero_infinity=True
    )