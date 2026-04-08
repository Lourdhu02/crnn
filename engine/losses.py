import torch.nn as nn

def get_ctc_loss(blank=0):
    return nn.CTCLoss(blank=blank, zero_infinity=True)
