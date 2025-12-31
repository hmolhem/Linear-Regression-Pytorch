class Loss:
    def __init__(self):
        import torch.nn as nn
        self.criterion = nn.MSELoss()

    def compute_loss(self, predictions, targets):
        return self.criterion(predictions, targets)