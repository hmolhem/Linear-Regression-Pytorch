class Model:
    def __init__(self, input_size, output_size):
        import torch.nn as nn
        self.model = nn.Linear(input_size, output_size)

    def get_model(self):
        return self.model