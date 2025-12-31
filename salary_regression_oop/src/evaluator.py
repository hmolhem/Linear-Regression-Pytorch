class evaluator:
    def __init__(self):
        pass

    def evaluate(self, model, data_loader):
        import torch
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
                loss = ((outputs - targets) ** 2).mean()
                total_loss += loss.item()
        average_loss = total_loss / len(data_loader)
        return average_loss