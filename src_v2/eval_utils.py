import torch


def eval(model, dataloader):
    """
    Evaluate the model on the given dataloader.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader containing the evaluation data.
    
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        inputs, targets = batch
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return {'avg_loss': avg_loss}