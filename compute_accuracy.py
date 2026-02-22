import torch

def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    
    outputs_idx = outputs.max(1)[1].type_as(labels)

    return (outputs_idx == labels).float().mean()