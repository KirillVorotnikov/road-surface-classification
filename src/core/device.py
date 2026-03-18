import torch

def get_device(preferred_device: str = 'auto') -> torch.device:
    """Determine the device to run the model on."""
    if preferred_device == 'auto':
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preferred_device)