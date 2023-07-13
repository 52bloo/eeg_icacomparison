import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import NLLLoss
from torch.utils.data import DataLoader, TensorDataset, Dataset


torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")