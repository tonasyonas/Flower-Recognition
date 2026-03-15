import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32, data_dir='./data'):
    """
    TODO: Implement Oxford Flowers 102 dataset loading.
    - Set up training (10/class), val (10/class), test (rest) splits.
    - Add standard transforms (resize to 224x224 for ViT, normalize).
    """
    pass
