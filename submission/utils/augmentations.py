from torch.utils.data.dataloader import default_collate
from torchvision.transforms import v2


class MixUpCollate:
    """
    Picklable collate function that applies official torchvision v2.MixUp.
    Works with num_workers > 0 on Windows.
    Usage: DataLoader(dataset, collate_fn=MixUpCollate(num_classes=102, alpha=0.2))
    """
    def __init__(self, num_classes=102, alpha=0.2):
        self.mixup = v2.MixUp(num_classes=num_classes, alpha=alpha)

    def __call__(self, batch):
        return self.mixup(*default_collate(batch))
