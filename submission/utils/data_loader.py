import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_transforms(strong_aug=False):
    """Return train and test transforms."""
    if strong_aug:
        # Stronger augmentation: RandomResizedCrop + ColorJitter
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Standard augmentation
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return transform_train, transform_test


def get_dataloaders(batch_size=32, data_dir='./data', mixup_alpha=0.0, strong_aug=False):
    """
    Loads Oxford Flowers 102 dataset.
    Train: 10/class, Val: 10/class, Test: rest
    If mixup_alpha > 0, applies official torchvision MixUp via collate_fn.
    If strong_aug, uses RandomResizedCrop + ColorJitter for training.
    """
    transform_train, transform_test = _get_transforms(strong_aug)

    train_dataset = datasets.Flowers102(root=data_dir, split='train', download=True, transform=transform_train)
    val_dataset = datasets.Flowers102(root=data_dir, split='val', download=True, transform=transform_test)
    test_dataset = datasets.Flowers102(root=data_dir, split='test', download=True, transform=transform_test)

    # MixUp collate function for training loader only
    train_collate = None
    if mixup_alpha > 0:
        from utils.augmentations import MixUpCollate
        train_collate = MixUpCollate(num_classes=102, alpha=mixup_alpha)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True,
                              collate_fn=train_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader


def get_fewshot_dataloaders(k_shot, batch_size=32, data_dir='./data', mixup_alpha=0.0, strong_aug=False):
    """
    Load Flowers 102 with only k_shot training images per class.
    Validation and test sets remain unchanged.
    """
    transform_train, transform_test = _get_transforms(strong_aug)

    train_dataset = datasets.Flowers102(root=data_dir, split='train', download=True, transform=transform_train)
    val_dataset = datasets.Flowers102(root=data_dir, split='val', download=True, transform=transform_test)
    test_dataset = datasets.Flowers102(root=data_dir, split='test', download=True, transform=transform_test)

    # Subsample k images per class
    targets = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    classes = targets.unique()
    indices = []
    for c in classes:
        class_indices = (targets == c).nonzero(as_tuple=True)[0].tolist()
        indices.extend(class_indices[:k_shot])

    train_subset = torch.utils.data.Subset(train_dataset, indices)

    # MixUp collate function for training loader only
    train_collate = None
    if mixup_alpha > 0:
        from utils.augmentations import MixUpCollate
        train_collate = MixUpCollate(num_classes=102, alpha=mixup_alpha)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True,
                              collate_fn=train_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader
