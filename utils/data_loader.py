import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=32, data_dir='./data'):
    """
    Loads Oxford Flowers 102 dataset.
    Train: 10/class, Val: 10/class, Test: rest
    """
    # ViT base patch 16 expects 224x224 and standard ImageNet normalization
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Note: download=True will fetch the dataset if it doesn't exist
    train_dataset = datasets.Flowers102(root=data_dir, split='train', download=True, transform=transform_train)
    val_dataset = datasets.Flowers102(root=data_dir, split='val', download=True, transform=transform_test)
    test_dataset = datasets.Flowers102(root=data_dir, split='test', download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader


def get_fewshot_dataloaders(k_shot, batch_size=32, data_dir='./data'):
    """
    Load Flowers 102 with only k_shot training images per class.
    Validation and test sets remain unchanged.
    """
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader
