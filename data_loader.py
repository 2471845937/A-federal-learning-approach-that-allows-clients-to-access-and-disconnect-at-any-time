import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_datasets(client_id):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split train set into two IID parts
    total_size = len(train_set)
    split = total_size // 2
    generator = torch.Generator().manual_seed(42)
    splits = [split, total_size - split]
    train_subsets = random_split(train_set, splits, generator=generator)

    client_train = train_subsets[client_id]
    client_test = test_set  # Use full test set for evaluation

    return client_train, client_test


def get_dataloaders(client_id, batch_size=32):
    train_data, test_data = get_datasets(client_id)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader