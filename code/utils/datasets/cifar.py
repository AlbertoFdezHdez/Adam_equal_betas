import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import os
from .utils import get_data_dir


def load_cifar10(batch_size=64, size=32, n_samples=None):
    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomCrop(size, padding=4),  # Data augmentation
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    n_classes = 10

    data_dir = get_data_dir()
    download = not os.path.isdir(data_dir + "/cifar-10-batches-py")
    # Datasets
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=download, transform=test_transform)
    # Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, (3, size, size), n_classes

def load_cifar100(batch_size=64, size=32, n_samples=None):
    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomCrop(size, padding=4),  # Data augmentation
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    n_classes = 100

    data_dir = get_data_dir()
    download = not os.path.isdir(data_dir + "/cifar-100-batches-py")
    # Datasets
    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=download, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=download, transform=test_transform)
    # Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader, (3, size, size), n_classes


def load_cifar100_vit(batch_size=256, size=224, n_samples=None):
    # Transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.0),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])

    n_classes = 100

    data_dir = get_data_dir()
    download = not os.path.isdir(data_dir + "/cifar-100-batches-py")
    # Datasets
    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=download, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=download, transform=test_transform)
    # Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader, (3, size, size), n_classes
