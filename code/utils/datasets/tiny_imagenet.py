'''
https://www.kaggle.com/datasets/xiataokang/tinyimagenettorch/data
'''
import torch
import torchvision
import torchvision.transforms as transforms
import os
from .utils import get_data_dir
from .utils import get_reduced_dataset

def download_tinyimagenet():
    # Download latest version
    try:
        import kaggle
    except ImportError as exc:
        raise RuntimeError(
            "TinyImageNet download requires the optional 'kaggle' package and a configured kaggle.json."
        ) from exc
    data_dir = get_data_dir()
    dataset = "xiataokang/tinyimagenettorch"
    kaggle.api.dataset_download_files(dataset, path=data_dir, unzip=True)

def load_tinyimagenet(batch_size=64, n_samples=None):
    # Download
    data_dir = get_data_dir()
    if not os.path.isdir(data_dir + "/tiny-imagenet-200"):
        download_tinyimagenet()

    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    n_classes = 200

    # Datasets
    trainset = torchvision.datasets.ImageFolder(root=data_dir + "/tiny-imagenet-200/train", transform=train_transform)
    if n_samples is not None:
        trainset = get_reduced_dataset(trainset, n_samples)
    testset = torchvision.datasets.ImageFolder(root=data_dir + "/tiny-imagenet-200/val", transform=val_transform)

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader, (3, 224, 224), n_classes
