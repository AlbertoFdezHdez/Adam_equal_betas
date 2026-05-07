import os
import warnings
from pathlib import Path

import torch


AUX_DATA_DIR = "./data"


def _candidate_data_dirs():
    env_dirs = [
        os.environ.get("TRAINING_DATA_DIR"),
        os.environ.get("DATASETS_ROOT"),
        os.environ.get("DATA_DIR"),
    ]
    default_scratch_candidates = [
        "/scratch1/hernanal/datasets",
        f"/scratch1/{os.environ.get('USER', '').strip()}/datasets" if os.environ.get("USER") else None,
    ]
    defaults = ["C:\\Users\\Desktop\\datasets_data", "/datasets_data"]
    return [d for d in [*env_dirs, *default_scratch_candidates, *defaults] if d]


def configure_cache_env(data_root=None):
    base_dir = Path(data_root).expanduser() if data_root else Path(get_data_dir()).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", base_dir / "hf_cache")).expanduser()
    hf_datasets_cache = Path(os.environ.get("HF_DATASETS_CACHE", hf_home / "datasets")).expanduser()
    hf_hub_cache = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub")).expanduser()
    transformers_cache = Path(os.environ.get("TRANSFORMERS_CACHE", hf_home / "transformers")).expanduser()
    torch_home = Path(os.environ.get("TORCH_HOME", base_dir / "torch_cache")).expanduser()

    for path in (base_dir, hf_home, hf_datasets_cache, hf_hub_cache, transformers_cache, torch_home):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["TRAINING_DATA_DIR"] = str(base_dir)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_datasets_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_hub_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))
    os.environ.setdefault("TORCH_HOME", str(torch_home))

    return {
        "data_root": str(base_dir),
        "hf_home": str(hf_home),
        "hf_datasets_cache": str(hf_datasets_cache),
        "hf_hub_cache": str(hf_hub_cache),
        "transformers_cache": str(transformers_cache),
        "torch_home": str(torch_home),
    }

def get_data_dir():
    """
    Return the first existing directory from the dynamic candidate list (Windows/Linux/macOS).
    If none exist, return `AUX_DATA_DIR` and emit a warning.
    """
    data_dirs = _candidate_data_dirs()
    for d in data_dirs:
        p = Path(os.path.expandvars(str(Path(d).expanduser())))
        if p.is_dir():
            return str(p)

    warnings.warn(
        f"None of the DATA_DIRS were found ({data_dirs}). Falling back to AUX_DATA_DIR: {AUX_DATA_DIR}",
        category=UserWarning,
        stacklevel=2,
    )
    fallback = Path(os.environ.get("TRAINING_DATA_DIR", AUX_DATA_DIR)).expanduser()
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback)


def get_n_inputs_from_dataloader(dataloader, n):
    """
    Function to fetch n samples from a DataLoader.

    Args:
        dataloader (DataLoader): PyTorch DataLoader object.
        n (int): Number of samples to fetch.

    Returns:
        Tensor: A batch of `n` samples (images) and their corresponding labels.
    """
    if n == 0:
        return None, None
    all_samples = []
    all_labels = []

    # Accumulate samples until we have `n`
    for inputs, labels in dataloader:
        all_samples.append(inputs)
        all_labels.append(labels)
        
        # Check if we reached `n` samples
        if len(torch.cat(all_samples)) >= n:
            break
    
    # Concatenate all batches collected so far and limit to `n` samples
    samples_batch = torch.cat(all_samples)[:n]
    labels_batch = torch.cat(all_labels)[:n]
    
    return samples_batch, labels_batch

def get_reduced_dataset(dataset, n_samples):
    return torch.utils.data.random_split(dataset, [n_samples, len(dataset)-n_samples])[0]

def get_reduced_dataloader(dataloader, n_samples):
    shuffle = type(dataloader.sampler) == torch.utils.data.sampler.RandomSampler
    dataset = get_reduced_dataset(dataloader.dataset, n_samples=n_samples)
    return torch.utils.data.DataLoader(dataset, batch_size=dataloader.batch_size,
                                        shuffle=shuffle, num_workers=dataloader.num_workers, 
                                        prefetch_factor=dataloader.prefetch_factor, pin_memory=dataloader.pin_memory)

def concatenate_dataloader(dataloaders):
    shuffle = type(dataloaders[0].sampler) == torch.utils.data.sampler.RandomSampler
    datasets = []
    for data in dataloaders:
        datasets.append(data.dataset)
    dataset = torch.utils.data.ConcatDataset(datasets)
    return torch.utils.data.DataLoader(dataset, batch_size=dataloaders[0].batch_size,
                                        shuffle=shuffle, num_workers=dataloaders[0].num_workers, 
                                        prefetch_factor=dataloaders[0].prefetch_factor, pin_memory=dataloaders[0].pin_memory)


def get_balanced_batch_images(dataloader, batch_size, num_classes=10):
    if batch_size == 0:
        return None
    samples_per_class = batch_size / num_classes
    
    batch_samples = []
    batch_labels = [0] * num_classes
    total_samples = 0

    for inputs, labels in dataloader:
        for input, label in zip(inputs, labels):
            if batch_labels[label] < samples_per_class:
                batch_samples.append(input)
                batch_labels[label] += 1
                total_samples += 1
        
        if total_samples >= batch_size:
            break
    
    # Concatenate all batches collected so far and limit to `n` samples
    batch_samples = torch.stack(batch_samples)[:batch_size]
    
    return batch_samples

