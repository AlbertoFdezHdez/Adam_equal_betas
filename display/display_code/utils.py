import torch
import torch.nn as nn
import json
import os

def _ensure_json_path(path: str) -> str:
    return path if path.lower().endswith(".json") else f"{path}.json"

def load_dict(path: str) -> dict:
    json_path = _ensure_json_path(path)
    with open(json_path, "r") as f:
        return json.load(f)

def save_dict(path: str, data: dict) -> None:
    json_path = _ensure_json_path(path)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

def _ensure_pkl_path(path: str) -> str:
    return path if path.lower().endswith(".pkl") else f"{path}.pkl"

def save_dict_pickle(path: str, data: dict) -> None:
    import pickle
    pkl_path = _ensure_pkl_path(path)
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict_pickle(path: str) -> dict:
    import pickle
    pkl_path = _ensure_pkl_path(path)
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def get_params(model) -> torch.Tensor:
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p.detach().reshape(-1))
    
    if not params:
        return torch.empty(0)

    return torch.cat(params).clone()

def get_gradients(model) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is None:
                grads.append(torch.zeros_like(p).reshape(-1))
            else:
                grads.append(p.grad.detach().reshape(-1))

    if not grads:
        return torch.empty(0)

    return torch.cat(grads).clone()

def get_weight_gradients(model) -> torch.Tensor:
    allowed = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)

    grads = []
    for module in model.modules():
        if isinstance(module, allowed):
            for p in module.parameters(recurse=False):
                if p.requires_grad:
                    if p.grad is None:
                        # Keep alignment predictable: insert zeros
                        grads.append(torch.zeros_like(p).reshape(-1))
                    else:
                        grads.append(p.grad.detach().reshape(-1))

    if not grads:
        return torch.empty(0)

    return torch.cat(grads).clone()

def get_weight_params(model) -> torch.Tensor:
    allowed = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)

    params = []
    for module in model.modules():
        if isinstance(module, allowed):
            for p in module.parameters(recurse=False):
                if p.requires_grad:
                    params.append(p.detach().reshape(-1))

    if not params:
        return torch.empty(0)

    return torch.cat(params).clone()