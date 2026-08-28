from .spatial_relator import WrappedSpatialRelator
from .cosi import CoSi
from omegaconf import DictConfig
import torch
from torch import nn
import os
from typing import Union, Optional

VERSION_INDICES = {
    0: {'principal': [0, 1], 'associate': [1, 0]},
    1: {'principal': [2, 3, 4], 'associate': [2, 4, 3]},
    2: {'principal': [0, 1, 2, 3, 4], 'associate': [1, 0, 2, 4, 3]},
}

def build_model(device: torch.device, 
                cfg: DictConfig, 
                verbose: bool = False):
    """
    Generalized function to build a model based on the configuration.

    Args:
        device (torch.device): The device to load the model onto.
        cfg (DictConfig): The configuration dictionary.
        verbose (bool, optional): Whether to print detailed loading info. Defaults to False.

    Returns:
        nn.Module: The constructed model.
    """
    
    model_name = cfg.model.model_name.lower()

    # Model selection dictionary
    model_builders = {
        "cosi": build_cosi_model,
        "spatial_relator": build_spatial_relator,
    }

    if model_name not in model_builders:
        raise ValueError(f"Unsupported model: '{model_name}'. Available models: {list(model_builders.keys())}")

    # Select and build the model
    model = model_builders[model_name](device, cfg, verbose)

    if cfg.pretrained_weights:
        load_pretrained_weights(model, cfg.pretrained_weights, verbose)
    else:
        print("[Warning] No pretrained weights loaded")

    model.to(device)
    return model


def build_cosi_model(device:torch.device, 
                  cfg: DictConfig,
                  verbose:bool=False):
    
    model = CoSi(
        cfg=cfg,
        device=device
    )
    
    return model

def build_spatial_relator(
        device: torch.device, cfg: DictConfig, 
        verbose:bool=False
    ):
    
    if cfg.model.version not in VERSION_INDICES:
        raise ValueError(f"Unsupported version: {cfg.model.version}, Check Configuration Files")

    model = WrappedSpatialRelator(
        indices_dict=VERSION_INDICES[cfg.model.version],
        device=device,
        use_consistency = cfg.model.use_consistency
    )
    return model

def load_pretrained_weights(
        model: nn.Module, pretrained_weights: str, verbose: bool = False
    ):
    if not os.path.exists(pretrained_weights):
        raise ValueError(f"[ERROR] Load weights failed: '{pretrained_weights}' does not exist.")

    pretrained_dict = torch.load(pretrained_weights, map_location='cpu')
    model_dict = model.state_dict()

    updated_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                updated_dict[k] = v
            else:
                print(f"[WARNING] Size mismatch for '{k}': "
                      f"pretrained {tuple(v.shape)} vs model {tuple(model_dict[k].shape)}. Skipped.")

    if verbose:
        for k in pretrained_dict:
            if k in model_dict:
                print(f"Updated: {k}")
            else:
                print(f"Skipped (not found): {k}")

    model_dict.update(updated_dict)
    model.load_state_dict(model_dict)

    return model
