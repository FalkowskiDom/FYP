"""
Mock for safetensors.torch module to work with Python 3.14
"""
import torch
import tempfile
import os

def save_file(tensor_dict, filename, metadata=None):
    """Mock implementation of safetensors.torch.save_file"""
    # Use torch.save as a fallback
    torch.save(tensor_dict, filename)

def load_file(filename, device="cpu"):
    """Mock implementation of safetensors.torch.load_file"""
    # Use torch.load as a fallback
    return torch.load(filename, map_location=device)
