"""
Site customization to patch safetensors.torch for Python 3.14 compatibility
"""
import sys
import os

# Add the project directory to sys.path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Create the mock module
import torch
import tempfile

class SafetensorsTorchMock:
    @staticmethod
    def save_file(tensor_dict, filename, metadata=None):
        """Mock implementation of safetensors.torch.save_file"""
        torch.save(tensor_dict, filename)
    
    @staticmethod
    def load_file(filename, device="cpu"):
        """Mock implementation of safetensors.torch.load_file"""
        return torch.load(filename, map_location=device)

# Inject the mock into safetensors
try:
    import safetensors
    safetensors.torch = SafetensorsTorchMock()
    print("Applied safetensors.torch patch via sitecustomize")
except ImportError:
    pass
