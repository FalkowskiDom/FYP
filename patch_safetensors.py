"""
Patch to inject safetensors.torch mock for Python 3.14 compatibility
"""
import sys
import os

# Add the project directory to sys.path to import our mock
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Import and inject the mock
import safetensors_torch_mock

# Create the torch module in safetensors package
import safetensors
safetensors.torch = safetensors_torch_mock

print("Applied safetensors.torch patch for Python 3.14 compatibility")
