"""TAM: Token Activation Maps for MLLM Interpretability.

A library for visualizing and interpreting the internal activations of 
Multimodal Large Language Models.
"""

from .core import compute_tam

__version__ = "0.1.0"
__all__ = ["compute_tam"]
