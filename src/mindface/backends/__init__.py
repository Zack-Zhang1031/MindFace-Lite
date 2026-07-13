"""Common inference backend adapters."""

from mindface.backends.base import MouthPredictor
from mindface.backends.onnx_backend import OnnxPredictor
from mindface.backends.pytorch_backend import PyTorchPredictor
from mindface.backends.rknn_backend import RknnPredictor

__all__ = ["MouthPredictor", "OnnxPredictor", "PyTorchPredictor", "RknnPredictor"]
