"""Versioned training and deployment artifacts."""

from mindface.artifacts.model_bundle import ModelBundle, load_model_bundle, save_model_bundle

__all__ = ["ModelBundle", "load_model_bundle", "save_model_bundle"]
