"""
lightly_mlflow_wrapper.py
=========================
MLflow pyfunc wrapper for a lightly_train semantic segmentation model.

predict() contract
------------------
  input : uint8 numpy array, shape (H, W, 3)   — single image, no batch dim
  output: int32 numpy array, shape (H, W)       — class index per pixel
"""

from __future__ import annotations

import logging

import mlflow.pyfunc
import numpy as np

logger = logging.getLogger(__name__)


class LightlySegWrapper(mlflow.pyfunc.PythonModel):
    """
    mlflow.pyfunc.PythonModel wrapper for a lightly_train task model.

    Artifacts expected in the MLflow context:
        "checkpoint"  →  path to the .ckpt file produced by lightly_train
    """

    # Populated by load_context()
    task_model      = None
    device          = None
    image_size      = None
    image_normalize = None

    # ------------------------------------------------------------------
    # mlflow.pyfunc interface
    # ------------------------------------------------------------------

    def load_context(self, context) -> None:
        # Guard: MLflow calls load_context during registration to validate
        # the model. We skip loading if the checkpoint path does not exist
        # yet (i.e. we are in the registration environment, not inference).
        if context is None or not hasattr(context, "artifacts"):
            return

        import os
        ckpt_path = context.artifacts.get("checkpoint")
        if not ckpt_path or not os.path.exists(ckpt_path):
            logger.warning(
                f"Checkpoint not found at '{ckpt_path}' — "
                "skipping load_context (registration environment)"
            )
            return

        import torch
        import lightly_train

        logger.info(f"Loading lightly_train checkpoint: {ckpt_path}")

        # Official public API for loading a trained task model
        self.task_model = lightly_train.load_model(ckpt_path)
        self.task_model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_model = self.task_model.to(self.device)

        self.image_size      = tuple(self.task_model.image_size)   # (H, W)
        self.image_normalize = self.task_model.image_normalize     # {mean, std}

        logger.info(
            f"Model ready — image_size={self.image_size}, device={self.device}"
        )

    def predict(self, context, model_input: np.ndarray) -> np.ndarray:
        """
        Run inference on a single RGB image.

        Args:
            context    : MLflow PythonModelContext (unused at inference time).
            model_input: uint8 numpy array, shape (H, W, 3).

        Returns:
            int32 numpy array, shape (H, W), with predicted class indices.
        """
        if not isinstance(model_input, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(model_input)}")
        if model_input.ndim != 3 or model_input.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) array, got shape {model_input.shape}"
            )

        tensor = self._preprocess(model_input)   # (1, 3, H, W)

        import torch
        with torch.no_grad():
            mask = self.task_model.predict(tensor[0])   # (H, W) tensor

        return mask.cpu().numpy().astype(np.int32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, img: np.ndarray):
        """Convert a uint8 HWC numpy image to a normalised NCHW tensor."""
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.image_normalize["mean"],
                std=self.image_normalize["std"],
            ),
        ])
        return transform(img).unsqueeze(0).to(self.device)   # (1, 3, H, W)