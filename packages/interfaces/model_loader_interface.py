"""Generic interface for YOLO model loaders.

This version avoids the new PEP-695 syntax (``type`` statements and
``class Foo[T]`` parameterisation) so that it runs on Python ≤ 3.11 – the
version used in the Docker image and on AWS App Runner.
"""

from typing import Any, Generic, TypeVar, Union
from pathlib import Path

import numpy as np
import torch
from PIL.Image import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Typing helpers that are compatible with Python 3.9–3.11
# ---------------------------------------------------------------------------

ModelLoaderImageType = Union[
    str,
    Path,
    int,  # webcam index
    Image,
    list,
    tuple,
    np.ndarray,
    torch.Tensor,
]

T_Image = TypeVar("T_Image", bound=ModelLoaderImageType)


class IModelLoader(Generic[T_Image]):
    """Base class that wraps a **single** Ultralytics YOLO model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(model_path)

    # ------------------------------------------------------------------
    # Public API – must be overridden by concrete loaders
    # ------------------------------------------------------------------

    def predict(self, image: T_Image, *args: Any, **kwargs: Any):  # noqa: D401,E501 – simple pass-through signature
        """Run inference on *image*.

        Concrete subclasses should delegate to ``self.model.predict`` and
        return the result.
        """

        raise NotImplementedError("Subclasses must implement predict()")
