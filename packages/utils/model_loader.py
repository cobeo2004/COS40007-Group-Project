from __future__ import annotations

"""Concrete YOLO model loaders (bbox / segmentation) with Py≤3.11 typing."""

from typing import Generic, TypeVar

from ..interfaces.model_loader_interface import IModelLoader, ModelLoaderImageType

T_Image = TypeVar("T_Image", bound=ModelLoaderImageType)


class BoundingBoxModelLoader(Generic[T_Image], IModelLoader[T_Image]):
    """Runs bbox-detection YOLO models."""

    def __init__(self, model_path: str):
        super().__init__(model_path)

    def predict(
        self,
        image: T_Image,
        save: bool = False,
        verbose: bool = False,
        conf: float = 0.25,
    ):
        return self.model.predict(image, save=save, verbose=verbose, conf=conf)


class SegmentationModelLoader(Generic[T_Image], IModelLoader[T_Image]):
    """Runs segmentation YOLO models."""

    def __init__(self, model_path: str):
        super().__init__(model_path)

    def predict(
        self,
        image: T_Image,
        save: bool = False,
        verbose: bool = False,
        conf: float = 0.25,
    ):
        return self.model.predict(image, save=save, verbose=verbose, conf=conf)
