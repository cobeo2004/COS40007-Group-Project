from pathlib import Path
from tkinter import Image
from typing import Generic
import numpy as np
import torch
from ultralytics import YOLO
from ..interfaces.model_loader_interface import IModelLoader, ModelLoaderImageType


class BoundingBoxModelLoader[I: ModelLoaderImageType](IModelLoader[I]):
    def __init__(self, model_path: str):
        super().__init__(model_path)

    def predict(self, image: I, save: bool = False, verbose: bool = False, conf: float = 0.25):
        return self.model.predict(image, save=save, verbose=verbose, conf=conf)


class SegmentationModelLoader[I: ModelLoaderImageType](IModelLoader[I]):
    def __init__(self, model_path: str):
        super().__init__(model_path)

    def predict(self, image: I, save: bool = False, verbose: bool = False, conf: float = 0.25):
        return self.model.predict(image, save=save, verbose=verbose, conf=conf)
