from pathlib import Path
from tkinter import Image
from typing import Generic
import numpy as np
import torch
from ultralytics import YOLO
from interfaces.model_loader_interface import IModelLoader, ModelLoaderType


class BoundingBoxModelLoader[I: ModelLoaderType](IModelLoader[I]):
    def __init__(self, model_path: str):
        super().__init__(model_path)

    def predict(self, image: I):
        return self.model.predict(image)


class SegmentationModelLoader[I: ModelLoaderType](IModelLoader[I]):
    def __init__(self, model_path: str):
        super().__init__(model_path)

    def predict(self, image: I):
        return self.model.predict(image)
