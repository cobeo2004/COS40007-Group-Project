from typing import Tuple
from pathlib import Path
from PIL.Image import Image
import numpy as np
import torch
from ultralytics import YOLO

type ModelLoaderImageType = Tuple[str, Path, int, Image, list, tuple, np.ndarray, torch.Tensor]
class IModelLoader[T: ModelLoaderImageType]:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(model_path)

    def predict(self, image: T):
        raise NotImplementedError("Subclasses must implement this method")

