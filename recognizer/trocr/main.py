from typing import List, Tuple

import recognizer.trocr.src.util as utility
from recognizer.trocr.src.configs import constants
from recognizer.trocr.src.dataset import MemoryDataset
from torch.utils.data import DataLoader
from recognizer.trocr.src.scripts import predict
from PIL import Image


class TROCRPredictor:
    def __init__(self, use_local_model: bool = True):
        self.processor = utility.load_processor()
        self.model = utility.load_model(use_local_model)

    def predict_images(self, image_paths: List[str]) -> List[Tuple[str, float]]:
        try:
            images = [Image.open(path) for path in image_paths]
            dataset = MemoryDataset(images, self.processor)
            dataloader = DataLoader(dataset, constants.batch_size)
            predictions, confidence_scores = predict(self.processor, self.model, dataloader)
            return predictions, confidence_scores
        except Exception as e:
            print(str(e))

