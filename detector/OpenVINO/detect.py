import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
from openvino.inference_engine import IECore


class TextDetector:

    def __init__(self):
        ie = IECore()
        self.network = ie.read_network(
            model="/home/sweekar/ocr_pipeline/detector/OpenVINO/models/horizontal-text-detection-0001.xml",
            weights="/home/sweekar/ocr_pipeline/detector/OpenVINO/models/horizontal-text-detection-0001.bin", )
        self.execution_net = ie.load_network(self.network, "CPU")
        self.input_layer = next(iter(self.execution_net.input_info))
        self.output_layer = next(iter(self.execution_net.outputs))

        self.colors = {"red": (0, 0, 255), "green": (0, 255, 0)}

    def draw_overlay(self, im_name, original_image, resized_image, predictions, res_folder='results/'):
        # Fetch image shapes to calculate ratio
        (real_y, real_x), (resized_y, resized_x) = original_image.shape[:2], resized_image.shape[:2]
        ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
        bboxes = []
        for box in predictions:
            # conf = box[-1]
            (x_min, y_min, x_max, y_max) = [int(max(corner_position * ratio_y, 10)) if idx % 2
                                            else int(corner_position * ratio_x) for idx, corner_position in
                                            enumerate(box[:-1])]
            bboxes.append((x_min, y_min, x_max, y_max))
        return bboxes

    def detect_text(self, image_path) -> List[Tuple[int, int, int, int]]:
        """
        :param image_path: path of image containing text to be recognized
        :return: list of bboxes, bbox format: [t,l,b,r]
        """
        im_name = image_path.split('/')[-1]
        # reading image
        if not os.path.isfile(image_path):
            print(f"No Image found in {image_path}: check file path/integrity")
            sys.exit(1)
        img = cv2.imread(image_path)
        # extracting model inputs: batch_size = 1, num-channels = 3 (RGB), height = 704, width = 704
        batch_size, num_channels, height, width = self.network.input_info[self.input_layer].tensor_desc.dims

        # resizing the input image to desired size
        resized_image = cv2.resize(img, (width, height))
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

        # inference on the input image
        result = self.execution_net.infer(inputs={self.input_layer: input_image})

        # output predictions
        predictions = result["boxes"]
        # removing no predictions
        predictions_req = predictions[~np.all(predictions == 0, axis=1)]

        bboxes = self.draw_overlay(im_name, img, resized_image, predictions_req)
        return bboxes
