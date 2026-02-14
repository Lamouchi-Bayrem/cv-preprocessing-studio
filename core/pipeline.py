import cv2
import numpy as np
from core.roi import crop_roi
from core.config import ProcessingConfig

class PreprocessingPipeline:
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def run(self, image, roi_coords=None):
        if image is None:
            raise ValueError("Input image is None")

        image_to_process = crop_roi(image, roi_coords)

        gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=self.config.clip_limit,
            tileGridSize=(8, 8),
        )
        clahe_img = clahe.apply(gray)

        denoised = cv2.fastNlMeansDenoising(
            clahe_img,
            None,
            h=self.config.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.block_size,
            self.config.c,
        )

        normalized = binary.astype(np.float32) / 255.0
        return binary, normalized
