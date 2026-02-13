import cv2
import numpy as np

class PreprocessingPipeline:
    def __init__(
        self,
        clip_limit=2.0,
        denoise_strength=10,
        block_size=11,
        c=2,
    ):
        self.clip_limit = clip_limit
        self.denoise_strength = denoise_strength
        self.block_size = block_size | 1
        self.c = c

    def run(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Input image is None")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(self.clip_limit, (8, 8))
        enhanced = clahe.apply(gray)

        denoised = cv2.fastNlMeansDenoising(
            enhanced, None, self.denoise_strength, 7, 21
        )

        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.block_size,
            self.c,
        )

        return binary.astype(np.float32) / 255.0
