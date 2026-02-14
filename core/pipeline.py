import cv2
import numpy as np

class AdvancedCVPipeline:
    def __init__(self, config):
        self.config = config

    def preprocess(self, img):
        if img is None:
            raise ValueError("Input image is None")

        # ROI
        if self.config.roi_coords:
            x1, y1, x2, y2 = self.config.roi_coords
            img = img[y1:y2, x1:x2]

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE
        if self.config.apply_clahe:
            clahe = cv2.createCLAHE(self.config.clip_limit, (8, 8))
            gray = clahe.apply(gray)

        # Denoise
        if self.config.apply_denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, self.config.denoise_strength, 7, 21)

        # Gaussian blur
        if self.config.apply_gaussian and self.config.gaussian_sigma > 0:
            k = int(self.config.gaussian_sigma * 3) | 1
            gray = cv2.GaussianBlur(gray, (k, k), self.config.gaussian_sigma)

        # Contrast / Brightness
        gray = cv2.convertScaleAbs(gray, alpha=self.config.contrast, beta=self.config.brightness)

        # Gamma correction
        if abs(self.config.gamma - 1.0) > 0.01:
            inv_gamma = 1.0 / self.config.gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            gray = cv2.LUT(gray, table)

        # Threshold
        if self.config.apply_threshold:
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.config.block_size,
                self.config.c,
            )

        #
