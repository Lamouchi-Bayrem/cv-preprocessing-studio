import cv2
import numpy as np


class AdvancedCVPipeline:
    def __init__(self, config):
        self.config = config

    def preprocess(self, img):
        if img is None:
            raise ValueError("Input image is None")

        # 1️⃣ Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2️⃣ CLAHE
        if self.config.apply_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clip_limit,
                tileGridSize=(8, 8),
            )
            gray = clahe.apply(gray)

        # 3️⃣ Denoising
        if self.config.apply_denoise:
            gray = cv2.fastNlMeansDenoising(
                gray,
                None,
                h=self.config.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21,
            )

        # 4️⃣ Gaussian
        if self.config.apply_gaussian:
            k = max(3, self.config.block_size | 1)
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        # 5️⃣ Brightness / Contrast
        gray = cv2.convertScaleAbs(
            gray,
            alpha=self.config.contrast,
            beta=self.config.brightness,
        )

        # 6️⃣ Gamma
        if abs(self.config.gamma - 1.0) > 0.01:
            inv_gamma = 1.0 / self.config.gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in range(256)
            ]).astype("uint8")
            gray = cv2.LUT(gray, table)

        # 7️⃣ Adaptive Threshold
        if self.config.apply_threshold:
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.config.block_size | 1,
                self.config.c,
            )

        # 8️⃣ Canny
        if self.config.apply_canny:
            gray = cv2.Canny(
                gray,
                self.config.canny_low,
                self.config.canny_high,
            )

        normalized = gray.astype(np.float32) / 255.0
        return normalized, gray
