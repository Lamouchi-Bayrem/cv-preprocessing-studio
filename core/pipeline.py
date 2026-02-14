import cv2
import numpy as np
from .config import PipelineConfig

class PreprocessingPipeline:
    def __init__(self):
        self.config = PipelineConfig()

    def update_config(self, cfg: dict):
        for key, value in cfg.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def preprocess(self, img: np.ndarray):
        if img is None:
            raise ValueError("Image is None")

        processed = img.copy()
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed.copy()

        # === CLAHE ===
        if self.config.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=self.config.clip_limit, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # === Denoise ===
        if self.config.apply_denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, h=self.config.denoise_strength)

        # === Gaussian ===
        if self.config.apply_gaussian:
            k = max(3, self.config.gaussian_ksize // 2 * 2 + 1)
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        # === Sharpen ===
        if self.config.apply_sharpen:
            kernel = np.array([[-1, -1, -1], [-1, 9 * self.config.sharpen_strength, -1], [-1, -1, -1]])
            gray = cv2.filter2D(gray, -1, kernel)

        # === Bilateral ===
        if self.config.apply_bilateral:
            gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # === Brightness / Contrast ===
        gray = cv2.convertScaleAbs(gray, alpha=self.config.contrast, beta=self.config.brightness)

        # === Gamma ===
        if abs(self.config.gamma - 1.0) > 0.01:
            inv_gamma = 1.0 / self.config.gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            gray = cv2.LUT(gray, table)

        # === Threshold ===
        if self.config.apply_threshold:
            block = max(3, self.config.block_size // 2 * 2 + 1)
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, block, self.config.c)

        # === Canny ===
        if self.config.apply_canny:
            gray = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)

        # === Morphology ===
        if self.config.apply_morph:
            k = max(3, self.config.morph_kernel_size // 2 * 2 + 1)
            kernel = np.ones((k, k), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=self.config.morph_iterations)

        # === ROI ===
        if self.config.roi_coords:
            x, y, w, h = self.config.roi_coords
            gray = gray[y:y+h, x:x+w]

        # === Resize ===
        if self.config.target_size:
            gray = cv2.resize(gray, self.config.target_size)

        normalized = gray.astype(np.float32) / 255.0
        return normalized, gray
