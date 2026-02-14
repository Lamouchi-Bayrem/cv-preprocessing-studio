import cv2
import numpy as np
import random
from typing import List

class AugmentationEngine:
    @staticmethod
    def apply_random(img: np.ndarray, n_variants: int = 8) -> List[np.ndarray]:
        variants = []
        for _ in range(n_variants):
            aug = img.copy()
            aug = AugmentationEngine.random_flip(aug)
            aug = AugmentationEngine.random_rotation(aug)
            aug = AugmentationEngine.random_affine(aug)
            aug = AugmentationEngine.random_noise(aug)
            aug = AugmentationEngine.random_brightness_contrast(aug)
            aug = AugmentationEngine.random_color_jitter(aug)
            aug = AugmentationEngine.elastic_transform(aug)
            aug = AugmentationEngine.cutout(aug)
            variants.append(aug)
        return variants

    @staticmethod
    def random_flip(img):
        if random.random() < 0.5: img = cv2.flip(img, 1)
        if random.random() < 0.3: img = cv2.flip(img, 0)
        return img

    @staticmethod
    def random_rotation(img):
        angle = random.uniform(-45, 45)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def random_affine(img):
        h, w = img.shape[:2]
        pts1 = np.float32([[0,0], [w,0], [0,h]])
        pts2 = pts1 + np.random.uniform(-30, 30, (3,2))
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(img, M, (w, h))

    @staticmethod
    def random_noise(img):
        noise = np.random.normal(0, random.randint(5, 25), img.shape).astype(np.uint8)
        return cv2.add(img, noise)

    @staticmethod
    def random_brightness_contrast(img):
        alpha = random.uniform(0.6, 1.4)
        beta = random.randint(-40, 40)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    @staticmethod
    def random_color_jitter(img):
        if len(img.shape) == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[..., 0] = (hsv[..., 0] + random.randint(-20, 20)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    @staticmethod
    def elastic_transform(img, alpha=1000, sigma=50):
        h, w = img.shape[:2]
        x = np.arange(w)
        y = np.arange(h)
        x, y = np.meshgrid(x, y)
        dx = cv2.GaussianBlur(np.random.rand(h, w) * 2 - 1, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(np.random.rand(h, w) * 2 - 1, (0, 0), sigma) * alpha
        mapx = (x + dx).astype(np.float32)
        mapy = (y + dy).astype(np.float32)
        return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    @staticmethod
    def cutout(img):
        h, w = img.shape[:2]
        size = random.randint(30, min(h, w) // 3)
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        img[y:y+size, x:x+size] = 0
        return img
