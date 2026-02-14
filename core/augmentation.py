import cv2
import numpy as np
import random

class AugmentationEngine:

    @staticmethod
    def random_flip(img):
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        if random.random() > 0.7:
            img = cv2.flip(img, 0)
        return img

    @staticmethod
    def random_rotation(img):
        angle = random.uniform(-30, 30)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    @staticmethod
    def random_noise(img):
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        return cv2.add(img, noise)

    @staticmethod
    def random_brightness(img):
        alpha = random.uniform(0.7, 1.3)
        beta = random.randint(-20, 20)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    @staticmethod
    def cutout(img):
        h, w = img.shape[:2]
        size = random.randint(20, 80)
        x = random.randint(0, w-size)
        y = random.randint(0, h-size)
        img[y:y+size, x:x+size] = 0
        return img

    def apply_all(self, img):
        img = self.random_flip(img)
        img = self.random_rotation(img)
        img = self.random_noise(img)
        img = self.random_brightness(img)
        img = self.cutout(img)
        return img
