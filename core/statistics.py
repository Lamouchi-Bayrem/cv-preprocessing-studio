import cv2
import os
import numpy as np

class DatasetStats:

    @staticmethod
    def compute_mean_std(folder):
        pixels = []

        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                img = cv2.imread(os.path.join(folder, fname))
                img = img / 255.0
                pixels.append(img.reshape(-1, 3))

        pixels = np.vstack(pixels)
        mean = np.mean(pixels, axis=0)
        std = np.std(pixels, axis=0)
        return mean, std
