import os
import cv2
from core.augmentation import AugmentationEngine

class DatasetGenerator:
    def __init__(self, output_dir, augmentations_per_image=5):
        self.output_dir = output_dir
        self.augmentations_per_image = augmentations_per_image
        self.augmentor = AugmentationEngine()

    def generate(self, input_folder):
        os.makedirs(self.output_dir, exist_ok=True)

        for fname in os.listdir(input_folder):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                img = cv2.imread(os.path.join(input_folder, fname))
                base_name = os.path.splitext(fname)[0]

                cv2.imwrite(os.path.join(self.output_dir, f"{base_name}_orig.png"), img)

                for i in range(self.augmentations_per_image):
                    aug = self.augmentor.apply_all(img.copy())
                    cv2.imwrite(
                        os.path.join(self.output_dir, f"{base_name}_aug_{i}.png"),
                        aug,
                    )
