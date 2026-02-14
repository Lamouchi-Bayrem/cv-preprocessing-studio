import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QCheckBox,
    QSlider, QComboBox, QGroupBox, QSpinBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

from core.pipeline import AdvancedCVPipeline
from core.augmentation import AugmentationEngine


class CVStudio(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Studio - Professional CV Data Engineering Suite")
        self.resize(1400, 900)

        self.image = None
        self.processed_image = None
        self.roi_coords = None

        self.pipeline = AdvancedCVPipeline(self)
        self.augmentor = AugmentationEngine()

        self.init_ui()

    # ================= UI =================

    def init_ui(self):

        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # ===== LEFT PANEL =====
        controls = QVBoxLayout()

        # File Buttons
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)

        save_btn = QPushButton("Save Processed")
        save_btn.clicked.connect(self.save_image)

        augment_btn = QPushButton("Generate Augmentation")
        augment_btn.clicked.connect(self.generate_augmentation)

        controls.addWidget(load_btn)
        controls.addWidget(save_btn)
        controls.addWidget(augment_btn)

        # ===== Pipeline Toggles =====
        toggle_group = QGroupBox("Processing Pipeline")
        toggle_layout = QVBoxLayout()

        self.apply_clahe = QCheckBox("CLAHE")
        self.apply_clahe.setChecked(True)

        self.apply_denoise = QCheckBox("Denoise")
        self.apply_denoise.setChecked(True)

        self.apply_threshold = QCheckBox("Adaptive Threshold")
        self.apply_threshold.setChecked(True)

        self.apply_gaussian = QCheckBox("Gaussian Blur")
        self.apply_canny = QCheckBox("Canny")
        self.apply_morph = QCheckBox("Morphology")

        for cb in [
            self.apply_clahe,
            self.apply_denoise,
            self.apply_threshold,
            self.apply_gaussian,
            self.apply_canny,
            self.apply_morph,
        ]:
            cb.stateChanged.connect(self.process_image)
            toggle_layout.addWidget(cb)

        toggle_group.setLayout(toggle_layout)
        controls.addWidget(toggle_group)

        # ===== Sliders =====
        controls.addWidget(self.create_slider("CLAHE Clip", 1, 10, 2))
        controls.addWidget(self.create_slider("Denoise", 1, 30, 10))
        controls.addWidget(self.create_slider("Block Size", 3, 51, 11))
        controls.addWidget(self.create_slider("Threshold C", 0, 10, 2))
        controls.addWidget(self.create_slider("Gamma", 1, 5, 1))
        controls.addWidget(self.create_slider("Brightness", -100, 100, 0))
        controls.addWidget(self.create_slider("Contrast", 1, 300, 100))
        controls.addWidget(self.create_slider("Canny Low", 0, 200, 50))
        controls.addWidget(self.create_slider("Canny High", 50, 400, 150))
        controls.addWidget(self.create_slider("Morph Kernel", 3, 21, 5))
        controls.addWidget(self.create_slider("Morph Iter", 1, 10, 1))

        controls.addStretch()

        # ===== IMAGE DISPLAY =====
        self.image_label = QLabel("Load an image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Layout combine
        main_layout.addLayout(controls, 1)
        main_layout.addWidget(self.image_label, 3)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    # ================= Slider Factory =================

    def create_slider(self, name, min_val, max_val, default):
        group = QGroupBox(name)
        layout = QHBoxLayout()

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        slider.valueChanged.connect(self.process_image)

        label = QLabel(str(default))

        slider.valueChanged.connect(lambda v: label.setText(str(v)))

        setattr(self, name.replace(" ", "_").lower(), slider)

        layout.addWidget(slider)
        layout.addWidget(label)
        group.setLayout(layout)
        return group

    # ================= Core =================

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self)
        if not path:
            return

        self.image = cv2.imread(path)
        self.display_image(self.image)

    def process_image(self):
        if self.image is None:
            return

        config = self.build_config()
        self.pipeline.config = config

        normalized, output = self.pipeline.preprocess(self.image)
        self.processed_image = normalized

        self.display_image(output)

    def generate_augmentation(self):
        if self.image is None:
            return

        aug = self.augmentor.apply_all(self.image.copy())
        self.display_image(aug)

    def save_image(self):
        if self.processed_image is None:
            return

        path, _ = QFileDialog.getSaveFileName(self)
        if not path:
            return

        cv2.imwrite(path, (self.processed_image * 255).astype(np.uint8))

    def display_image(self, img):
        if len(img.shape) == 2:
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            qimg = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888)

        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    # ================= Config Builder =================

    def build_config(self):
        class Config:
            pass

        cfg = Config()
        cfg.apply_clahe = self.apply_clahe.isChecked()
        cfg.apply_denoise = self.apply_denoise.isChecked()
        cfg.apply_threshold = self.apply_threshold.isChecked()
        cfg.apply_gaussian = self.apply_gaussian.isChecked()
        cfg.apply_canny = self.apply_canny.isChecked()
        cfg.apply_morph = self.apply_morph.isChecked()

        cfg.clip_limit = self.clahe_clip.value()
        cfg.denoise_strength = self.denoise.value()
        cfg.block_size = self.block_size.value() | 1
        cfg.c = self.threshold_c.value()

        cfg.gamma = self.gamma.value()
        cfg.brightness = self.brightness.value()
        cfg.contrast = self.contrast.value() / 100

        cfg.canny_low = self.canny_low.value()
        cfg.canny_high = self.canny_high.value()

        cfg.morph_kernel_size = self.morph_kernel.value() | 1
        cfg.morph_iterations = self.morph_iter.value()
        cfg.morph_type = "Open"

        cfg.roi_coords = None

        return cfg
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QFileDialog, QSlider,
    QLabel
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from core.pipeline import PreprocessingPipeline
from core.config import ProcessingConfig
from gui.image_widget import ImageWidget

class CVStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Image Preprocessing Tool")

        self.config = ProcessingConfig()
        self.pipeline = PreprocessingPipeline(self.config)
        self.image = None

        self.original_widget = ImageWidget()
        self.processed_widget = QLabel()
        self.processed_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)

        save_btn = QPushButton("Save Processed")
        save_btn.clicked.connect(self.save_image)

        process_btn = QPushButton("Process")
        process_btn.clicked.connect(self.process_image)

        layout = QVBoxLayout()
        layout.addWidget(self.original_widget)
        layout.addWidget(self.processed_widget)
        layout.addWidget(load_btn)
        layout.addWidget(process_btn)
        layout.addWidget(save_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self)
        if not path:
            return
        self.image = cv2.imread(path)
        self.display(self.image, self.original_widget)

    def process_image(self):
        if self.image is None:
            return

        roi_coords = self.original_widget.get_roi_coords()
        binary, normalized = self.pipeline.run(self.image, roi_coords)

        self.processed = normalized
        self.display(binary, self.processed_widget)

    def save_image(self):
        if self.processed is None:
            return
        path, _ = QFileDialog.getSaveFileName(self)
        if not path:
            return
        cv2.imwrite(path, (self.processed * 255).astype(np.uint8))

    def display(self, img, widget):
        if len(img.shape) == 2:
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888)

        widget.setPixmap(QPixmap.fromImage(qimg))
