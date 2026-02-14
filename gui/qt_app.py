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
