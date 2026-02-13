import cv2
from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QWidget
)
from PyQt6.QtGui import QPixmap, QImage
from core.pipeline import PreprocessingPipeline

class CVStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Preprocessing Studio")
        self.resize(800, 600)

        self.pipeline = PreprocessingPipeline()
        self.image = None

        self.label = QLabel("Load an image")
        self.label.setScaledContents(True)

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)

        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process_image)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.load_btn)
        layout.addWidget(self.process_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self)
        if not path:
            return
        self.image = cv2.imread(path)
        self.show_image(self.image)

    def process_image(self):
        if self.image is None:
            return
        out = self.pipeline.run(self.image)
        out = (out * 255).astype("uint8")
        self.show_image(out, gray=True)

    def show_image(self, img, gray=False):
        if gray:
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))
