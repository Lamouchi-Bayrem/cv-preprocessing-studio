import sys
import os
import cv2
import numpy as np
import json
from pathlib import Path


from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTabWidget, QLabel, QPushButton, QCheckBox, QSlider, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QSplitter, QStatusBar, QGroupBox, QListWidget,
    QGridLayout, QSizePolicy, QScrollArea
)

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QAction, QIcon

from core.pipeline import PreprocessingPipeline
from core.augmentation import AugmentationEngine
from core.config import PipelineConfig


class BatchWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, folder_path: str, pipeline: PreprocessingPipeline, config: dict):
        super().__init__()
        self.folder_path = folder_path
        self.pipeline = pipeline
        self.config = config

    def run(self):
        output_dir = Path(self.folder_path) / "processed"
        output_dir.mkdir(exist_ok=True)

        files = [f for f in os.listdir(self.folder_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        for i, filename in enumerate(files):
            img_path = Path(self.folder_path) / filename
            img = cv2.imread(str(img_path))
            if img is not None:
                self.pipeline.update_config(self.config)
                _, processed = self.pipeline.preprocess(img)
                cv2.imwrite(str(output_dir / filename), processed)
            self.progress.emit(int((i + 1) / len(files) * 100))

        self.finished.emit(str(output_dir))


class CVStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Studio Pro - Computer Vision Preprocessing & Generation")
        self.resize(1920, 1080)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #121212;
                color: #e0e0e0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #333;
                margin-top: 10px;
            }
            QSlider::handle:horizontal {
                background: #00bfff;
                border: 1px solid #00bfff;
                width: 14px;
                margin: -4px 0;
            }
            QLabel { color: #e0e0e0; }
        """)

        self.image = None          # Original BGR
        self.processed = None      # Processed uint8
        self.roi = None
        self.pipeline = PreprocessingPipeline()
        self.augmentor = AugmentationEngine()

        self.init_ui()

    def init_ui(self):
        # ==================== MENU ====================
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_act = QAction("Open Image", self)
        open_act.triggered.connect(self.open_image)
        file_menu.addAction(open_act)

        save_act = QAction("Save Processed Image", self)
        save_act.triggered.connect(self.save_processed)
        file_menu.addAction(save_act)

        batch_act = QAction("Batch Process Folder", self)
        batch_act.triggered.connect(self.process_folder)
        file_menu.addAction(batch_act)

        # ==================== CENTRAL WIDGET ====================
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # ==================== LEFT PANEL (CONTROLS) ====================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        tabs = QTabWidget()
        tabs.setStyleSheet("QTabBar::tab { padding: 10px; }")

        # ── Preprocessing Tab ──
        prep_tab = QWidget()
        prep_form = QFormLayout(prep_tab)
        prep_form.setSpacing(12)

        # CLAHE
        self.clahe_check = QCheckBox("CLAHE")
        self.clahe_check.setChecked(True)
        self.clahe_check.stateChanged.connect(self.on_param_changed)
        self.clahe_slider = QSlider(Qt.Orientation.Horizontal)
        self.clahe_slider.setRange(1, 20)
        self.clahe_slider.setValue(3)
        self.clahe_slider.valueChanged.connect(self.on_param_changed)
        prep_form.addRow(self.clahe_check, self.clahe_slider)

        # Denoise
        self.denoise_check = QCheckBox("Denoise")
        self.denoise_check.setChecked(True)
        self.denoise_check.stateChanged.connect(self.on_param_changed)
        self.denoise_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_slider.setRange(1, 50)
        self.denoise_slider.setValue(15)
        self.denoise_slider.valueChanged.connect(self.on_param_changed)
        prep_form.addRow(self.denoise_check, self.denoise_slider)

        # Gaussian
        self.gaussian_check = QCheckBox("Gaussian Blur")
        self.gaussian_check.stateChanged.connect(self.on_param_changed)
        self.gaussian_slider = QSlider(Qt.Orientation.Horizontal)
        self.gaussian_slider.setRange(3, 21)
        self.gaussian_slider.setValue(5)
        self.gaussian_slider.valueChanged.connect(self.on_param_changed)
        prep_form.addRow(self.gaussian_check, self.gaussian_slider)

        # Sharpen
        self.sharpen_check = QCheckBox("Sharpen")
        self.sharpen_check.stateChanged.connect(self.on_param_changed)
        self.sharpen_slider = QSlider(Qt.Orientation.Horizontal)
        self.sharpen_slider.setRange(5, 50)
        self.sharpen_slider.setValue(15)
        self.sharpen_slider.valueChanged.connect(self.on_param_changed)
        prep_form.addRow(self.sharpen_check, self.sharpen_slider)

        # Bilateral
        self.bilateral_check = QCheckBox("Bilateral Filter")
        self.bilateral_check.stateChanged.connect(self.on_param_changed)
        prep_form.addRow(self.bilateral_check, QLabel(""))

        # Threshold
        self.threshold_check = QCheckBox("Adaptive Threshold")
        self.threshold_check.setChecked(True)
        self.threshold_check.stateChanged.connect(self.on_param_changed)
        self.block_slider = QSlider(Qt.Orientation.Horizontal)
        self.block_slider.setRange(3, 31)
        self.block_slider.setValue(11)
        self.block_slider.valueChanged.connect(self.on_param_changed)
        self.c_slider = QSlider(Qt.Orientation.Horizontal)
        self.c_slider.setRange(0, 20)
        self.c_slider.setValue(2)
        self.c_slider.valueChanged.connect(self.on_param_changed)
        prep_form.addRow(self.threshold_check, QLabel("Block / C"))
        prep_form.addRow("", self.block_slider)
        prep_form.addRow("", self.c_slider)

        # Canny
        self.canny_check = QCheckBox("Canny Edges")
        self.canny_check.stateChanged.connect(self.on_param_changed)
        self.canny_low = QSlider(Qt.Orientation.Horizontal)
        self.canny_low.setRange(0, 255)
        self.canny_low.setValue(50)
        self.canny_high = QSlider(Qt.Orientation.Horizontal)
        self.canny_high.setRange(0, 255)
        self.canny_high.setValue(150)
        self.canny_low.valueChanged.connect(self.on_param_changed)
        self.canny_high.valueChanged.connect(self.on_param_changed)
        prep_form.addRow(self.canny_check, QLabel("Low / High"))
        prep_form.addRow("", self.canny_low)
        prep_form.addRow("", self.canny_high)

        # Morphology
        self.morph_check = QCheckBox("Morphology")
        self.morph_check.stateChanged.connect(self.on_param_changed)
        self.morph_k = QSlider(Qt.Orientation.Horizontal)
        self.morph_k.setRange(3, 21)
        self.morph_k.setValue(5)
        self.morph_iter = QSlider(Qt.Orientation.Horizontal)
        self.morph_iter.setRange(1, 8)
        self.morph_iter.setValue(1)
        self.morph_k.valueChanged.connect(self.on_param_changed)
        self.morph_iter.valueChanged.connect(self.on_param_changed)
        prep_form.addRow(self.morph_check, QLabel("Kernel / Iter"))
        prep_form.addRow("", self.morph_k)
        prep_form.addRow("", self.morph_iter)

        # Brightness / Contrast / Gamma
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.on_param_changed)

        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0.5, 2.0)
        self.contrast_spin.setSingleStep(0.05)
        self.contrast_spin.setValue(1.0)
        self.contrast_spin.valueChanged.connect(self.on_param_changed)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.2, 3.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.valueChanged.connect(self.on_param_changed)

        prep_form.addRow("Brightness", self.brightness_slider)
        prep_form.addRow("Contrast", self.contrast_spin)
        prep_form.addRow("Gamma", self.gamma_spin)

        # Resize
        self.resize_check = QCheckBox("Resize Output")
        self.resize_check.stateChanged.connect(self.on_param_changed)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 4096)
        self.width_spin.setValue(512)
        self.width_spin.valueChanged.connect(self.on_param_changed)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 4096)
        self.height_spin.setValue(512)
        self.height_spin.valueChanged.connect(self.on_param_changed)
        prep_form.addRow(self.resize_check, QLabel("W × H"))
        prep_form.addRow("", self.width_spin)
        prep_form.addRow("", self.height_spin)

        # ROI Button
        roi_btn = QPushButton("🎯 Select ROI (on original)")
        roi_btn.clicked.connect(self.select_roi)
        prep_form.addRow(roi_btn)

        tabs.addTab(prep_tab, "Preprocessing")

        # ── Augmentation Tab ──
        aug_tab = QWidget()
        aug_layout = QVBoxLayout(aug_tab)
        gen_btn = QPushButton("🚀 Generate 8 Random Augmentations")
        gen_btn.clicked.connect(self.generate_augmentations)
        aug_layout.addWidget(gen_btn)

        self.aug_scroll = QScrollArea()
        self.aug_scroll.setWidgetResizable(True)
        self.aug_container = QWidget()
        self.aug_grid = QGridLayout(self.aug_container)
        self.aug_scroll.setWidget(self.aug_container)
        aug_layout.addWidget(self.aug_scroll)

        tabs.addTab(aug_tab, "Augmentations")

        left_layout.addWidget(tabs)

        # Preset buttons
        preset_box = QGroupBox("Presets")
        preset_layout = QHBoxLayout(preset_box)
        save_preset_btn = QPushButton("💾 Save Preset")
        save_preset_btn.clicked.connect(self.save_preset)
        load_preset_btn = QPushButton("📂 Load Preset")
        load_preset_btn.clicked.connect(self.load_preset)
        preset_layout.addWidget(save_preset_btn)
        preset_layout.addWidget(load_preset_btn)
        left_layout.addWidget(preset_box)

        # ==================== CENTER - SPLIT VIEW ====================
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(8)

        self.orig_label = QLabel("Original")
        self.orig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.orig_label.setStyleSheet("border: 2px solid #333; background: #1e1e1e;")
        self.orig_label.setMinimumSize(600, 400)
        self.orig_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.proc_label = QLabel("Processed")
        self.proc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.proc_label.setStyleSheet("border: 2px solid #00bfff; background: #1e1e1e;")
        self.proc_label.setMinimumSize(600, 400)
        self.proc_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.splitter.addWidget(self.orig_label)
        self.splitter.addWidget(self.proc_label)

        # ==================== RIGHT PANEL (INFO) ====================
        right_panel = QGroupBox("Info")
        right_layout = QVBoxLayout(right_panel)
        self.info_label = QLabel("No image loaded")
        self.info_label.setWordWrap(True)
        right_layout.addWidget(self.info_label)
        right_layout.addStretch()

        # ==================== ASSEMBLE LAYOUT ====================
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(self.splitter, 4)
        main_layout.addWidget(right_panel, 1)

        # Status bar
        self.statusBar().showMessage("Ready - Open an image to begin")

    # ====================== HELPER METHODS ======================
    def build_config(self) -> dict:
        return {
            "apply_clahe": self.clahe_check.isChecked(),
            "clip_limit": self.clahe_slider.value(),
            "apply_denoise": self.denoise_check.isChecked(),
            "denoise_strength": self.denoise_slider.value(),
            "apply_gaussian": self.gaussian_check.isChecked(),
            "gaussian_ksize": self.gaussian_slider.value(),
            "apply_sharpen": self.sharpen_check.isChecked(),
            "sharpen_strength": self.sharpen_slider.value() / 10.0,
            "apply_bilateral": self.bilateral_check.isChecked(),
            "apply_threshold": self.threshold_check.isChecked(),
            "block_size": self.block_slider.value(),
            "c": self.c_slider.value(),
            "apply_canny": self.canny_check.isChecked(),
            "canny_low": self.canny_low.value(),
            "canny_high": self.canny_high.value(),
            "apply_morph": self.morph_check.isChecked(),
            "morph_kernel_size": self.morph_k.value(),
            "morph_iterations": self.morph_iter.value(),
            "gamma": self.gamma_spin.value(),
            "brightness": self.brightness_slider.value(),
            "contrast": self.contrast_spin.value(),
            "target_size": (self.width_spin.value(), self.height_spin.value())
            if self.resize_check.isChecked() else None,
            "roi_coords": self.roi,
        }

    def on_param_changed(self):
        if self.image is not None:
            self.process_image()

    def update_display(self):
        if self.image is None:
            return

        # Original
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(900, 700, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.orig_label.setPixmap(pix)

        # Processed
        if self.processed is not None:
            proc = self.processed
            if proc.dtype != np.uint8:
                proc = (proc * 255).astype(np.uint8)

            if len(proc.shape) == 2:
                proc_rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
            else:
                proc_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

            h, w = proc_rgb.shape[:2]
            qimg = QImage(proc_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(900, 700, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.proc_label.setPixmap(pix)

    def process_image(self):
        config = self.build_config()
        self.pipeline.update_config(config)
        _, self.processed = self.pipeline.preprocess(self.image)
        self.update_display()

    # ====================== ACTIONS ======================
    def open_image(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file:
            self.image = cv2.imread(file)
            if self.image is not None:
                self.statusBar().showMessage(f"Loaded: {Path(file).name}")
                self.info_label.setText(f"Image: {self.image.shape[1]}×{self.image.shape[0]}")
                self.process_image()
            else:
                QMessageBox.critical(self, "Error", "Failed to load image")

    def save_processed(self):
        if self.processed is None:
            QMessageBox.warning(self, "No Image", "Process an image first")
            return
        file, _ = QFileDialog.getSaveFileName(self, "Save Processed", "", "PNG (*.png)")
        if file:
            cv2.imwrite(file, self.processed)
            self.statusBar().showMessage(f"Saved to {file}")

    def select_roi(self):
        if self.image is None:
            return
        cv2.namedWindow("Select ROI - Press ENTER when done", cv2.WINDOW_NORMAL)
        roi = cv2.selectROI("Select ROI - Press ENTER when done", self.image)
        cv2.destroyAllWindows()
        if roi[2] > 0 and roi[3] > 0:
            self.roi = roi
            self.statusBar().showMessage(f"ROI set: {roi}")
            self.process_image()

    def process_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Process")
        if not folder:
            return

        config = self.build_config()
        self.worker = BatchWorker(folder, self.pipeline, config)
        self.worker.progress.connect(lambda p: self.statusBar().showMessage(f"Processing... {p}%"))
        self.worker.finished.connect(self.batch_finished)
        self.worker.start()

    def batch_finished(self, output_dir):
        self.statusBar().showMessage(f"Batch complete → {output_dir}")
        QMessageBox.information(self, "Success", f"Processed images saved to:\n{output_dir}")

    def save_preset(self):
        config = self.build_config()
        file, _ = QFileDialog.getSaveFileName(self, "Save Preset", "", "JSON (*.json)")
        if file:
            with open(file, "w") as f:
                json.dump(config, f, indent=4)
            self.statusBar().showMessage("Preset saved")

    def load_preset(self):
        file, _ = QFileDialog.getOpenFileName(self, "Load Preset", "", "JSON (*.json)")
        if file:
            with open(file) as f:
                config = json.load(f)
            self.pipeline.update_config(config)

            # Update UI widgets from config
            self.clahe_check.setChecked(config.get("apply_clahe", True))
            self.clahe_slider.setValue(config.get("clip_limit", 3))
            # ... (you can add more if you want full sync)

            self.process_image()
            self.statusBar().showMessage("Preset loaded")

    def generate_augmentations(self):
        if self.image is None:
            return

        variants = self.augmentor.apply_random(self.image, n_variants=8)

        # Clear old grid
        for i in reversed(range(self.aug_grid.count())):
            widget = self.aug_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        for i, var in enumerate(variants):
            row, col = divmod(i, 4)
            rgb = cv2.cvtColor(var, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(220, 220, Qt.KeepAspectRatio)

            lbl = QLabel()
            lbl.setPixmap(pix)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.aug_grid.addWidget(lbl, row, col)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Nice dark look
    window = CVStudio()
    window.show()
    sys.exit(app.exec())
