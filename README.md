# CV Preprocessing Studio
<img width="1920" height="991" alt="python_HczHHhamXK" src="https://github.com/user-attachments/assets/3fbd2bf2-c761-426d-9c71-85b9148fb411" />

**Interactive computer vision preprocessing & data augmentation toolkit**  
for creating clean, high-quality datasets — especially useful for document processing, OCR, medical imaging, industrial inspection, and general CV model training.

<p align="center">
  
  <br/>
  <em>Modern dark-themed GUI with real-time before/after comparison</em>
</p>

## ✨ Main Features

- **Modern Qt6 GUI** with dark theme and real-time preview
- **Split-view** (original vs processed side-by-side)
- **Interactive ROI selection** (crop region of interest)
- **Rich preprocessing pipeline**:
  - CLAHE contrast enhancement
  - Fast NLMeans + Gaussian + Bilateral denoising
  - Unsharp masking & high-pass sharpening
  - Adaptive thresholding (Gaussian/mean)
  - Canny edge detection
  - Morphological cleaning (open/close)
  - Gamma / brightness / contrast correction
  - Final resize with aspect ratio preservation option
- **Powerful augmentation engine** (8+ realistic transformations shown live)
  - Horizontal/vertical flip
  - Random rotation & affine transforms
  - Gaussian noise
  - Brightness/contrast jitter
  - Color jitter (HSV)
  - Elastic deformation
  - Cutout / random erasing
- **Batch processing** of entire folders (multi-threaded progress)
- **Pipeline presets** — save/load your favorite settings (JSON)
- **Gradio-based web interface** (alternative lightweight frontend)

## Screenshots

*(Add 3–5 real screenshots here later)*

| GUI Main View              | Augmentation Tab           | Batch Processing           |
|----------------------------|----------------------------|----------------------------|
| ![gui](screenshots/gui.png) | ![aug](screenshots/aug.png) | ![batch](screenshots/batch.png) |

## Installation

### Prerequisites

- Python 3.9 – 3.11 recommended
- Qt6 compatible environment (PyQt6)

```bash
# Recommended: create virtual environment first
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows


