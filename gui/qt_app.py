
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os

class CVStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("CV Studio - Full Computer Vision Preprocessing Studio")
        self.root.geometry("1400x900")
        
        # Core variables
        self.original_image = None
        self.processed_image = None
        self.roi = None
        self.roi_coords = None
        self.display_roi_only = False
        
        # Enhanced parameters
        self.clip_limit = 2.0
        self.denoise_strength = 10
        self.block_size = 11
        self.c = 2
        
        # New advanced parameters
        self.gamma = 1.0
        self.brightness = 0
        self.contrast = 1.0
        self.gaussian_sigma = 0.0
        self.canny_low = 50
        self.canny_high = 150
        self.morph_kernel_size = 5
        self.morph_iterations = 1
        self.morph_type = "None"
        
        # Operation toggles
        self.apply_clahe = tk.BooleanVar(value=True)
        self.apply_denoise = tk.BooleanVar(value=True)
        self.apply_threshold = tk.BooleanVar(value=True)
        self.apply_gaussian = tk.BooleanVar(value=False)
        self.apply_canny = tk.BooleanVar(value=False)
        self.apply_morph = tk.BooleanVar(value=False)
        
        # Create GUI
        self.create_widgets()
        
        # Status bar
        self.status = tk.StringVar(value="Ready - Load an image to begin")
        status_bar = ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_widgets(self):
        # Main container
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        left_frame = ttk.Frame(main_paned, width=420)
        main_paned.add(left_frame, weight=1)
        
        # Right panel - Images
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=4)
        
        # === IMAGE DISPLAYS ===
        self.image_frame = ttk.Frame(right_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Original
        self.original_panel = ttk.LabelFrame(self.image_frame, text="Original Image")
        self.original_panel.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        
        self.original_canvas = tk.Canvas(self.original_panel, width=520, height=520, bg='#f0f0f0')
        self.original_canvas.pack(padx=8, pady=8)
        self.original_canvas.bind("<Button-1>", self.start_roi_selection)
        self.original_canvas.bind("<B1-Motion>", self.update_roi_selection)
        self.original_canvas.bind("<ButtonRelease-1>", self.finalize_roi_selection)
        
        # Processed
        self.processed_panel = ttk.LabelFrame(self.image_frame, text="Processed Image")
        self.processed_panel.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")
        
        self.processed_canvas = tk.Canvas(self.processed_panel, width=520, height=520, bg='#f0f0f0')
        self.processed_canvas.pack(padx=8, pady=8)
        
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.columnconfigure(1, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        
        # === CONTROLS ===
        controls = ttk.Frame(left_frame)
        controls.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # File Operations
        file_frame = ttk.LabelFrame(controls, text="File Operations")
        file_frame.pack(fill=tk.X, pady=6)
        
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)
        
        ttk.Button(btn_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Save Processed", command=self.save_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Batch Process Folder", command=self.batch_process).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Reset All", command=self.reset_all).pack(side=tk.LEFT, padx=4)
        
        # ROI Operations
        roi_frame = ttk.LabelFrame(controls, text="ROI Tools")
        roi_frame.pack(fill=tk.X, pady=6)
        
        roi_btns = ttk.Frame(roi_frame)
        roi_btns.pack(fill=tk.X, padx=8, pady=8)
        
        ttk.Button(roi_btns, text="Toggle ROI Only", command=self.toggle_roi_display).pack(side=tk.LEFT, padx=4)
        ttk.Button(roi_btns, text="Clear ROI", command=self.clear_roi_selection).pack(side=tk.LEFT, padx=4)
        
        # Operation Toggles
        toggle_frame = ttk.LabelFrame(controls, text="Processing Pipeline")
        toggle_frame.pack(fill=tk.X, pady=6)
        
        toggles = ttk.Frame(toggle_frame)
        toggles.pack(fill=tk.X, padx=8, pady=8)
        
        ttk.Checkbutton(toggles, text="CLAHE", variable=self.apply_clahe, command=self.update_params).grid(row=0, column=0, sticky="w", padx=8, pady=2)
        ttk.Checkbutton(toggles, text="Denoise", variable=self.apply_denoise, command=self.update_params).grid(row=0, column=1, sticky="w", padx=8, pady=2)
        ttk.Checkbutton(toggles, text="Adaptive Threshold", variable=self.apply_threshold, command=self.update_params).grid(row=1, column=0, sticky="w", padx=8, pady=2)
        ttk.Checkbutton(toggles, text="Gaussian Blur", variable=self.apply_gaussian, command=self.update_params).grid(row=1, column=1, sticky="w", padx=8, pady=2)
        ttk.Checkbutton(toggles, text="Canny Edges", variable=self.apply_canny, command=self.update_params).grid(row=2, column=0, sticky="w", padx=8, pady=2)
        ttk.Checkbutton(toggles, text="Morphology", variable=self.apply_morph, command=self.update_params).grid(row=2, column=1, sticky="w", padx=8, pady=2)
        
        # Basic Parameters
        basic_frame = ttk.LabelFrame(controls, text="Basic Parameters")
        basic_frame.pack(fill=tk.X, pady=6)
        
        self.create_slider(basic_frame, "CLAHE Clip Limit", 1, 10, 2.0, self.update_clahe, 0)
        self.create_slider(basic_frame, "Denoising Strength", 1, 30, 10, self.update_denoise, 1)
        self.create_slider(basic_frame, "Threshold Block Size", 3, 51, 11, self.update_thresh, 2)
        self.create_slider(basic_frame, "Threshold Constant", 0, 10, 2, self.update_c, 3)
        
        # Advanced Parameters
        adv_frame = ttk.LabelFrame(controls, text="Advanced Filters & Transforms")
        adv_frame.pack(fill=tk.X, pady=6)
        
        self.create_slider(adv_frame, "Gamma Correction", 0.1, 5.0, 1.0, self.update_gamma, 0)
        self.create_slider(adv_frame, "Brightness", -100, 100, 0, self.update_brightness, 1)
        self.create_slider(adv_frame, "Contrast", 0.5, 3.0, 1.0, self.update_contrast, 2)
        self.create_slider(adv_frame, "Gaussian Sigma", 0, 10, 0, self.update_gaussian, 3)
        self.create_slider(adv_frame, "Canny Low", 0, 200, 50, self.update_canny_low, 4)
        self.create_slider(adv_frame, "Canny High", 50, 400, 150, self.update_canny_high, 5)
        self.create_slider(adv_frame, "Morph Kernel", 3, 21, 5, self.update_morph_kernel, 6)
        self.create_slider(adv_frame, "Morph Iterations", 1, 10, 1, self.update_morph_iter, 7)
        
        # Morphology Type (fixed: use grid to avoid mixing with existing grid slaves)
        morph_row = ttk.Frame(adv_frame)
        morph_row.grid(row=8, column=0, columnspan=3, sticky="ew", padx=8, pady=4)
        
        ttk.Label(morph_row, text="Morphology Type:").pack(side=tk.LEFT, padx=5)
        self.morph_combo = ttk.Combobox(morph_row, values=["None", "Erode", "Dilate", "Open", "Close"], state="readonly", width=12)
        self.morph_combo.set("None")
        self.morph_combo.pack(side=tk.LEFT, padx=5)
        self.morph_combo.bind("<<ComboboxSelected>>", self.update_morph_type)
        
        # Configure weights
        basic_frame.columnconfigure(1, weight=1)
        adv_frame.columnconfigure(1, weight=1)
    
    def create_slider(self, parent, label_text, from_val, to_val, default, command, row):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=8, pady=3)
        
        ttk.Label(frame, text=label_text, width=20).pack(side=tk.LEFT)
        
        slider = ttk.Scale(frame, from_=from_val, to=to_val, value=default, command=command)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        value_label = ttk.Label(frame, text=f"{default:.2f}" if isinstance(default, float) else str(default), width=6)
        value_label.pack(side=tk.RIGHT)
        
        # Store references
        if label_text == "CLAHE Clip Limit":
            self.clahe_slider = slider
            self.clahe_value = value_label
        elif label_text == "Denoising Strength":
            self.denoise_slider = slider
            self.denoise_value = value_label
        elif label_text == "Threshold Block Size":
            self.thresh_slider = slider
            self.thresh_value = value_label
        elif label_text == "Threshold Constant":
            self.c_slider = slider
            self.c_value = value_label
        elif label_text == "Gamma Correction":
            self.gamma_slider = slider
            self.gamma_value = value_label
        elif label_text == "Brightness":
            self.brightness_slider = slider
            self.brightness_value = value_label
        elif label_text == "Contrast":
            self.contrast_slider = slider
            self.contrast_value = value_label
        elif label_text == "Gaussian Sigma":
            self.gaussian_slider = slider
            self.gaussian_value = value_label
        elif label_text == "Canny Low":
            self.canny_low_slider = slider
            self.canny_low_value = value_label
        elif label_text == "Canny High":
            self.canny_high_slider = slider
            self.canny_high_value = value_label
        elif label_text == "Morph Kernel":
            self.morph_slider = slider
            self.morph_value = value_label
        elif label_text == "Morph Iterations":
            self.morph_iter_slider = slider
            self.morph_iter_value = value_label
    
    # Update methods for each slider
    def update_clahe(self, val):
        self.clip_limit = float(val)
        self.clahe_value.config(text=f"{self.clip_limit:.1f}")
        if self.original_image is not None:
            self.process_image()
    
    def update_denoise(self, val):
        self.denoise_strength = int(float(val))
        self.denoise_value.config(text=str(self.denoise_strength))
        if self.original_image is not None:
            self.process_image()
    
    def update_thresh(self, val):
        bs = int(float(val))
        self.block_size = bs if bs % 2 == 1 else bs + 1
        self.thresh_value.config(text=str(self.block_size))
        if self.original_image is not None:
            self.process_image()
    
    def update_c(self, val):
        self.c = int(float(val))
        self.c_value.config(text=str(self.c))
        if self.original_image is not None:
            self.process_image()
    
    def update_gamma(self, val):
        self.gamma = float(val)
        self.gamma_value.config(text=f"{self.gamma:.2f}")
        if self.original_image is not None:
            self.process_image()
    
    def update_brightness(self, val):
        self.brightness = int(float(val))
        self.brightness_value.config(text=str(self.brightness))
        if self.original_image is not None:
            self.process_image()
    
    def update_contrast(self, val):
        self.contrast = float(val)
        self.contrast_value.config(text=f"{self.contrast:.2f}")
        if self.original_image is not None:
            self.process_image()
    
    def update_gaussian(self, val):
        self.gaussian_sigma = float(val)
        self.gaussian_value.config(text=f"{self.gaussian_sigma:.1f}")
        if self.original_image is not None:
            self.process_image()
    
    def update_canny_low(self, val):
        self.canny_low = int(float(val))
        self.canny_low_value.config(text=str(self.canny_low))
        if self.original_image is not None:
            self.process_image()
    
    def update_canny_high(self, val):
        self.canny_high = int(float(val))
        self.canny_high_value.config(text=str(self.canny_high))
        if self.original_image is not None:
            self.process_image()
    
    def update_morph_kernel(self, val):
        ks = int(float(val))
        self.morph_kernel_size = ks if ks % 2 == 1 else ks + 1
        self.morph_value.config(text=str(self.morph_kernel_size))
        if self.original_image is not None:
            self.process_image()
    
    def update_morph_iter(self, val):
        self.morph_iterations = int(float(val))
        self.morph_iter_value.config(text=str(self.morph_iterations))
        if self.original_image is not None:
            self.process_image()
    
    def update_morph_type(self, event=None):
        self.morph_type = self.morph_combo.get()
        if self.original_image is not None:
            self.process_image()
    
    def update_params(self, event=None):
        if self.original_image is not None:
            self.process_image()
    
    # === IMAGE LOADING & SAVING ===
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.display_image(self.original_image, self.original_canvas)
                self.roi = None
                self.roi_coords = None
                self.display_roi_only = False
                self.process_image()
                self.status.set(f"Loaded: {os.path.basename(file_path)}")
    
    def save_image(self):
        if self.processed_image is None:
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            save_img = (self.processed_image * 255).astype(np.uint8) if self.processed_image.dtype == np.float32 else self.processed_image
            cv2.imwrite(file_path, save_img)
            self.status.set(f"Saved to: {os.path.basename(file_path)}")
    
    # === ROI HANDLING ===
    def start_roi_selection(self, event):
        self.roi_start = (event.x, event.y)
        if hasattr(self, 'roi_rect'):
            self.original_canvas.delete(self.roi_rect)
        self.roi_rect = self.original_canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='red', width=3)
    
    def update_roi_selection(self, event):
        self.original_canvas.coords(self.roi_rect, self.roi_start[0], self.roi_start[1], event.x, event.y)
    
    def finalize_roi_selection(self, event):
        x1, y1 = self.roi_start
        x2, y2 = event.x, event.y
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            cw = self.original_canvas.winfo_width()
            ch = self.original_canvas.winfo_height()
            
            scale_x = w / cw
            scale_y = h / ch
            
            x1 = max(0, int(x1 * scale_x))
            y1 = max(0, int(y1 * scale_y))
            x2 = min(w, int(x2 * scale_x))
            y2 = min(h, int(y2 * scale_y))
            
            if x2 > x1 and y2 > y1:
                self.roi_coords = (x1, y1, x2, y2)
                self.roi = self.original_image[y1:y2, x1:x2]
                self.process_image()
                self.status.set("ROI selected")
    
    def clear_roi_selection(self):
        self.roi_coords = None
        self.roi = None
        self.display_roi_only = False
        if self.original_image is not None:
            self.display_image(self.original_image, self.original_canvas)
            self.process_image()
    
    def toggle_roi_display(self):
        self.display_roi_only = not self.display_roi_only
        if self.original_image is not None:
            self.process_image()
    
    # === CORE PROCESSING PIPELINE ===
    def apply_preprocessing(self, img):
        if img is None:
            return None, None
        
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. CLAHE (optional)
        if self.apply_clahe.get():
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # 3. Denoising (optional)
        if self.apply_denoise.get():
            denoised = cv2.fastNlMeansDenoising(gray, None, h=self.denoise_strength, templateWindowSize=7, searchWindowSize=21)
        else:
            denoised = gray.copy()
        
        # 4. Gaussian Blur (optional)
        if self.apply_gaussian.get() and self.gaussian_sigma > 0.1:
            ksize = max(3, int(self.gaussian_sigma * 3) | 1)  # ensure odd
            denoised = cv2.GaussianBlur(denoised, (ksize, ksize), self.gaussian_sigma)
        
        # 5. Brightness & Contrast
        adjusted = cv2.convertScaleAbs(denoised, alpha=self.contrast, beta=self.brightness)
        
        # 6. Gamma Correction
        if abs(self.gamma - 1.0) > 0.01:
            inv_gamma = 1.0 / self.gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            adjusted = cv2.LUT(adjusted, table)
        
        # 7. Adaptive Thresholding (optional)
        if self.apply_threshold.get():
            binary = cv2.adaptiveThreshold(
                adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, self.block_size, self.c
            )
        else:
            binary = adjusted
        
        # 8. Morphology (optional)
        if self.apply_morph.get() and self.morph_type != "None":
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            if self.morph_type == "Erode":
                binary = cv2.erode(binary, kernel, iterations=self.morph_iterations)
            elif self.morph_type == "Dilate":
                binary = cv2.dilate(binary, kernel, iterations=self.morph_iterations)
            elif self.morph_type == "Open":
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations)
            elif self.morph_type == "Close":
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)
        
        # 9. Canny Edge Detection (optional)
        if self.apply_canny.get():
            binary = cv2.Canny(binary, self.canny_low, self.canny_high)
        
        # 10. Normalization (for ML-ready output)
        normalized = binary.astype(np.float32) / 255.0
        
        return normalized, binary
    
    def process_image(self):
        if self.original_image is None:
            return
        
        # Determine input
        if self.display_roi_only and self.roi is not None:
            to_process = self.roi.copy()
        else:
            to_process = self.original_image.copy()
        
        # Run the full pipeline
        self.processed_image, display_img = self.apply_preprocessing(to_process)
        
        # Display
        self.display_image(display_img, self.processed_canvas)
        
        # Overlay ROI on original if active
        if self.roi_coords is not None and not self.display_roi_only:
            self.display_image_with_roi(self.original_image, self.original_canvas)
        
        self.status.set("Image processed with current pipeline")
    
    # === DISPLAY HELPERS ===
    def display_image(self, image, canvas):
        if image is None:
            return
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Smart resize
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        h, w = image_rgb.shape[:2]
        
        ratio = min(canvas_w / w, canvas_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        img_pil = Image.fromarray(resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        canvas.delete("all")
        canvas.create_image(canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=img_tk)
        canvas.image = img_tk  # keep reference
    
    def display_image_with_roi(self, image, canvas):
        if image is None:
            return
        img_copy = image.copy()
        if self.roi_coords:
            x1, y1, x2, y2 = self.roi_coords
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 4)
        self.display_image(img_copy, canvas)
    
    # === BATCH PROCESSING ===
    def batch_process(self):
        input_dir = filedialog.askdirectory(title="Select Input Folder")
        if not input_dir:
            return
        
        output_dir = filedialog.askdirectory(title="Select Output Folder")
        if not output_dir:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        count = 0
        
        for fname in os.listdir(input_dir):
            if fname.lower().endswith(extensions):
                path = os.path.join(input_dir, fname)
                img = cv2.imread(path)
                if img is not None:
                    _, display_img = self.apply_preprocessing(img)
                    out_path = os.path.join(output_dir, fname)
                    cv2.imwrite(out_path, display_img)
                    count += 1
        
        messagebox.showinfo("Batch Complete", f"Successfully processed {count} images!")
        self.status.set(f"Batch processed {count} images")
    
    # === RESET ===
    def reset_all(self):
        self.clahe_slider.set(2.0)
        self.denoise_slider.set(10)
        self.thresh_slider.set(11)
        self.c_slider.set(2)
        self.gamma_slider.set(1.0)
        self.brightness_slider.set(0)
        self.contrast_slider.set(1.0)
        self.gaussian_slider.set(0)
        self.canny_low_slider.set(50)
        self.canny_high_slider.set(150)
        self.morph_slider.set(5)
        self.morph_iter_slider.set(1)
        self.morph_combo.set("None")
        self.morph_type = "None"
        
        self.apply_clahe.set(True)
        self.apply_denoise.set(True)
        self.apply_threshold.set(True)
        self.apply_gaussian.set(False)
        self.apply_canny.set(False)
        self.apply_morph.set(False)
        
        self.update_params()
        self.status.set("All parameters reset to defaults")

if __name__ == "__main__":
    root = tk.Tk()
    app = CVStudio(root)
    root.mainloop()
