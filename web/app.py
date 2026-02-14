# web/app.py
import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import zipfile
import io

# Assuming these are in the parent folder (adjust import if needed)
try:
    from core.pipeline import PreprocessingPipeline
    from core.augmentation import AugmentationEngine
    from core.config import PipelineConfig
except ImportError:
    # If running directly from /web folder, try this instead:
    from ..core.pipeline import PreprocessingPipeline
    from ..core.augmentation import AugmentationEngine
    from ..core.config import PipelineConfig

pipeline = PreprocessingPipeline()
augmentor = AugmentationEngine()

# ────────────────────────────────────────────────
#   SINGLE IMAGE PROCESSING
# ────────────────────────────────────────────────
def process_image(
    image,
    apply_clahe, clip_limit,
    apply_denoise, denoise_strength,
    apply_gaussian, gaussian_ksize,
    apply_sharpen, sharpen_strength,
    apply_bilateral,
    apply_threshold, block_size, c_value,
    apply_canny, canny_low, canny_high,
    apply_morph, morph_kernel, morph_iter,
    brightness, contrast, gamma,
    resize_enabled, target_width, target_height
):
    if image is None:
        return None, "No image uploaded"

    # Convert PIL → OpenCV BGR
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Build config dict
    config = {
        "apply_clahe": apply_clahe,
        "clip_limit": clip_limit,
        "apply_denoise": apply_denoise,
        "denoise_strength": denoise_strength,
        "apply_gaussian": apply_gaussian,
        "gaussian_ksize": gaussian_ksize,
        "apply_sharpen": apply_sharpen,
        "sharpen_strength": sharpen_strength,
        "apply_bilateral": apply_bilateral,
        "apply_threshold": apply_threshold,
        "block_size": block_size,
        "c": c_value,
        "apply_canny": apply_canny,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "apply_morph": apply_morph,
        "morph_kernel_size": morph_kernel,
        "morph_iterations": morph_iter,
        "brightness": brightness,
        "contrast": contrast,
        "gamma": gamma,
        "target_size": (target_width, target_height) if resize_enabled else None,
        # roi_coords omitted in web version for simplicity
    }

    pipeline.update_config(config)
    _, processed = pipeline.preprocess(img)

    # Convert back to RGB for Gradio
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    else:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    return processed, "Done"


# ────────────────────────────────────────────────
#   BATCH AUGMENTATION → ZIP DOWNLOAD
# ────────────────────────────────────────────────
def generate_augmented_dataset(images, num_augs_per_image: int):
    if not images:
        return None, "No images uploaded"

    all_augmented = []

    for idx, pil_img in enumerate(images):
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        variants = augmentor.apply_random(img_bgr, n_variants=num_augs_per_image)

        for i, variant in enumerate(variants):
            # Convert back to RGB for Gradio + naming
            rgb_variant = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
            all_augmented.append((rgb_variant, f"img_{idx+1:03d}_aug_{i+1:02d}.png"))

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_array, filename in all_augmented:
            success, buffer = cv2.imencode(".png", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            if success:
                zf.writestr(filename, buffer.tobytes())

    zip_buffer.seek(0)
    return zip_buffer, f"Generated {len(all_augmented)} images"


# ────────────────────────────────────────────────
#   GRADIO INTERFACE
# ────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Dark(), title="CV Preprocessing Studio – Web") as demo:
    gr.Markdown("""
    # 🖼️ CV Preprocessing Studio  
    Interactive preprocessing + augmentation tool  
    Built for document cleaning, OCR prep, medical imaging, defect detection, ...
    """)

    with gr.Tab("🛠 Preprocess Single Image"):
        with gr.Row():
            with gr.Column(scale=1):
                inp_image = gr.Image(type="pil", label="Upload Image")

                gr.Markdown("### Contrast & Denoise")
                clahe = gr.Checkbox(label="CLAHE", value=True)
                clip = gr.Slider(0.1, 8.0, 2.0, step=0.1, label="Clip Limit")
                denoise = gr.Checkbox(label="Denoise", value=True)
                denoise_str = gr.Slider(1, 50, 10, label="Denoise Strength")
                bilateral = gr.Checkbox(label="Bilateral Filter", value=False)

                gr.Markdown("### Sharpen / Blur")
                sharpen = gr.Checkbox(label="Sharpen", value=False)
                sharpen_str = gr.Slider(0.5, 3.0, 1.0, step=0.1, label="Sharpen Strength")
                gauss = gr.Checkbox(label="Gaussian Blur", value=False)
                gauss_k = gr.Slider(3, 21, 5, step=2, label="Kernel Size")

                gr.Markdown("### Threshold / Edges / Morphology")
                thresh = gr.Checkbox(label="Adaptive Threshold", value=True)
                block = gr.Slider(3, 31, 11, step=2, label="Block Size")
                c_val = gr.Slider(0, 20, 2, label="Constant C")
                canny = gr.Checkbox(label="Canny Edges", value=False)
                c_low = gr.Slider(0, 255, 50, label="Canny Low")
                c_high = gr.Slider(0, 255, 150, label="Canny High")
                morph = gr.Checkbox(label="Morphology Clean", value=False)
                morph_k = gr.Slider(3, 21, 5, step=2, label="Kernel Size")
                morph_it = gr.Slider(1, 8, 1, label="Iterations")

                gr.Markdown("### Color / Gamma / Resize")
                bright = gr.Slider(-100, 100, 0, label="Brightness")
                contr = gr.Slider(0.5, 2.0, 1.0, step=0.05, label="Contrast")
                gam = gr.Slider(0.3, 3.0, 1.0, step=0.1, label="Gamma")
                do_resize = gr.Checkbox(label="Resize final output", value=False)
                w = gr.Number(value=640, label="Width")
                h = gr.Number(value=640, label="Height")

                btn_process = gr.Button("Process Image", variant="primary")

            with gr.Column(scale=1):
                out_image = gr.Image(label="Processed Result", type="numpy")
                status = gr.Textbox(label="Status", interactive=False)

        btn_process.click(
            process_image,
            inputs=[
                inp_image, clahe, clip, denoise, denoise_str,
                gauss, gauss_k, sharpen, sharpen_str, bilateral,
                thresh, block, c_val, canny, c_low, c_high,
                morph, morph_k, morph_it, bright, contr, gam,
                do_resize, w, h
            ],
            outputs=[out_image, status]
        )

    with gr.Tab("🔄 Generate Augmented Dataset"):
        gr.Markdown("Upload one or more images → get many realistic variants")
        inp_files = gr.File(file_count="multiple", file_types=["image"], label="Upload Images")
        num_augs = gr.Slider(1, 50, 8, step=1, label="Variants per image")
        btn_gen = gr.Button("Generate & Download ZIP", variant="primary")
        out_zip = gr.File(label="Download Augmented Dataset (.zip)")

        btn_gen.click(
            generate_augmented_dataset,
            inputs=[inp_files, num_augs],
            outputs=[out_zip, gr.Textbox(label="Status (hidden)")]
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # ← allows access from local network if needed
        # inbrowser=True,       # ← auto open browser (optional)
    )
