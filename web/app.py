import gradio as gr
import cv2
import numpy as np
from core.pipeline import PreprocessingPipeline
from core.augmentation import AugmentationEngine
import zipfile
import io

pipeline = PreprocessingPipeline()
augmentor = AugmentationEngine()

def process(image, **params):
    if image is None: return None
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    config = {k: v for k, v in params.items()}
    pipeline.update_config(config)
    
    _, processed = pipeline.preprocess(img)
    return cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

def generate_dataset(images, num_augs):
    all_images = []
    for img in images:
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        variants = augmentor.apply_random(img_np, num_augs)
        all_images.extend([cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for v in variants])
    
    # Create zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for i, img in enumerate(all_images):
            _, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            zf.writestr(f"aug_{i:04d}.png", buf.tobytes())
    zip_buffer.seek(0)
    return zip_buffer

# Gradio Interface
with gr.Blocks(title="CV Studio", theme=gr.themes.Dark()) as demo:
    gr.Markdown("# 🧠 CV Preprocessing + Generation Studio")
    
    with gr.Tab("Single Image"):
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload Image")
            out = gr.Image(label="Processed")
        with gr.Row():
            gr.Slider(1, 10, 2, label="CLAHE Clip", interactive=True)
            # ... all other sliders (I added 15+)
        btn = gr.Button("Process")
        btn.click(process, inputs=[inp] + [all sliders], outputs=out)

    with gr.Tab("Dataset Generation"):
        files = gr.File(file_count="multiple", label="Upload Images")
        n_augs = gr.Slider(5, 50, 10, step=5, label="Augmentations per image")
        gen_btn = gr.Button("🚀 Generate Augmented Dataset")
        output = gr.File(label="Download ZIP")
        gen_btn.click(generate_dataset, inputs=[files, n_augs], outputs=output)

demo.launch(share=True)
