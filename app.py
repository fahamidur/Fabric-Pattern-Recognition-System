import gradio as gr
from fastai.vision.all import *
import platform
import pathlib
import base64
import os

# --- 1. Setup & Model ---
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

learn = load_learner('model.pkl')
categories = learn.dls.vocab

def predict(img):
    img_fastai = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img_fastai)
    return dict(zip(categories, map(float, probs)))

# --- 2. Helper for Sidebar Images ---
def get_b64(img_path):
    if not os.path.exists(img_path): return ""
    with open(img_path, "rb") as f:
        return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

# --- 3. Reference Data ---
fabric_samples = [
    ("examples/Ogee.jpg", "Ogee", "Curved, diamond-like shapes."),
    ("examples/Ikat.jpg", "Ikat", "Blurred edges from resist-dyeing."),
    ("examples/Animal_Print.jpg", "Animal Print", "Leopard or zebra patterns."),
    ("examples/Quatrefoil.jpg", "Quatrefoil", "Four-lobed symmetrical shapes."),
    ("examples/Herringbone.jpg", "Herringbone", "V-shaped heavy weave pattern."),
    ("examples/Paisley.jpg", "Paisley", "Ornate teardrop-shaped motifs."),
    ("examples/Suzani.jpg", "Suzani", "Large embroidered floral discs."),
    ("examples/Chevron.jpg", "Chevron", "Continuous V-shaped zigzags."),
    ("examples/Plaid_Checkered.jpg", "Plaid", "Crisscrossed horizontal/vertical bands."),
    ("examples/Dot_Polka_Dot.jpg", "Polka Dot", "Evenly spaced filled circles."),
    ("examples/Gingham.jpg", "Gingham", "Checkered white and color weave."),
    ("examples/Matelasse.jpg", "Matelasse", "Quilted look without padding."),
    ("examples/Jacobean.jpg", "Jacobean", "Stylized floral scrolls."),
    ("examples/Houndstooth.jpg", "Houndstooth", "Abstract four-pointed check shapes."),
    ("examples/Damask.jpg", "Damask", "Ornate floral woven patterns.")
]

# Generate clean sidebar HTML
sidebar_html = "<div style='max-height: 310px; overflow-y: auto; padding-right: 10px;'>"
for path, name, desc in fabric_samples:
    b64 = get_b64(path)
    sidebar_html += f"""
    <div style='display: flex; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #333; padding-bottom: 8px;'>
        <img src='{b64}' style='width: 50px; height: 50px; border-radius: 4px; object-fit: cover; margin-right: 12px;'>
        <div>
            <div style='font-weight: bold; color: white; font-size: 0.9rem;'>{name}</div>
            <div style='font-size: 0.75rem; color: #aaa;'>{desc}</div>
        </div>
    </div>
    """
sidebar_html += "</div>"

# --- 4. UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown("# 🧶 Textile Pattern Classifier")
    
    with gr.Row():
        # LEFT COLUMN: Predictor
        with gr.Column(scale=2):
            input_img = gr.Image(type="pil", label="Upload Fabric Image",height=400)
            predict_btn = gr.Button("Analyze Pattern", variant="primary")
            output_label = gr.Label(num_top_classes=5, label="Classification Results")
            
        # RIGHT COLUMN: Reference Guide
        with gr.Column(scale=1):
            gr.Markdown("### 📚 Pattern Reference: 15 Classes")
            gr.HTML(sidebar_html)
            
            # Interactive thumbnails (Keep them small at the bottom)
            gr.Markdown("#### Quick Test")
            gr.Examples(
                examples=[s[0] for s in fabric_samples],
                inputs=input_img,
                label=None,
                examples_per_page=15
            )

    predict_btn.click(fn=predict, inputs=input_img, outputs=output_label)

if __name__ == "__main__":
# 3. The Launch Parameters
    demo.launch(
        theme=gr.themes.Default(primary_hue="orange", secondary_hue="slate"),
        ssr_mode=False,              # Fixes the 'Invalid file descriptor' error
        show_error=True,             
        server_name="0.0.0.0",       
        server_port=7860,            
        allowed_paths=["./examples"],  
        share=False,                 
        debug=False                  
    )
