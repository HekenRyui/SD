from google.colab import drive
drive.mount('/content/drive')
import os 
import random
import uuid
import torch
import numpy as np
import gradio as gr
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

# Constants
MAX_SEED = np.iinfo(np.int32).max
SAVE_DIR = "/content/images"
MODELS_PATH = "/content/drive/MyDrive/StableDiffusion/models"
DEFAULT_MODEL_PATH = f'{MODELS_PATH}/model_deffault.safetensors'

# Setup
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)  # Ensure the models path exists
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load default model
os.system(f'wget -O {DEFAULT_MODEL_PATH} "https://civitai.com/api/download/models/128078?type=Model&format=SafeTensor&size=pruned&fp=fp16"')
pipe = StableDiffusionXLPipeline.from_single_file(DEFAULT_MODEL_PATH, use_safetensors=True, torch_dtype=torch.float16).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
print("\033[1;32mDefault Model Loaded!\033[0m")

# Initialize a global list to store generated image paths
generated_images = []

# Download model function
def download_model(url):
    model_filename = url.split("/")[-1]
    model_path = os.path.join(MODELS_PATH, model_filename)
    os.system(f'wget -O {model_path} "{url}"')
    return f"Model downloaded: {model_path}"

# Load selected model
def load_model(model_name):
    model_path = os.path.join(MODELS_PATH, model_name)
    global pipe  # We will reinitialize the pipeline
    pipe = StableDiffusionXLPipeline.from_single_file(model_path, use_safetensors=True, torch_dtype=torch.float16).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    return f"Model loaded: {model_name}"

# Infer function
def infer(prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps):
    if seed == -1:  # -1 indicates random seed
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale, 
        num_inference_steps=num_inference_steps, 
        width=width, 
        height=height,
        generator=generator,
    ).images[0]
    
    image_filename = f"{uuid.uuid4()}.png"
    image_path = os.path.join(SAVE_DIR, image_filename)
    image.save(image_path)
    
    # Add the new image to the list of generated images
    generated_images.append(image_path)
    
    return image, generated_images

# Get available models
def get_available_models():
    models = [f for f in os.listdir(MODELS_PATH) if f.endswith(".safetensors")]
    if not models:
        models.append('No models available')
    return models

# UI setup
css = """
#col-container {
    margin: 0 auto;
    max-width: 580px;
}
footer {
    display: none !important;
}
#image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
    grid-gap: 10px;
    margin-top: 20px;
}
#image-grid img {
    width: 100%;
    height: auto;
    object-fit: cover;
    border-radius: 8px;
}
"""

examples = [
    "a cat",
    "a cat in the hat",
    "a cat in the cowboy hat",
]

with gr.Blocks(css=css, theme='ParityError/Interstellar') as app:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
    # Stable Diffusion with python

    Google Colab's free tier offers about 4 hours of GPU usage per day. No authorization, no data storing or tracking. Your session data will be deleted when this session closes.
""")

        with gr.Group():
            with gr.Row():
                prompt = gr.Text(label="Prompt", show_label=False, lines=1, max_lines=7,
                                 placeholder="Enter your prompt", container=False, scale=4)
                run_button = gr.Button("🚀 Run", scale=1, variant='primary')
        
        result = gr.Image(label="Result", show_label=False)
        gr.Examples(examples=examples, inputs=[prompt])
        
        with gr.Group():
            with gr.Accordion("⚙️ Settings", open=False):
                negative_prompt = gr.Text(label="Negative prompt", placeholder="Enter a negative prompt",
                                          lines=3, value='lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature')
                
                seed = gr.Slider(label="Seed (-1 for random)", minimum=-1, maximum=MAX_SEED, step=1, value=-1)
                
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=256, maximum=3840, step=64, value=1024)
                    height = gr.Slider(label="Height", minimum=256, maximum=2160, step=64, value=1024)
                
                with gr.Row():
                    guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=10.0, step=0.1, value=5.0)
                    num_inference_steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=20)
        
        with gr.Accordion("⚙️ Advanced", open=False):
            with gr.Row():
                model_url = gr.Textbox(label="Model URL", placeholder="Enter URL to download model")
                download_button = gr.Button("Download Model")
            download_status = gr.Textbox(label="Download Status", interactive=False)
            
            download_button.click(
                fn=download_model,
                inputs=[model_url],
                outputs=[download_status]
            )
            
            available_models = gr.Dropdown(label="Select Model", choices=get_available_models(), value="model_deffault.safetensors")
            load_button = gr.Button("Load Model")
            load_status = gr.Textbox(label="Load Status", interactive=False)
            
            load_button.click(
                fn=load_model,
                inputs=[available_models],
                outputs=[load_status]
            )

        # Section to display the generated image gallery
        gr.Markdown("# Generated Images")
        image_gallery = gr.Gallery(elem_id="image-grid").style(grid=[3], height="auto")
        
    run_button.click(
        fn=infer,
        inputs=[prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps],
        outputs=[result, image_gallery],
    )

if __name__ == "__main__":
    app.launch(share=True, inline=False, inbrowser=False, debug=True)
