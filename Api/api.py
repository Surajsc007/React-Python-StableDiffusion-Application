from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import base64 
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from transformers import CLIPImageProcessor
from transformers import CLIPProcessor

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

device = "cuda"
model_id = "CompVis/stable-diffusion-v1-4"

# Initialize the CLIPImageProcessor
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Initialize the CLIP text and image processors
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Load the diffuser model
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    revision="fp16",
    torch_dtype=torch.float32,
    image_processor=image_processor,
    text_processor=clip_processor,
    device=device
)

@app.get("/")
def generate(prompt: str): 
    with autocast(device): 
        image = pipe(prompt, guidance_scale=8.5).images[0]

    image.save("testimage.png")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr, media_type="image/png")
