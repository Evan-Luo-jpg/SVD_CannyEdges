import os
import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder
from peft.utils import get_peft_model_state_dict
from diffusers.utils import load_image, convert_state_dict_to_diffusers
from modified_pipeline import CustomSVDPipeline
from src.modified_unet import UNetSpatioTemporalConditionModel


class EdgeEncoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),  # halve spatial dims
            nn.MaxPool2d(2),  # halve spatial dims
            nn.Conv2d(32, out_channels, 3, padding=1),
        )

    def forward(self, edge_map):  # (B, 1, H, W)
        return self.encoder(edge_map)  # (B, out_channels, H, W)


# ----- CONFIGURATION -----
BASE_DIR = os.path.expanduser(os.environ.get("BASE_DIR", "~/scratch/CVFinal"))
PRETRAINED_MODEL = "stabilityai/stable-video-diffusion-img2vid"
LORA_PATH = f"{BASE_DIR}/modified_lora"
IMAGE_PATH = f"{BASE_DIR}/SVD_Xtend/demo.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load Model Components -----
print("Loading model...")
unet = UNetSpatioTemporalConditionModel.from_pretrained(
    PRETRAINED_MODEL, subfolder="unet", low_cpu_mem_usage=True, variant="fp16"
).to(DEVICE, dtype=torch.float16)
unet.load_attn_procs(LORA_PATH)  # loads LoRA weights into UNet

vae = AutoencoderKLTemporalDecoder.from_pretrained(
    PRETRAINED_MODEL, subfolder="vae", variant="fp16"
).to(DEVICE, dtype=torch.float16)

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    PRETRAINED_MODEL, subfolder="image_encoder", variant="fp16"
).to(DEVICE, dtype=torch.float16)

feature_extractor = CLIPImageProcessor.from_pretrained(PRETRAINED_MODEL, subfolder="feature_extractor")

# Optional: load edge encoder used in training
edge_encoder = EdgeEncoder(out_channels=320).to(DEVICE, dtype=torch.float16)
edge_encoder.load_state_dict(torch.load(f"{BASE_DIR}/modified-checkpoint-500/edge_encoder.pt"))  # if saved separately
edge_encoder.eval()

# ----- Create Pipeline -----
pipeline = CustomSVDPipeline.from_pretrained(
    PRETRAINED_MODEL,
    unet=unet,
    vae=vae,
    image_encoder=image_encoder,
    feature_extractor=feature_extractor,
    torch_dtype=torch.float16,
).to(DEVICE)

# ----- Prepare Input Image and Edge Map -----
image = load_image(IMAGE_PATH).resize((320, 192))
rgb_array = np.array(image)
gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 100, 200)
edge_tensor = torch.from_numpy(edges).float().unsqueeze(0).unsqueeze(0) / 127.5 - 1  # (1, 1, H, W)
edge_tensor = edge_tensor.to(DEVICE, dtype=torch.float16)

control_tensor = edge_encoder(edge_tensor)
num_frames = 8

# ----- Run Inference -----
print("Generating video...")
with torch.no_grad():
    out = pipeline(
        image,
        height=64,
        width=128,
        num_frames=num_frames,
        motion_bucket_id=127,
        fps=7,
        noise_aug_strength=0.02,
        control_input=control_tensor,
    )["frames"][0]

# ----- Save as GIF -----
out_path = os.path.join(BASE_DIR, "modified_inference_output.gif")
out[0].save(out_path, save_all=True, append_images=out[1:], duration=500, loop=0)
print(f"Saved inference video to {out_path}")
