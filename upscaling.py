import sys
import os
import glob
import torchvision.transforms.functional as T_F
from types import ModuleType

# --- THE MONKEY PATCH (Keep this at the top) ---
fake_module = ModuleType("torchvision.transforms.functional_tensor")
fake_module.rgb_to_grayscale = T_F.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = fake_module

import cv2
import torch
import warnings
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

warnings.filterwarnings("ignore")

def batch_upscale(input_folder, output_folder, scale=4):
    # 1. Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # 2. Setup the model (optimized for GTX 1650)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=scale,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=400, 
        half=False
    )

    # 3. Find all images (jpg, jpeg, png)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    image_list = []
    for ext in extensions:
        image_list.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_list:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_list)} images. Starting batch process...")

    # 4. Loop through images
    for i, img_path in enumerate(image_list):
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, f"upscaled_{filename}")
        
        print(f"[{i+1}/{len(image_list)}] Processing: {filename}...", end="\r")
        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        try:
            output, _ = upsampler.enhance(img, outscale=scale)
            cv2.imwrite(save_path, output)
        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    print(f"\n\nFinished! All images are in: {output_folder}")

if __name__ == "__main__":
    # Point these to your folders
    input_dir = "inputs"   # Create a folder named 'inputs' and put your photos there
    output_dir = "results"
    
    batch_upscale(input_dir, output_dir)