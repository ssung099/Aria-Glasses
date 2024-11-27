import numpy as np
import pathlib
import torch
import os
from PIL import Image
import torchvision.transforms as transforms

# Paths
output_dir = pathlib.Path("Inputs")
dataset = pathlib.Path("Download")
output_dir.mkdir(parents=True, exist_ok=True)  # Create output folder if it doesn't exist

# Image transformation
transform_f = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize all images to 64x64
    transforms.ToTensor(),       # Convert to tensor and normalize to [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize: mean=0.5, std=0.5
])

def create_input():
    for audio_source in dataset.iterdir():
        for ir_folder in audio_source.iterdir():
            angle = int(os.path.basename(ir_folder).split('IR_')[-1])

            output_file = output_dir / audio_source.name / f"{angle}_tensor.pt"
            # print(output_file)
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if output_file.exists():
                # print(f"Skipping {output_file} as both spectrograms already exist.")
                continue

            image_files = sorted([os.path.join(ir_folder, file) for file in os.listdir(ir_folder) if file.endswith(".png")])

            ir_spectrograms = []
            for image_file in image_files:
                image = Image.open(image_file).convert("L")  # Convert to grayscale
                image = transform_f(image)  # Apply transformations
                ir_spectrograms.append(image)
            
            if len(image_files) == 14:  # Ensure all 14 spectrograms are present
                spectrogram_tensor = torch.stack(ir_spectrograms, dim=0)  # Stack into a single tensor
                # print(spectrogram_tensor.shape)
                torch.save((spectrogram_tensor.squeeze()), output_file)
                print(output_file)

if __name__ == "__main__":
    create_input()
