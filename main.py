import numpy as np
import pathlib  
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile
import os
import torch
from PIL import Image
import torchvision.transforms as transforms

data = pathlib.Path("Audio_Samples")
output_dir = pathlib.Path("Download")
tensor_dir = pathlib.Path("Input")

# Image transformation
transform_f = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize all images to 64x64
    transforms.ToTensor(),       # Convert to tensor and normalize to [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize: mean=0.5, std=0.5
])

def generate_stft(audio_file):
    relative_path = audio_file.relative_to(data)
    
    amplitude_filename = output_dir / f"{relative_path.parent}/{relative_path.stem}_amplitude.png"
    amplitude_filename.parent.mkdir(parents=True, exist_ok=True)

    phase_filename = output_dir / f"{relative_path.parent}/{relative_path.stem}_phase.png"
    phase_filename.parent.mkdir(parents=True, exist_ok=True)
    
    if phase_filename.exists() and amplitude_filename.exists():
        # print(f"Skipping {audio_file} as both spectrograms already exist.")
        return
    
    # Load the audio file (replace 'your_audio_file.wav' with your file path)
    sampling_rate, audio_signal = wavfile.read(audio_file)

    # Perform STFT to get both amplitude and phase information
    frequencies, times, Zxx = stft(audio_signal, fs=sampling_rate, nperseg=1024)

    # Compute the amplitude (magnitude) and phase spectrograms
    amplitude_spectrogram = np.abs(Zxx)
    phase_spectrogram = np.angle(Zxx)

    # Plot amplitude spectrogram
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, frequencies, 20 * np.log10(amplitude_spectrogram), shading='gouraud')
    plt.title("Amplitude Spectrogram (dB)")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.colorbar(label="Amplitude (dB)")
    plt.savefig(amplitude_filename)
    plt.close()

    # Plot phase spectrogram
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, frequencies, phase_spectrogram, shading='gouraud')
    plt.title("Phase Spectrogram")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.colorbar(label="Phase (radians)")
    plt.savefig(phase_filename)
    plt.close()

    # print(relative_path)

if __name__ == "__main__":
    # Iterates through every audio sample we have
    for audio_source in data.iterdir():
        for ir_folder in audio_source.iterdir():
            angle = int(os.path.basename(ir_folder).split('IR_')[-1])

            tensor_file = tensor_dir / audio_source.name / f"{angle}_tensor.pt"

            if tensor_file.exists():
                # print(f"Skipping {output_file} as both spectrograms already exist.")
                continue
            
            tensor_file.parent.mkdir(parents=True, exist_ok=True)

            for file in ir_folder.iterdir():
                generate_stft(file)
            
            image_files = sorted([os.path.join(output_dir, audio_source.name, ir_folder.name, file) 
                      for file in os.listdir(output_dir / audio_source.name / ir_folder.name) 
                      if file.endswith(".png")])
            
            ir_spectrograms = []
            for image_file in image_files:
                image = Image.open(image_file).convert("L")  # Convert to grayscale
                image = transform_f(image)  # Apply transformations
                ir_spectrograms.append(image)
            
            if len(image_files) == 14:  # Ensure all 14 spectrograms are present
                spectrogram_tensor = torch.stack(ir_spectrograms, dim=0)  # Stack into a single tensor
                # print(spectrogram_tensor.shape)
                torch.save((spectrogram_tensor.squeeze()), tensor_file)
                print(tensor_file)
            
            for image_file in image_files:
                os.remove(image_file)
        
        for ir_folder in audio_source.iterdir():
            for file in ir_folder.iterdir():
                os.remove(file)
            os.rmdir(ir_folder)
        os.rmdir(audio_source)