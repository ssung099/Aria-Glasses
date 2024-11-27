import numpy as np
import pathlib  
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile

data = pathlib.Path("Audio_Samples")
output_dir = pathlib.Path("Download")
check_dir = pathlib.Path("Spectrograms")

def generate_stft(audio_file):
    relative_path = audio_file.relative_to(data)
    
    check_amp = check_dir / f"{relative_path.parent}/{relative_path.stem}_amplitude.png"
    check_phase = check_dir / f"{relative_path.parent}/{relative_path.stem}_phase.png"
    
    if check_amp.exists() and check_phase.exists():
        # print(f"Skipping {audio_file} as both spectrograms already exist.")
        return
    
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

    print(relative_path)

if __name__ == "__main__":
    # Iterates through every audio sample we have
    for f in data.rglob('*'):
        if f.is_file():
            file = f.relative_to(data)
            # print(file)
            generate_stft(f)