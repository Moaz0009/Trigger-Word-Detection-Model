import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment

# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx
def compute_spectrogram(samples, fs=16000):
    """
    Compute a spectrogram from raw audio samples.
    
    Arguments:
    samples     -- NumPy array of audio samples.
    sample_rate -- The sampling rate of the audio.
    
    Returns:
    pxx -- The computed spectrogram.
    """
    nfft = 200    # Length of each window segment.
    noverlap = 120  # Overlap between windows.
    
    # Compute the spectrogram using matplotlib's specgram.
    # The function returns (pxx, freqs, bins, im). We only need pxx here.
    freqs, times, pxx = signal.spectrogram(np.array(samples), fs=fs, nperseg=nfft, noverlap=noverlap)
    return pxx
# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def load_raw_audio(path):
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir(path + "dd"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav(path + "dd/" + filename)
            activates.append(activate)
    for filename in os.listdir(path + "bk"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav(path + "bk/" + filename)
            backgrounds.append(background)
    for filename in os.listdir(path + "rr"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav(path + "rr/" + filename)
            negatives.append(negative)
    return activates, negatives, backgrounds