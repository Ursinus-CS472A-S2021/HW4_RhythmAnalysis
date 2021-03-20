import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.ndimage import maximum_filter


def sonify_novfn(novfn, hop_length):
    """
    Shape noise according to a novelty function

    Parameters
    ----------
    novfn: ndarray(N)
        A novelty function with N samples
    hop_length: int
        The hop length, in audio samples, between each novelty function sample
    
    Returns
    -------
    ndarray(N*hop_length)
        Shaped noise according to the audio novelty function
    """
    x = np.random.randn(len(novfn)*hop_length)
    for i in range(len(novfn)):
        x[i*hop_length:(i+1)*hop_length] *= novfn[i]
    return x


def get_novfn(x, sr, hop_length=512, win_length=1024):
    """
    Our vanilla audio novelty function from module 16
    https://ursinus-cs472a-s2021.github.io/Modules/Module16/Video1

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    sr: int
        Sample rate
    hop_length: int
        Hop length between frames in the stft
    win_length: int
        Window length between frames in the stft
    """
    S = librosa.stft(x, hop_length=hop_length, n_fft=win_length)
    S = np.abs(S)
    Sdb = librosa.amplitude_to_db(S,ref=np.max)
    N = Sdb.shape[0]
    novfn = np.zeros(N-1) # Pre-allocate space to hold some kind of difference between columns
    diff = Sdb[:, 1::] - Sdb[:, 0:-1]
    diff[diff < 0] = 0 # Cut out the differences that are less than 0
    novfn = np.sum(diff, axis=0)
    return novfn


def get_mel_filterbank(K, win_length, sr, min_freq, max_freq, n_bins):
    """
    Compute a mel-spaced filterbank
    
    Parameters
    ----------
    K: int
        Number of non-redundant frequency bins
    win_length: int
        Window length (should be around 2*K)
    sr: int
        The sample rate, in hz
    min_freq: int
        The center of the minimum mel bin, in hz
    max_freq: int
        The center of the maximum mel bin, in hz
    n_bins: int
        The number of mel bins to use
    
    Returns
    -------
    ndarray(n_bins, K)
        The triangular mel filterbank
    """
    bins = np.logspace(np.log10(min_freq), np.log10(max_freq), n_bins+2)*win_length/sr
    bins = np.array(np.round(bins), dtype=int)
    Mel = np.zeros((n_bins, K))
    for i in range(n_bins):
        i1 = bins[i]
        i2 = bins[i+1]
        if i1 == i2:
            i2 += 1
        i3 = bins[i+2]
        if i3 <= i2:
            i3 = i2+1
        tri = np.zeros(K)
        tri[i1:i2] = np.linspace(0, 1, i2-i1)
        tri[i2:i3] = np.linspace(1, 0, i3-i2)
        Mel[i, :] = tri
    return Mel


def get_superflux_novfn(x, sr, hop_length=512, win_length=1024, max_win = 1, mu=1, Gamma=10):
    """
    Implement the superflux audio novelty function, as described in [1]
    [1] "Maximum Filter Vibrato Suppresion for Onset Detection," 
            Sebastian Boeck, Gerhard Widmer, DAFX 2013

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    sr: int
        Sample rate
    hop_length: int
        Hop length between frames in the stft
    win_length: int
        Window length between frames in the stft
    max_win: int
        Amount by which to apply a maximum filter
    mu: int
        The gap between windows to compare
    Gamma: float
        An offset to add to the log spectrogram; log10(|S| + Gamma)
    """
    S = librosa.stft(x, hop_length=hop_length, n_fft=win_length)
    S = np.abs(S)
    ## TODO: Fill this in