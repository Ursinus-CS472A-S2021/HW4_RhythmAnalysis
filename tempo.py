import numpy as np
import matplotlib.pyplot as plt
import glob
from novfn import *

def autocorr(x):
    """
    Fast autocorrelation based on the Wiener-Khinchin Theorem, which allows us
    to use the fast fourier transform of the input to compute the autocorrelation

    Parameters
    ----------
    x: ndarray(N)
        Array on which to compute the autocorrelation
    
    Returns
    -------
    ndarray(N): The autocorrelation
    """
    N = len(x)
    xpad = np.zeros(N*2)
    xpad[0:N] = x
    F = np.fft.fft(xpad)
    FConv = np.real(F)**2 + np.imag(F)**2 # Fourier transform of the convolution of x and its reverse
    return np.real(np.fft.ifft(FConv)[0:N])


def get_fourier_tempo(novfn, hop_length, sr):
    """
    Parameters
    ----------
    novfn: ndarray(N)
        The novelty function
    hop_length: int
        Hop length, in audio samples, between the samples in the audio
        novelty function
    sr: int
        Sample rate of the audio
    
    Returns
    -------
    float: Estimate, in beats per minute, of the tempo
    """
    ## TODO: Fill this in
    return 0


def dft_warped(x):
    """
    Compute the DFT, resampled to coincide with the samples in 
    the autocorrelation

    Parameters
    ----------
    x: ndarray(N)
        Signal on which to compute the warped DFT
    
    Returns
    -------
    ndarray(N): The warped DFT
    """
    f = np.abs(np.fft.fft(x))
    ret = np.zeros_like(f)
    ## TODO: Fill this in to warp the samples of f to coincide 
    ## with the samples of an autocorrelation function
    return ret


def get_acf_dft_tempo(novfn, hop_length, sr):
    """
    Estimate the tempo, in bpm, based on a combination of a warped
    DFT and the autocorrelation of a novelty function, as described in 
    section 3.1.1 of [1]
    [1] "Template-Based Estimation of Time-Varying Tempo." Geoffroy Peeters.
            EURASIP Journal on Advances in Signal Processing

    Parameters
    ----------
    novfn: ndarray(N)
        The novelty function
    hop_length: int
        Hop length, in audio samples, between the samples in the audio
        novelty function
    sr: int
        Sample rate of the audio
    
    Returns
    -------
    float: Estimate, in beats per minute, of the tempo
    """
    ## TODO: Fill this in to use the product of warped fourier and 
    ## autocorrelation
    return 0


def evaluate_tempos(f_novfn, f_tempofn, hop_length, sr, tol = 0.08):
    """
    Evaluate the example dataset of 20 clips from MIREX
    https://www.music-ir.org/mirex/wiki/2019:Audio_Tempo_Estimation
    based on a particular novelty function and tempo function working together

    Parameters
    ----------
    f_novfn: (x, sr) -> ndarray(N)
        A function from audio samples and their sample rate to an audio novelty function
    f_tempofn: (novfn, hop_length, sr) -> float
        A function to estimate the tempo from an audio novelty function.  The hop
        length and sample rate are needed to infer absolute beats per minute
    hop_length: int
        The hop length, in audio samples, between the samples of the audio novelty
        function
    sr: int
        Sample rate of the audio
    tol: float
        The fraction of error tolerated between ground truth tempos to declare
        success on a particular track
    
    Returns
    -------
    A pandas dataframe with the filename, 
    """
    import pandas as pd
    from collections import OrderedDict
    files = glob.glob("Tempo/train/*.txt")
    num_close = 0
    names = []
    gt_tempos = []
    est_tempos = []
    close_enough = []
    for f in files:
        fin = open(f)
        tempos = [float(x) for x in fin.readlines()[0].split()][0:2]
        f = f.split(".txt")[0] + ".wav"
        x, _ = librosa.load(f, sr=sr)
        novfn = f_novfn(x, sr)
        tempo = f_tempofn(novfn, hop_length, sr)
        close = np.abs(tempo-tempos[0])/tempos[0] < tol
        close = close or np.abs(tempo-tempos[1])/tempos[1] < tol
        close_enough.append(close)
        names.append(f.split("/")[-1])
        gt_tempos.append(tempos)
        est_tempos.append(tempo)
        if close:
            num_close += 1
    nums = np.array([int(s.split(".wav")[0].split("train")[-1]) for s in names])
    idx = np.argsort(nums)
    names = [names[i] for i in idx]
    gt_tempos = [gt_tempos[i] for i in idx]
    est_tempos = [est_tempos[i] for i in idx]
    close_enough = [close_enough[i] for i in idx]
    df = pd.DataFrame(OrderedDict([("names", names), ("Ground-Truth Tempos", gt_tempos), \
                                    ("Estimated Tempos", est_tempos), ("Close Enough", close_enough)]))
    print("{} / {}".format(num_close, len(files)))
    return df