import numpy as np
import librosa
import matplotlib.pyplot as plt
from novfn import *
from tempo import *

def get_gt_beats(filename, anno=0):
    """
    Load in the ground truth beats for some clip
    
    Parameters
    ----------
    filename: string
        Path to annotations
    anno: int
        Annotator number
    """
    fin = open(filename)
    lines = fin.readlines()
    fin.close()
    beats = lines[anno]
    beats = np.array([float(x) for x in beats.split()])
    return beats

def plot_beats(novfn, beats, sr, hop_length):
    """
    Plot the location of beats superimposed on an audio novelty
    function

    Parameters
    ----------
    novfn: ndarray(N)
        An audio novelty function
    beats: ndarray(M)
        An array of beat locations, in seconds
    sr: int
        Sample rate
    hop_length: int
        Hop length used in the STFT to construct novfn
    """
    h1 = np.min(novfn)
    h2 = np.max(novfn)
    diff = (h2-h1)
    h2 += 0.3*diff
    h1 -= 0.3*diff
    for b in beats:
        plt.plot([b, b], [h1, h2], linewidth=1)
    ts = np.arange(novfn.size)
    plt.plot(ts*hop_length/sr, novfn, c='C0', linewidth=2)
    plt.xlabel("Time (Sec)")


def sonify_beats(filename, sr, beats, blip_len=0.03):
    """
    Put short little 440hz blips at each beat location
    in an audio file

    Parameters
    ----------
    filename: string
        Path to audio file
    sr: int
        Sample rate of audio
    beats: ndarray(N)
        Beat locations, in seconds, of each beat
    blip_len: float 
        The length, in seconds, of each 440hz cosine blip
    """
    x, sr = librosa.load(filename, sr=sr)
    ## TODO: Fill this in and add the blips
    return x

def get_beats(novfn, sr, hop_length, tempo, alpha):
    """
    An implementation of dynamic programming beat tracking

    Parameters
    ----------
    novfn: ndarray(N)
        An audio novelty function
    sr: int
        Sample rate
    hop_length: int
        Hop length used in the STFT to construct novfn
    tempo: float
        The estimated tempo, in beats per minute
    alpha: float
        The penalty for tempo deviation
    
    Returns
    -------
    ndarray(B): 
        Beat locations, in seconds, of each beat
    """
    N = len(novfn)
    cscore = np.array(novfn) # Dynamic programming array
    backlink = np.ones(N, dtype=int) # Links for backtracing
    T = int((60*sr/hop_length)/tempo) # Period, in units of novfn samples per beat, of the tempo
    beats = []

    ## TODO: Fill this in

    return beats

def get_beats_fromaudio(x, sr):
    """
    Compute beats from raw audio using whatever technique you would like.
    You will have to compute an audio novelty function at least.  You'll
    probably also want to estimate tempo to use as a prior if you're going
    to use dynamic programming beat tracking

    Parameters
    ----------
    x: ndarray(N)
        The audio samples
    sr: int
        The sample rate

    Returns
    -------
    ndarray(B): 
        Beat locations, in seconds, of each beat
    """
    ## TODO: Fill this in
    pass
    
