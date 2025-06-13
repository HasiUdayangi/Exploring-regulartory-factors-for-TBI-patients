
import sys
from numpy import ndarray
from matplotlib.figure import Figure
from scipy.interpolate import CubicSpline
from scipy.signal import lombscargle
from typing import List
import matplotlib.patches as patch
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


fd_cols = ['P_VLF (s^2/Hz)', 'P_LF (s^2/Hz)', 'P_HF (s^2/Hz)', 'P_VLF (%)', 'P_LF (%)', 
           'P_HF (%)', 'pf_VLF (Hz)', 'pf_LF (Hz)', 'pf_HF (Hz)', 'LF/HF']
td_cols = ['SDNN (ms)', 'SDANN (ms)', 'MeanRR (ms)', 'RMSSD (ms)', 'pNN50 (%)']
nl_cols = ['REC (%)', 'DET (%)', 'LAM (%)', 'Lmean (bts)', 'Lmax (bts)', 
           'Vmean (bts)', 'Vmax (bts)', 'SD1', 'SD2', 'alpha1', 'alpha2']



   
