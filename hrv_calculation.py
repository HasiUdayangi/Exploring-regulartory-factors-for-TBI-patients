
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


class hrvlib:


    def calctimedomainhrv(RRI, t_unit='ms', decim: int = 2):

        rri = RRI[~np.isnan(RRI)]
        if t_unit == 's':
            rri *= 1e3
            
        R_peaks = bf.rpeaks_from_rri(rri)

        # Calculating SDNN
        MeanRR = np.round(np.mean(rri), decim)
        SDNN = np.round(np.std(rri), decim)
        
        # Calculating SDANN
        timejump = 300*1e3
        timestamp = timejump
        runs = int(np.ceil(R_peaks[-1] / timejump))
        if runs > 12:
            runs = 12
        SDNN_5 = np.zeros(runs)
        i = 0
        while timestamp <= timejump * runs:
            section = R_peaks[R_peaks <= timestamp]
            section = section[section > (timestamp - timejump)]
            timestamp += timejump
            R_td_5 = np.diff(section)
            SDNN_5[i] = np.std(R_td_5)
            i += 1
        SDNN_5 = SDNN_5[~np.isnan(SDNN_5)]
        SDANN = np.round(np.mean(SDNN_5), decim)

        # Calculating pNN50               
        NN_50 = np.diff(rri)
        pNN50 = np.round((len(NN_50[NN_50 > 50]) / len(R_peaks) * 100), decim)

        # Calculating RMSSD
        try:
            RMSSD = np.round((np.sqrt(np.sum(np.power(rri, 2)) / (len(R_peaks) - 1))), decim)
        except:
            RMSSD = 'nan'

        return pd.DataFrame([[SDNN, SDANN, MeanRR, RMSSD, pNN50]], columns=td_cols)

    # Freq_Analysis

    def calcfreqdomainhrv(RRI, t_unit='ms', meth=1, decim=3, M=5, O=50, BTval=10, omega_max=500, order=100):

        rri = RRI[~np.isnan(RRI)]
        
        if t_unit == 'ms':
            rri /= 1e3
        
        R_peaks = bf.rpeaks_from_rri(rri)

        FS = 240
        cs = CubicSpline(R_peaks, rri)
        x_sampled = np.arange(0, np.round(R_peaks[-1]), 1 / FS)
        rri_sampled = cs(x_sampled)
        N = len(rri_sampled)
        xt = rri_sampled - np.mean(rri_sampled)

        L = bf.calc_zero_padding(N)
        f = np.arange(L) / L * FS
        centre = int(L / 2 + 1)
        f = f[0:centre]
        XX = np.concatenate((xt, np.zeros(L - N)))

        if meth == 1:
            # Welch's method
            P = bf.welchPSD(XX, L, M, O)
            P_2 = P[1:centre + 1] / FS

        else:
            psd = bf.lpcPSD(XX, order, L)
            P_2 = psd[0:centre]

        VLF_upperlim = len(f[f < 0.04])
        LF_upperlim = len(f[f < 0.15])
        HF_upperlim = len(f[f < 0.4])
        powVLF = np.around(np.sum(P_2[0:VLF_upperlim]) * 1e3, decim) 
        powLF = np.around(np.sum(P_2[VLF_upperlim:LF_upperlim]) * 1e3, decim)
        powHF = np.around(np.sum(P_2[LF_upperlim:HF_upperlim]) * 1e3, decim)
        perpowVLF = np.around(powVLF / (powVLF + powLF + powHF) * 100, decim)
        perpowLF = np.around(powLF / (powVLF + powLF + powHF) * 100, decim)
        perpowHF = np.around(powHF / (powVLF + powLF + powHF) * 100, decim)
        
        posVLF = np.argmax(P_2[0:VLF_upperlim])
        posLF = np.argmax(P_2[VLF_upperlim:LF_upperlim])
        posHF = np.argmax(P_2[LF_upperlim:HF_upperlim])
        peak_freq_VLF = np.around(f[posVLF], decim)
        peak_freq_LF = np.around(f[posLF + VLF_upperlim], decim)
        peak_freq_HF = np.around(f[posHF + LF_upperlim], decim)
        LFHF = np.around(np.true_divide(powLF, powHF), decim)

        return pd.DataFrame([[powVLF, powLF, powHF, perpowVLF, perpowLF, perpowHF,
                              peak_freq_VLF, peak_freq_LF, peak_freq_HF, LFHF]], columns=fd_cols)

    def calcnonlinearhrv(self, RRI, t_unit='ms', m=10, L=1, min_box=4, max_box=64, inc=1, cop=12, decim=2):
        
        rri = RRI[~np.isnan(RRI)]
        
        if t_unit == 'ms':
            rri /= 1e3
        
        REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = self.RQA(rri, m, L, decim)
        sd1, sd2, _, _, _, _ = self.Poincare(rri, decim)
        alpha1, alpha2, _ = self.DFA(rri, min_box, max_box, inc, cop, decim)

        return pd.DataFrame([[REC, DET, LAM, Lmean, Lmax, Vmean, Vmax,
                              sd1, sd2, alpha1, alpha2]], columns=nl_cols)
    

    def RQA(RRI, m=10, L=1, decim=2):

        Matrix, N = bf.RQA_matrix(RRI=RRI, m=m, L=L)

        FlVec = np.copy(Matrix)
        diagsums = np.zeros((N, N))
        for i in range(N):
            vert = np.diag(FlVec, k=i)
            init = 0
            dsums = 0
            for j in range(len(vert)):
                if vert[j] == 1:
                    init = init + 1
                    if j == len(vert) & (init > 1):
                        diagsums[i, dsums] = init
                else:
                    if init > 1:
                        diagsums[i, dsums] = init
                        dsums = dsums + 1
                        init = 0
                    else:
                        init = 0

        V_Matrix = np.copy(Matrix)
        for i in range(N):
            for j in range(i, N):
                V_Matrix[i, j] = 0

        vertsums = np.zeros((N, N))
        for i in range(N):
            vert = V_Matrix[:, i]
            init = 0
            vsums = 1
            for j in range(len(vert)):
                if vert[j] == 1:
                    init = init + 1
                    if (j == len(vert)) & (init > 1):
                        vertsums[i + 1, vsums] = init
                else:
                    if init > 1:
                        vertsums[i + 1, vsums] = init
                        vsums = vsums + 1
                        init = 0
                    else:
                        init = 0

        REC = np.round(np.sum(Matrix) / np.power(N, 2) * 100, decim)
        diagsums = diagsums[2:N, :]
        DET = np.round(np.sum(diagsums) / (np.sum(FlVec) / 2) * 100, decim)
        nzdiag = np.sum(diagsums > 0)
        Lmean = np.round(np.sum(diagsums) / nzdiag, decim)
        try:
            Lmax = np.max(diagsums)
        except:
            Lmax = 0
        LAM = np.round(np.sum(vertsums) / np.sum(V_Matrix) * 100, decim)
        nzvert = np.sum(vertsums > 0) 
        Vmean = np.round(np.sum(vertsums) / nzvert, decim)
        try:
            Vmax = np.max(vertsums)
        except:
            Vmax = 0

        return REC, DET, LAM, Lmean, int(Lmax), Vmean, int(Vmax)


    def Poincare(RRI, decim=3):

        len = np.size(RRI)
        RRI = np.reshape(RRI, [len, ])
        x = RRI[0:len - 1]
        y = RRI[1:len]
        c1 = np.mean(x)
        c2 = np.mean(y)

        sd1_s = 0.5 * np.power(np.std(np.diff(x)), 2)
        sd1 = np.sqrt(sd1_s)

        sd2_s = 2 * np.power(np.std(x), 2) - sd1_s
        sd2 = np.sqrt(sd2_s)

        if decim is not None:
            return np.round(sd1 * 1e3, decim), np.round(sd2 * 1e3, decim), c1, c2, x, y
        else:
            return sd1, sd2, c1, c2, x, y


    def DFA(RRI, min_box=4, max_box=64, inc=1, cop=12, decim=3):

        NN = np.size(RRI)
        RRI = np.reshape(RRI, [NN, ])
        box_len = np.arange(min_box, max_box + 1, inc)
        y = np.zeros(NN)
        mm = np.mean(RRI)
        y[0] = RRI[0] - mm
        for k in range(1, NN):
            y[k] = y[k - 1] + RRI[k] - mm

        M = len(box_len)

        F = np.zeros(M)
        for q in range(M):
            n = box_len[q]
            N = int(np.floor(len(y) / n))
            y_n2 = np.zeros((n, N))
            y2 = np.reshape(y[0:N * n], [n, N],
                            order='F')
            k = np.reshape(np.arange(N * n), [n, N])
            for m in range(N):
                P = np.polyfit(k[:, m], y2[:, m], 1)
                y_n2[:, m] = np.polyval(P, k[:, m])
            if NN > N * n:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=np.RankWarning)
                    y3 = y[N * n:len(y)]
                    k = np.arange(N * n, NN)
                    P = np.polyfit(k, y3, 1)
                    y_n3 = np.polyval(P, k)
                    y_n = np.append(y_n2.flatten('F'), y_n3.flatten('F'))
            else:
                y_n = y_n2.flatten('F')

            F[q] = np.sqrt(np.sum(np.power((y.flatten('F') - y_n.flatten('F')), 2)) / NN)
        # Short-term DFA - alpha 1
        x_alp1 = box_len[box_len <= cop]
        F_alp1 = F[0:len(x_alp1)]
        x_vals1 = np.log10(x_alp1)
        y_vals1 = np.log10(F_alp1)
        P1 = np.polyfit(x_vals1, y_vals1, 1)

        # Long-term DFA - alpha 2
        x_alp2 = box_len[box_len >= (cop + 1)]
        x_vals2 = np.log10(x_alp2)
        F_alp2 = F[len(x_alp1):len(F)]
        y_vals2 = np.log10(F_alp2)
        P_2 = np.polyfit(x_vals2, y_vals2, 1)

        alp1 = np.round(P1[0], decim)
        alp2 = np.round(P_2[0], decim)
        F = np.round(F, decim)

        return alp1, alp2, F

   