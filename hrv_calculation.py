import sys
sys.path.append("/home/ec2-user/SageMaker/")
from utils.s3 import S3
from utils.fiducial_marker.hrv_functions.episodebasedhrv import HRVEpisodeExtract
from utils.fiducial_marker.hrv_functions import basefunctions as bf
from dateutil.relativedelta import relativedelta
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

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
fd_cols = ['P_VLF (s^2/Hz)', 'P_LF (s^2/Hz)', 'P_HF (s^2/Hz)', 'P_VLF (%)', 'P_LF (%)', 
           'P_HF (%)', 'pf_VLF (Hz)', 'pf_LF (Hz)', 'pf_HF (Hz)', 'LF/HF']
td_cols = ['SDNN (ms)', 'SDANN (ms)', 'MeanRR (ms)', 'RMSSD (ms)', 'pNN50 (%)']
nl_cols = ['REC (%)', 'DET (%)', 'LAM (%)', 'Lmean (bts)', 'Lmax (bts)', 
           'Vmean (bts)', 'Vmax (bts)', 'SD1', 'SD2', 'alpha1', 'alpha2']


class hrvcalculation:
    """
    HRV Library
    """
    
    def __init__(self):
        self.s3 = S3()

    @staticmethod
    def calctimedomainhrv(RRI, t_unit='ms', decim: int = 2):
        """
        This function estimates HRV features in the time-domain from a given RRI series.
        :param rri: RR interval series.
        :param t_unit: Time unit of input RRI series. Default is milliseconds (ms). Note this does not
                       impact the output units. Output units are fixed.
        :param int decim: Optional. The number of decimal places in results.
        :return: SDNN, SDANN, Mean RRI, RMSSD, pNN50
        """
        
        rri = RRI[~np.isnan(RRI)]
        if t_unit == 's':
            rri *= 1e3
            
        R_peaks = bf.rpeaks_from_rri(rri)

        # Calculating SDNN
        MeanRR = np.round(np.mean(rri), decim)
        SDNN = np.round(np.std(rri), decim)
        #
        # Calculating SDANN
        timejump = 300*1e3  # 5 minutes
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

        # Calculating pNN50                      pNN50 = (NN50 count) / (total NN count)
        NN_50 = np.diff(rri)
        pNN50 = np.round((len(NN_50[NN_50 > 50]) / len(R_peaks) * 100), decim)

        # Calculating RMSSD
        try:
            RMSSD = np.round((np.sqrt(np.sum(np.power(rri, 2)) / (len(R_peaks) - 1))), decim)
        except:
            RMSSD = 'nan'

        return pd.DataFrame([[SDNN, SDANN, MeanRR, RMSSD, pNN50]], columns=td_cols)

        # Freq_Analysis

    @staticmethod
    def calcfreqdomainhrv(RRI, t_unit='ms', meth=1, decim=3, M=5, O=50, BTval=10, omega_max=500, order=100):
        """
        This function estimates HRV features in the frequency-domain from a given RRI series.
        :param rri: RR interval series.
        :param t_unit: Time unit of input RRI series. Default is milliseconds (ms). Note this does not
                       impact the output units. Output units are fixed.
        :param meth: PSD estimation method. 1 = Welch's Method, 2 = Blackman-Tukey's method,
                     3 = Lombscargle's method. 4 = LPC method.
        :param decim: Optional. The number of decimal places in results.
        :param M: Welch Method parameter - segement length (% signal length).
        :param O: Welch Method parameter - overlap length (% segment length).
        :param BTval: Blackman-Tukey's parameter - computes biased autocorrelations up to lag value, BTval.
        :param omega_max: Lombscargle's method - omega.
        :param order: LPC method - order of magnitude.
        :return: PowerVLF, PowerLF, PowerHF, PowerVLF (%), PowerLF (%), PowerHF (%),
                 peakfrequencyVLF, peakfrequencyLF, peakfrequencyHF, LF/HF ratio
        """
        rri = RRI[~np.isnan(RRI)].astype(float)
        
        if t_unit == 'ms':
            rri = rri.astype(float) / 1e3
        
        R_peaks = bf.rpeaks_from_rri(rri)

        # Resample x at even intervals
        FS = 100
        t = np.cumsum(rri) / FS  # Use cumulative sum of RR intervals as time values
        
        #sorted_indices = np.argsort(t)
        #t = t[sorted_indices]
        #rri = rri[sorted_indices]
    
        cs = CubicSpline(t, rri)
        x_sampled = np.arange(0, np.round(t[-1]), 1 / FS)
        rri_sampled = cs(x_sampled)
        N = len(rri_sampled)
        xt = rri_sampled - np.mean(rri_sampled)

        L = bf.calc_zero_padding(N)
        f = np.arange(L) / L * FS
        centre = int(L / 2 + 1)
        f = f[0:centre]
        XX = np.concatenate((xt, np.zeros(L - N)))

        if meth == 1:
            # Welch's method (M = segement length, O = overlap)
            P = bf.welchPSD(XX, L, M, O)
            P_2 = P[1:centre + 1] / FS

        elif meth == 2:
            # Blackman-Tukey's method
            K = int(L / BTval)
            P = bf.blackmanTukeyPSD(XX, L, K) / 1e3
            P_2 = P[0:centre] / FS

        elif meth == 3:
            # Lombscargle's method
            rri = rri - np.mean(rri)
            omega = np.linspace(0.0001, np.pi * 2, omega_max)
            P_2 = lombscargle(R_peaks, rri, omega, normalize=True)
            f = omega / (2 * np.pi)

        else:
            #        rri = rri - np.mean(rri)
            psd = bf.lpcPSD(XX, order, L)  # psd is double-sided power spectra
            P_2 = psd[0:centre]

        # Calculate parameters
        # Power in VLF, LF, & HF frequency ranges
        VLF_upperlim = len(f[f < 0.04])
        LF_upperlim = len(f[f < 0.15])
        HF_upperlim = len(f[f < 0.4])
        powVLF = np.around(np.sum(P_2[0:VLF_upperlim]) * 1e3, decim)  # Convert to milliseconds
        powLF = np.around(np.sum(P_2[VLF_upperlim:LF_upperlim]) * 1e3, decim)  # Convert to milliseconds
        powHF = np.around(np.sum(P_2[LF_upperlim:HF_upperlim]) * 1e3, decim)  # Convert to milliseconds
        perpowVLF = np.around(powVLF / (powVLF + powLF + powHF) * 100, decim)
        perpowLF = np.around(powLF / (powVLF + powLF + powHF) * 100, decim)
        perpowHF = np.around(powHF / (powVLF + powLF + powHF) * 100, decim)

        # Peak Frequencies
        try:
            posVLF = np.argmax(P_2[0:VLF_upperlim])
            posLF = np.argmax(P_2[VLF_upperlim:LF_upperlim])
            posHF = np.argmax(P_2[LF_upperlim:HF_upperlim])
            peak_freq_VLF = np.around(f[posVLF], decim)
            peak_freq_LF = np.around(f[posLF + VLF_upperlim], decim)
            peak_freq_HF = np.around(f[posHF + LF_upperlim], decim)
        except:
            posVLF = np.nan
            posLF = np.nan
            posHF = np.nan
            peak_freq_VLF = np.nan
            peak_freq_LF = np.nan
            peak_freq_HF = np.nan

        LFHF = np.around(np.true_divide(powLF, powHF), decim)

        return pd.DataFrame([[powVLF, powLF, powHF, perpowVLF, perpowLF, perpowHF,
                              peak_freq_VLF, peak_freq_LF, peak_freq_HF, LFHF]], columns=fd_cols)

    
    def calcnonlinearhrv(self, RRI, t_unit='ms', m=10, L=1, min_box=4, max_box=64, inc=1, cop=12, decim=2):
        """
        This function estimates nonlinear HRV features for a given RRI series.
        :param rri: RR interval series.
        :param t_unit: Time unit of input RRI series. Default is milliseconds (ms). Note this does not
                       impact the output units. Output units are fixed.
        :param m: RQA (Recurrence Quantification Analysis) parameter - embedding dimension.
        :param L: RQA parameter - lag.
        :param min_box: DFA parameter - Start value for short term fluctuations.
        :param max_box: DFA parameter - End value for long term fluctuations.
        :param inc: DFA (Detrended Fluctuation Analysis) parameter - Step size.
        :param cop: DFA parameter - cross over point between short and long term fluctuations.
        :param decim: Optional. The number of decimal places in results.
        :returns: RQA parameters, Poincare Deviations, DFA trends.
        """
        
        rri = RRI[~np.isnan(RRI)].astype(float)
        
        if t_unit == 'ms':
            rri = rri.astype(float) / 1e3
        
        REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = self.RQA(rri, m, L, decim)
        sd1, sd2, _, _, _, _ = self.Poincare(rri, decim)
        alpha1, alpha2, _ = self.DFA(rri, min_box, max_box, inc, cop, decim)

        return pd.DataFrame([[REC, DET, LAM, Lmean, Lmax, Vmean, Vmax,
                              sd1, sd2, alpha1, alpha2]], columns=nl_cols)


    # RECURRENCE PLOT
    @staticmethod
    def RQA_plot(Matrix, N, Fig: Figure = None):
        """

        :param Matrix:
        :param N:
        :param Fig:
        :return:
        """
        xplot = np.zeros((N, N))
        yplot = np.zeros((N, N))

        RQA_plt = Fig.add_subplot(111)
        RQA_plt.clear()
        for i in range(0, len(Matrix)):
            yplot[:, i] = np.arange(1, len(Matrix) + 1) * Matrix[:, i]
            xplot[:, i] = np.ones(len(Matrix)) * (i + 1)
        RQA_plt.scatter(xplot, yplot, c='k', s=0.5)
        RQA_plt.set_title('Recurrence Plot')
        RQA_plt.set_xlabel('Heart beat (sample number)')
        RQA_plt.set_ylabel('Heart beat (sample number)')

        return Fig

    @staticmethod
    def RQA(RRI, m=10, L=1, decim=2):
        """

        :param RRI:
        :param m:
        :param L:
        :param decim:
        :return:
        """

        Matrix, N = bf.RQA_matrix(RRI=RRI, m=m, L=L)

        # Analyse Diagonals of RP
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
                V_Matrix[i, j] = 0  # Zeros out half of the matrix

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

        # %Calculate Features
        REC = np.round(np.sum(Matrix) / np.power(N, 2) * 100, decim)
        diagsums = diagsums[2:N, :]
        DET = np.round(np.sum(diagsums) / (np.sum(FlVec) / 2) * 100, decim)
        nzdiag = np.sum(diagsums > 0)  # Number of non-zero diagonals
        Lmean = np.round(np.sum(diagsums) / nzdiag, decim)
        try:
            Lmax = np.max(diagsums)
        except:
            Lmax = 0
        LAM = np.round(np.sum(vertsums) / np.sum(V_Matrix) * 100, decim)
        nzvert = np.sum(vertsums > 0)  # Number of non-zero verticals
        Vmean = np.round(np.sum(vertsums) / nzvert, decim)
        try:
            Vmax = np.max(vertsums)
        except:
            Vmax = 0

        return REC, DET, LAM, Lmean, int(Lmax), Vmean, int(Vmax)

    @staticmethod
    def Poincare_Plot(sd1, sd2, c1, c2, x, y, Fig: Figure = None):
        """

        :param sd1:
        :param sd2:
        :param c1:
        :param c2:
        :param x:
        :param y:
        :param Fig:
        """
        A = sd2 * np.cos(np.pi / 4)
        B = sd1 * np.sin(np.pi / 4)

        ellipse = patch.Ellipse((c1, c2), sd2 * 2, sd1 * 2, 45, facecolor="none", edgecolor="b", linewidth=2, zorder=5)
        poin_plt = Fig.add_subplot(111)
        poin_plt.clear()
        if poin_plt.axes.axes.yaxis_inverted() == 1:
            poin_plt.axes.axes.invert_yaxis()
        poin_plt.plot(x, y, 'ko', markersize=3, zorder=0)  # ,'MarkerFaceColor', 'k', 'MarkerSize',4)
        poin_plt.add_patch(ellipse)
        poin_plt.set_title('Poincare Plot')
        poin_plt.set_xlabel('RRI$_{n}$ (s)')
        poin_plt.set_ylabel('RRI$_{n+1}$ (s)')
        poin_plt.plot([c1, c1 + A], [c2, c2 + A], 'm', label="SD1", zorder=10)
        poin_plt.plot([c1 - 4 * A, c1 + 4 * A], [c2 - 4 * A, c2 + 4 * A], 'b', dashes=[6, 2])
        poin_plt.plot([c1, c1 - B], [c2, c2 + B], 'c', label="SD2", zorder=10)
        poin_plt.plot([c1 + B * 4, c1 - B * 4], [c2 - 4 * B, c2 + 4 * B], 'b', dashes=[4, 2, 10, 2])
        poin_plt.legend()

    @staticmethod
    def Poincare(RRI, decim=3):
        """
        :param RRI: RR interval series
        :param decim:
        :return:
        """
        lenx = np.size(RRI)
        RRI = np.reshape(RRI, [lenx, ])
        x = RRI[0:lenx - 1]
        y = RRI[1:lenx]
        c1 = np.mean(x)
        c2 = np.mean(y)

        sd1_sqed = 0.5 * np.power(np.std(np.diff(x)), 2)
        sd1 = np.sqrt(sd1_sqed)

        sd2_sqed = 2 * np.power(np.std(x), 2) - sd1_sqed
        sd2 = np.sqrt(sd2_sqed)

        if decim is not None:
            return np.round(sd1 * 1e3, decim), np.round(sd2 * 1e3, decim), c1, c2, x, y
        else:
            return sd1, sd2, c1, c2, x, y

    @staticmethod
    def DFA(RRI, min_box=4, max_box=64, inc=1, cop=12, decim=3):
        """

        :param RRI:
        :param min_box: minimum point
        :param max_box: max point
        :param inc: increment/step size
        :param cop: cross-over point for SD1 and SD2 or up and lower division
        :param decim:
        :return:
        """

        NN = np.size(RRI)
        RRI = np.reshape(RRI, [NN, ])
        box_lengths = np.arange(min_box, max_box + 1, inc)  # Box length
        y = np.zeros(NN)
        mm = np.mean(RRI)
        y[0] = RRI[0] - mm
        for k in range(1, NN):
            y[k] = y[k - 1] + RRI[k] - mm

        M = len(box_lengths)

        F = np.zeros(M)
        for q in range(M):
            n = box_lengths[q]
            N = int(np.floor(len(y) / n))
            y_n2 = np.zeros((n, N))
            y2 = np.reshape(y[0:N * n], [n, N],
                            order='F')  # Order 'F' fills column by column, whereas order 'C' fills row by row
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
        x_alp1 = box_lengths[box_lengths <= cop]
        F_alp1 = F[0:len(x_alp1)]
        x_vals1 = np.log10(x_alp1)
        y_vals1 = np.log10(F_alp1)
        P1 = np.polyfit(x_vals1, y_vals1, 1)

        # Long-term DFA - alpha 2
        x_alp2 = box_lengths[box_lengths >= (cop + 1)]
        x_vals2 = np.log10(x_alp2)
        F_alp2 = F[len(x_alp1):len(F)]
        y_vals2 = np.log10(F_alp2)
        P_2 = np.polyfit(x_vals2, y_vals2, 1)

        alp1 = np.round(P1[0], decim)
        alp2 = np.round(P_2[0], decim)
        F = np.round(F, decim)

        return alp1, alp2, F

    @staticmethod
    def DFA_fig(RRI, min_box=4, max_box=64, inc=1, cop=12, decim=3):
        """

        :param RRI:
        :param min_box:
        :param max_box:
        :param inc:
        :param cop:
        :param decim:
        :return:
        """
        NN = np.size(RRI)
        RRI = np.reshape(RRI, [NN, ])
        box_lengths = np.arange(min_box, max_box + 1, inc)  # Box length
        y = np.zeros(NN)
        mm = np.mean(RRI)
        y[0] = RRI[0] - mm
        for k in range(1, NN):
            y[k] = y[k - 1] + RRI[k] - mm

        M = len(box_lengths)

        F = np.zeros(M)
        for q in range(M):
            n = box_lengths[q]
            N = int(np.floor(len(y) / n))
            y_n2 = np.zeros((n, N))
            y2 = np.reshape(y[0:N * n], [n, N],
                            order='F')  # Order 'F' fills column by column, whereas order 'C' fills row by row
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
        x_alp1 = box_lengths[box_lengths <= cop]
        F_alp1 = F[0:len(x_alp1)]
        x_vals1 = np.log10(x_alp1)
        y_vals1 = np.log10(F_alp1)

        P1 = np.polyfit(x_vals1, y_vals1, 1)
        y_new1 = np.polyval(P1, x_vals1)

        # Long-term DFA - alpha 2
        x_alp2 = box_lengths[box_lengths >= (cop + 1)]
        x_vals2 = np.log10(x_alp2)
        F_alp2 = F[len(x_alp1):len(F)]
        y_vals2 = np.log10(F_alp2)
        P_2 = np.polyfit(x_vals2, y_vals2, 1)
        y_new2 = np.polyval(P_2, x_vals2)

        a1 = np.round(P1[0], decim)
        a2 = np.round(P_2[0], decim)

        return x_vals1, y_vals1, y_new1, x_vals2, y_vals2, y_new2, a1, a2
