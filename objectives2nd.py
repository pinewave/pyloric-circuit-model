from __future__ import division
import numpy as np

def getPeakDiffErr(peaksPD, peaksLP, peaksPY):
    peaksPD_std = np.mean(peaksPD)
    peaksLP_std = np.mean(peaksLP)
    peaksPY_std = np.mean(peaksPY)
    peaks_std = (peaksPD_std + peaksLP_std + peaksPY_std) / 3

    if peaks_std < 1.5:
        peaks_err = 0
    else:
        peaks_err = peaks_std

    peaks_err = peaks_std ** 2 if np.isnan(peaks_err).any() else peaks_err

    return peaks_err

def errFunc(featuresOf):
    # features
    featuresOfpd = featuresOf[0]
    featuresOflp = featuresOf[1]
    featuresOfpy = featuresOf[2]
    peaksPD = featuresOfpd[2]
    peaksLP = featuresOflp[2]
    peaksPY = featuresOfpy[2]
    ridgePD = featuresOfpd[8]
    ridgeLP = featuresOflp[8]
    ridgePY = featuresOfpy[8]

    # calling objective functions
    peaksErr = getPeakDiffErr(peaksPD, peaksLP, peaksPY)
    ridgeErr = (ridgePD + ridgeLP + ridgePY) / 3

    objectives2nd = [peaksErr, ridgeErr]

    return objectives2nd