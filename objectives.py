# Date: 101316
# Author: Kun Tian
# Changed the objectives to phase and period

from __future__ import division
import numpy as np

### parameters
# phase - control
PDoff_meanE = 0.2
PDoff_stdE = 0.02
PDoff_minE = PDoff_meanE - 2 * PDoff_stdE
PDoff_maxE = PDoff_meanE + 2 * PDoff_stdE

LPon_meanE = 0.38
LPon_stdE = 0.05
LPon_minE = LPon_meanE - 2 * LPon_stdE
LPon_maxE = LPon_meanE + 2 * LPon_stdE

LPoff_meanE = 0.7
LPoff_stdE = 0.05
LPoff_minE = LPoff_meanE - 2 * LPoff_stdE
LPoff_maxE = LPoff_meanE + 2 * LPoff_stdE

PYon_meanE = 0.68
PYon_stdE = 0.05
PYon_minE = PYon_meanE - 2 * PYon_stdE
PYon_maxE = PYon_meanE + 2 * PYon_stdE

PYoff_meanE = 0.99
PYoff_stdE = 0.015
PYoff_minE = PYoff_meanE - 2 * PYoff_stdE
PYoff_maxE = PYoff_meanE + 2 * PYoff_stdE

# period - control
period_meanE = 0.9
period_stdE = 0.17
period_minE = period_meanE - 2 * period_stdE
period_maxE = period_meanE + 2 * period_stdE

### feature-based error functions
# obj1: PD off
def getPDoffErr(PD_off):
    PDoff_mean = np.mean(PD_off)
    if PDoff_minE <= PDoff_mean <= PDoff_maxE:
        PDoff_Err = 0
    elif PDoff_mean != 0:
        PDoff_Err = ((PDoff_mean - PDoff_meanE) / PDoff_stdE) ** 2
    else:
        PDoff_Err = (PDoff_meanE / PDoff_stdE) ** 3

    PDoff_Err = (PDoff_meanE / PDoff_stdE) ** 3 if np.isnan(PDoff_Err).any() else PDoff_Err
    return PDoff_Err

# obj2: LP on
def getLPonErr(LP_on):
    LPon_mean = np.mean(LP_on)
    if LPon_minE <= LPon_mean <= LPon_maxE:
        LPon_Err = 0
    elif LPon_mean != 0:
        LPon_Err = ((LPon_mean - LPon_meanE) / LPon_stdE) ** 2
    else:
        LPon_Err = (LPon_meanE / LPon_stdE) ** 3

    LPon_Err = (LPon_meanE / LPon_stdE) ** 3 if np.isnan(LPon_Err).any() else LPon_Err
    return LPon_Err

# obj3: LP off
def getLPoffErr(LP_off):
    LPoff_mean = np.mean(LP_off)
    if LPoff_minE <= LPoff_mean <= LPoff_maxE:
        LPoff_Err = 0
    elif LPoff_mean != 0:
        LPoff_Err = ((LPoff_mean - LPoff_meanE) / LPoff_stdE) ** 2
    else:
        LPoff_Err = (LPoff_meanE / LPoff_stdE) ** 3

    LPoff_Err = (LPoff_meanE / LPoff_stdE) ** 3 if np.isnan(LPoff_Err).any() else LPoff_Err
    return LPoff_Err

# obj4: PY on
def getPYonErr(PY_on):
    PYon_mean = np.mean(PY_on)
    if PYon_minE <= PYon_mean <= PYon_maxE:
        PYon_Err = 0
    elif PYon_mean != 0:
        PYon_Err = ((PYon_mean - PYon_meanE) / PYon_stdE) ** 2
    else:
        PYon_Err = (PYon_meanE / PYon_stdE) ** 3

    PYon_Err = (PYon_meanE / PYon_stdE) ** 3 if np.isnan(PYon_Err).any() else PYon_Err
    return PYon_Err

# obj5: PY off
def getPYoffErr(PY_off):
    PYoff_mean = np.mean(PY_off)
    if PYoff_minE <= PYoff_mean <= PYoff_maxE:
        PYoff_Err = 0
    elif PYoff_mean != 0:
        PYoff_Err = ((PYoff_mean - PYoff_meanE) / PYoff_stdE) ** 2
    else:
        PYoff_Err = (PYoff_meanE / PYoff_stdE) ** 3

    PYoff_Err = (PYoff_meanE / PYoff_stdE) ** 3 if np.isnan(PYoff_Err).any() else PYoff_Err
    return PYoff_Err

# obj6: period (range, order, and regularity)
def getPeriodErr(period, rightOrder):
    period_mean = np.mean(period)
    if period_minE <= period_mean <= period_maxE and rightOrder is True:
        period_err = 0
    elif period_mean != 0:
        period_err = ((period_mean - period_meanE) / period_stdE) ** 2
    else:
        period_err = (period_meanE / period_stdE) ** 3

    period_err = (period_meanE / period_stdE) ** 3 if np.isnan(period_err).any() else period_err

    return period_err

def errFunc(featuresOf):
    # features
    circuit_features = featuresOf[5]
    period = circuit_features[0]
    PD_off = circuit_features[1]
    LP_on = circuit_features[2]
    LP_off = circuit_features[3]
    PY_on = circuit_features[4]
    PY_off = circuit_features[5]

    rightOrder = featuresOf[6]

    # calling objective functions
    PDoffErr = getPDoffErr(PD_off)
    LPonErr = getLPonErr(LP_on)
    LPoffErr = getLPoffErr(LP_off)
    PYonErr = getPYonErr(PY_on)
    PYoffErr = getPYoffErr(PY_off)
    periodErr = getPeriodErr(period, rightOrder)

    objectives = [PDoffErr, LPonErr, LPoffErr, PYonErr, PYoffErr, periodErr]

    return objectives
