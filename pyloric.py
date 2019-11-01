# 02/15/2016
# Author: Kun Tian
# Pyloric circuit model

from __future__ import division
from numpy import exp
from numpy import log
from numpy import power


def step(y, t, params):

    # h
    h = params[0]

    # conductances
    conductances = params[1]
    GmaxNaPD = conductances[0]             # mS/cm^2
    GmaxCaTPD = conductances[1] * 1e-2
    GmaxCaSPD = conductances[2] * 1e-2
    GmaxAPD = conductances[3]
    GmaxKCaPD = conductances[4] * 0.1
    GmaxKPD = conductances[5]
    GmaxhPD = conductances[6] * 1e-4
    GmaxLPD = conductances[7] * 1e-4
    GmaxNaLP = conductances[8]
    GmaxCaTLP = conductances[9] * 1e-2
    GmaxCaSLP = conductances[10] * 0.1
    GmaxALP = conductances[11] * 0.1
    GmaxKCaLP = conductances[12] * 0.1
    GmaxKLP = conductances[13]
    GmaxhLP = conductances[14] * 1e-3
    GmaxLLP = conductances[15] * 1e-4
    GmaxNaPY = conductances[16]
    GmaxCaTPY = conductances[17] * 1e-2
    GmaxCaSPY = conductances[18] * 1e-2
    GmaxAPY = conductances[19]
    GmaxKCaPY = conductances[20] * 0.1
    GmaxKPY = conductances[21]
    GmaxhPY = conductances[22] * 1e-4
    GmaxLPY = conductances[23] * 1e-4
    gmaxPDLP_g = conductances[24] * 1e-6       # nS
    gmaxPDLP_a = conductances[25] * 1e-6
    gmaxPDPY_g = conductances[26] * 1e-6
    gmaxPDPY_a = conductances[27] * 1e-6
    gmaxLPPD_g = conductances[28] * 1e-6
    gmaxLPPY_g = conductances[29] * 1e-6
    gmaxPYLP_g = conductances[30] * 1e-6
    GmaxProcPD_0 = conductances[31] * 1e-3
    GmaxProcLP_0 = conductances[32] * 1e-3
    GmaxProcPY_0 = conductances[33] * 1e-3

    # constants
    constants = params[2]
    Cs = constants[0]
    A = constants[1]
    EL = constants[2]
    EK = constants[3]
    Eh = constants[4]
    ENa = constants[5]
    EProc = constants[6]
    decent = constants[7]

    tau_Ca = constants[8]
    f = constants[9]
    Ca_init = constants[10]
    Ca_out = constants[11]

    # R = constants[12]
    # Farad = constants[13]
    # Temp = constants[14]
    # Z_Ca = constants[15]
    Nernstfactor = constants[16]

    EGlu = constants[17]
    EACh = constants[18]
    k_glu = constants[19]
    k_ach = constants[20]
    Vth = constants[21]          # half-activation voltage
    deltaV = constants[22]

    # parameters
    sPDLPglu = y[0]
    sPDLPach = y[1]
    sPDPYglu = y[2]
    sPDPYach = y[3]
    sLPPDglu = y[4]
    sLPPYglu = y[5]
    sPYLPglu = y[6]

    ConCaPD = y[7]
    VsPD = y[8]
    mKPD = y[9]
    mNaPD = y[10]
    hNaPD = y[11]
    mCaSPD = y[12]
    hCaSPD = y[13]
    mKCaPD = y[14]
    mhPD = y[15]
    mCaTPD = y[16]
    hCaTPD = y[17]
    mAPD = y[18]
    hAPD = y[19]

    ConCaLP = y[20]
    VsLP = y[21]
    mKLP = y[22]
    mNaLP = y[23]
    hNaLP = y[24]
    mCaSLP = y[25]
    hCaSLP = y[26]
    mKCaLP = y[27]
    mhLP = y[28]
    mCaTLP = y[29]
    hCaTLP = y[30]
    mALP = y[31]
    hALP = y[32]

    ConCaPY = y[33]
    VsPY = y[34]
    mKPY = y[35]
    mNaPY = y[36]
    hNaPY = y[37]
    mCaSPY = y[38]
    hCaSPY = y[39]
    mKCaPY = y[40]
    mhPY = y[41]
    mCaTPY = y[42]
    hCaTPY = y[43]
    mAPY = y[44]
    hAPY = y[45]

    ## calculate synaptic conductance
    # 1: PD; 2: LP; 3: PY; g: glutamatergic; a: cholinergic

    # PD to LP, glu
    smax12_g = 1 / (1 + exp((Vth - VsPD) / deltaV))
    tau12_g = (1 - smax12_g) * k_glu
    sPDLPglu_next = smax12_g
    if h < tau12_g:
        sPDLPglu_next = smax12_g + (sPDLPglu - smax12_g) * exp(- h / tau12_g)
    gPDLP_g = gmaxPDLP_g * sPDLPglu_next

    # PD to LP, ACh
    smax12_a = 1 / (1 + exp((Vth - VsPD) / deltaV))
    tau12_a = (1 - smax12_a) * k_ach
    sPDLPach_next = smax12_a
    if h < tau12_a:
        sPDLPach_next = smax12_a + (sPDLPach - smax12_a) * exp(- h / tau12_a)
    gPDLP_a = gmaxPDLP_a * sPDLPach_next

    # PD to PY, glu
    smax13_g = 1 / (1 + exp((Vth - VsPD) / deltaV))
    tau13_g = (1 - smax13_g) * k_glu
    sPDPYglu_next = smax13_g
    if h < tau13_g:
        sPDPYglu_next = smax13_g + (sPDPYglu - smax13_g) * exp(- h / tau13_g)
    gPDPY_g = gmaxPDPY_g * sPDPYglu_next

    # PD to PY, ACh
    smax13_a = 1 / (1 + exp((Vth - VsPD) / deltaV))
    tau13_a = (1 - smax13_a) * k_ach
    sPDPYach_next = smax13_a
    if h < tau13_a:
        sPDPYach_next = smax13_a + (sPDPYach - smax13_a) * exp(- h / tau13_a)
    gPDPY_a = gmaxPDPY_a * sPDPYach_next

    # LP to PD, glu
    smax21_g = 1 / (1 + exp((Vth - VsLP) / deltaV))
    tau21_g = (1 - smax21_g) * k_glu
    sLPPDglu_next = smax21_g
    if h < tau21_g:
        sLPPDglu_next = smax21_g + (sLPPDglu - smax21_g) * exp(- h / tau21_g)
    gLPPD_g = gmaxLPPD_g * sLPPDglu_next

    # LP to PY, glu
    smax23_g = 1 / (1 + exp((Vth - VsLP) / deltaV))
    tau23_g = (1 - smax23_g) * k_glu
    sLPPYglu_next = smax23_g
    if h < tau23_g:
        sLPPYglu_next = smax23_g + (sLPPYglu - smax23_g) * exp(- h / tau23_g)
    gLPPY_g = gmaxLPPY_g * sLPPYglu_next

    # PY to LP, glu
    smax32_g = 1 / (1 + exp((Vth - VsPY) / deltaV))
    tau32_g = (1 - smax32_g) * k_glu
    sPYLPglu_next = smax32_g
    if h < tau32_g:
        sPYLPglu_next = smax32_g + (sPYLPglu - smax32_g) * exp(- h / tau32_g)
    gPYLP_g = gmaxPYLP_g * sPYLPglu_next


    ## calculate PD neuron conductance
    # leak current
    GLPD = GmaxLPD * A

    # CaS current
    ECaPD = Nernstfactor * log(Ca_out / ConCaPD)

    mmaxCaSPD = 1 / (1 + exp(-(VsPD + 33.0) / 8.1))
    hmaxCaSPD = 1 / (1 + exp((VsPD + 60.0) / 6.2))
    tau_mCaSPD = (2.8 + 14.0 / (exp((VsPD + 27.0) / 10.0) + exp(-(VsPD + 70.0) / 13.0))) * 1000.0
    tau_hCaSPD = (120.0 + 300.0 / (exp((VsPD + 55) / 9) + exp(-(VsPD + 65) / 16))) * 1000

    mCaSPD_next = mmaxCaSPD
    if h < tau_mCaSPD:
        mCaSPD_next = mmaxCaSPD + (mCaSPD - mmaxCaSPD) * exp(- h / tau_mCaSPD)

    hCaSPD_next = hmaxCaSPD
    if h < tau_hCaSPD:
        hCaSPD_next = hmaxCaSPD + (hCaSPD - hmaxCaSPD) * exp(- h / tau_hCaSPD)

    GCaSPD = GmaxCaSPD * power(mCaSPD_next, 3) * hCaSPD_next * A
    ICaSPD = GCaSPD * (VsPD - ECaPD)

    # Na current
    mmaxNaPD = 1 / (1 + exp(-(VsPD + 25.5) / 5.29))
    hmaxNaPD = 1 / (1 + exp((VsPD + 48.9) / 5.18))
    tau_mNaPD = (2.64 - 2.52 / (1 + exp(-(VsPD + 120) / 25))) * 1000.0
    tau_hNaPD = (1.34 / (1 + exp(-(VsPD + 62.9) / 10)) * (1.5 + 1 / (1 + exp((VsPD + 34.9) / 3.6)))) * 1000

    mNaPD_next = mmaxNaPD
    if h < tau_mNaPD:
        mNaPD_next = mmaxNaPD + (mNaPD - mmaxNaPD) * exp(- h / tau_mNaPD)

    hNaPD_next = hmaxNaPD
    if h < tau_hNaPD:
        hNaPD_next = hmaxNaPD + (hNaPD - hmaxNaPD) * exp(- h / tau_hNaPD)

    GNaPD = GmaxNaPD * power(mNaPD_next, 3) * hNaPD_next * A

    # K current
    mmaxKPD = 1 / (1 + exp(-(12.3 + VsPD) / 11.8))
    tau_mKPD = (14.4 - 12.8 / (1 + exp(-(VsPD + 28.3) / 19.2))) * 1000

    mKPD_next = mmaxKPD
    if h < tau_mKPD:
        mKPD_next = mmaxKPD + (mKPD - mmaxKPD) * exp(- h / tau_mKPD)

    GKPD = GmaxKPD * power(mKPD_next, 4) * A

    # KCa current
    mmaxKCaPD = (ConCaPD / (ConCaPD + 3)) * 1 / (1 + exp(-(VsPD + 28.3) / 12.6))
    tau_mkCaPD = (180.6 - 150.2 / (1 + exp(-(VsPD + 46) / 22.7))) * 1000

    mKCaPD_next = mmaxKCaPD
    if h < tau_mkCaPD:
        mKCaPD_next = mmaxKCaPD + (mKCaPD - mmaxKCaPD) * exp(- h / tau_mkCaPD)

    GKCaPD = GmaxKCaPD * power(mKCaPD_next, 4) * A

    # h current
    mmaxhPD = 1 / (1 + exp((VsPD + 75) / 5.5))
    tauhPD = (2 / (exp(-(VsPD + 169.7) / 11.6) + exp((VsPD - 26.7) / 14.3))) * 1000

    mhPD_next = mmaxhPD
    if h < tauhPD:
        mhPD_next = mmaxhPD + (mhPD - mmaxhPD) * exp(- h / tauhPD)

    GhPD = GmaxhPD * mhPD_next * A

    # CaT current
    mmaxCaTPD = 1 / (1 + exp(-(VsPD + 27.1) / 7.2))
    hmaxCaTPD = 1 / (1 + exp((VsPD + 32.1) / 5.5))
    tau_mCaTPD = (43.4 - 42.6 / (1 + exp(-(VsPD + 68.1) / 20.5))) * 1000
    tau_hCaTPD = (210 - 179.6 / (1 + exp(-(VsPD + 55) / 16.9))) * 1000

    mCaTPD_next = mmaxCaTPD
    if h < tau_mCaTPD:
        mCaTPD_next = mmaxCaTPD + (mCaTPD - mmaxCaTPD) * exp(- h / tau_mCaTPD)

    hCaTPD_next = hmaxCaTPD
    if h < tau_hCaTPD:
        hCaTPD_next = hmaxCaTPD + (hCaTPD - hmaxCaTPD) * exp(- h / tau_hCaTPD)

    GCaTPD = GmaxCaTPD * power(mCaTPD_next, 3) * hCaTPD_next * A
    ICaTPD = GCaTPD * (VsPD - ECaPD)

    # A current
    mmaxAPD = 1 / (1 + exp(-(VsPD + 27.2) / 8.7))
    hmaxAPD = 1 / (1 + exp((VsPD + 56.9) / 4.9))
    tau_mAPD = (23.2 - 20.8 / (1 + exp(-(VsPD + 32.9) / 15.2))) * 1000
    tau_hAPD = (77.2 - 58.4 / (1 + exp(-(VsPD + 38.9) / 26.5))) * 1000

    mAPD_next = mmaxAPD
    if h < tau_mAPD:
        mAPD_next = mmaxAPD + (mAPD - mmaxAPD) * exp(- h / tau_mAPD)

    hAPD_next = hmaxAPD
    if h < tau_hAPD:
        hAPD_next = hmaxAPD + (hAPD - hmaxAPD) * exp(- h / tau_hAPD)

    GAPD = GmaxAPD * power(mAPD_next, 3) * hAPD_next * A

    # Proctolin current
    turnoff = t < decent

    GmaxProcPD = GmaxProcPD_0 * turnoff
    mProcPD = 1 / (1 + exp(0.2 * (-55 - VsPD)))
    GProcPD = GmaxProcPD * mProcPD * A

    # update VsPD
    GmPD = GLPD + GNaPD + GKPD + GCaSPD + GCaTPD + GAPD + GhPD + GKCaPD + GProcPD + gLPPD_g
    VinfPD = (GLPD * EL + GNaPD * ENa + GKPD * EK + GCaSPD * ECaPD + GCaTPD * ECaPD + GAPD * EK + GhPD * Eh +
              GKCaPD * EK + GProcPD * EProc + gLPPD_g * EGlu) / GmPD
    tauPD = Cs / GmPD
    VsPD_next = VinfPD + (VsPD - VinfPD) * exp(-h / tauPD)

    # update [Ca2+]
    Ca_infPD = -f * (ICaSPD + ICaTPD) + Ca_init
    ConCaPD_next = Ca_infPD + (ConCaPD - Ca_infPD) * exp(-h / tau_Ca)

    ## calculate LP neuron conductance
    # leak current
    GLLP = GmaxLLP * A

    # CaS current
    ECaLP = Nernstfactor * log(Ca_out / ConCaLP)

    mmaxCaSLP = 1 / (1 + exp(-(VsLP + 33) / 8.1))
    hmaxCaSLP = 1 / (1 + exp((VsLP + 60) / 6.22))
    tau_mCaSLP = (2.8 + 14 / (exp((VsLP + 27) / 10) + exp(-(VsLP + 70) / 13))) * 1000
    tau_hCaSLP = (120 + 300 / (exp((VsLP + 55) / 9) + exp(-(VsLP + 65) / 16))) * 1000

    mCaSLP_next = mmaxCaSLP
    if h < tau_mCaSLP:
        mCaSLP_next = mmaxCaSLP + (mCaSLP - mmaxCaSLP) * exp(- h / tau_mCaSLP)

    hCaSLP_next = hmaxCaSLP
    if h < tau_hCaSLP:
        hCaSLP_next = hmaxCaSLP + (hCaSLP - hmaxCaSLP) * exp(- h / tau_hCaSLP)

    GCaSLP = GmaxCaSLP * power(mCaSLP_next, 3) * hCaSLP_next * A
    ICaSLP = GCaSLP * (VsLP - ECaLP)

    # CaT current
    mmaxCaTLP = 1 / (1 + exp(-(VsLP + 27.1) / 7.2))
    hmaxCaTLP = 1 / (1 + exp((VsLP + 32.1) / 5.5))
    tau_mCaTLP = (43.4 - 42.6 / (1 + exp(-(VsLP + 68.1) / 20.5))) * 1000
    tau_hCaTLP = (210 - 179.6 / (1 + exp(-(VsLP + 55) / 16.9))) * 1000

    mCaTLP_next = mmaxCaTLP
    if h < tau_mCaTLP:
        mCaTLP_next = mmaxCaTLP + (mCaTLP - mmaxCaTLP) * exp(- h / tau_mCaTLP)

    hCaTLP_next = hmaxCaTLP
    if h < tau_hCaTLP:
        hCaTLP_next = hmaxCaTLP + (hCaTLP - hmaxCaTLP) * exp(- h / tau_hCaTLP)

    GCaTLP = GmaxCaTLP * power(mCaTLP_next, 3) * hCaTLP_next * A
    ICaTLP = GCaTLP * (VsLP - ECaLP)

    # Na current
    mmaxNaLP = 1 / (1 + exp(-(VsLP + 25.5) / 5.29))
    hmaxNaLP = 1 / (1 + exp((VsLP + 48.9) / 5.18))
    tau_mNaLP = (2.64 - 2.52 / (1 + exp(-(VsLP + 120) / 25))) * 1000
    tau_hNaLP = (1.34 / (1 + exp(-(VsLP + 62.9) / 10)) * (1.5 + 1 / (1 + exp((VsLP + 34.9) / 3.6)))) * 1000

    mNaLP_next = mmaxNaLP
    if h < tau_mNaLP:
        mNaLP_next = mmaxNaLP + (mNaLP - mmaxNaLP) * exp(- h / tau_mNaLP)

    hNaLP_next = hmaxNaLP
    if h < tau_hNaLP:
        hNaLP_next = hmaxNaLP + (hNaLP - hmaxNaLP) * exp(- h / tau_hNaLP)

    GNaLP = GmaxNaLP * power(mNaLP_next, 3) * hNaLP_next * A

    # K current
    mmaxKLP = 1 / (1 + exp(-(12.3 + VsLP) / 11.8))
    tau_mKLP = (14.4 - 12.8 / (1 + exp(-(VsLP + 28.3) / 19.2))) * 1000

    mKLP_next = mmaxKLP
    if h < tau_mKLP:
        mKLP_next = mmaxKLP + (mKLP - mmaxKLP) * exp(- h / tau_mKLP)

    GKLP = GmaxKLP * power(mKLP_next, 4) * A

    # KCa current
    mmaxKCaLP = (ConCaLP / (ConCaLP + 3)) * 1 / (1 + exp(-(VsLP + 28.3) / 12.6))
    tau_mKCaLP = (180.6 - 150.2 / (1 + exp(-(VsLP + 46) / 22.7))) * 1000

    mKCaLP_next = mmaxKCaLP
    if h < tau_mKCaLP:
        mKCaLP_next = mmaxKCaLP + (mKCaLP - mmaxKCaLP) * exp(- h / tau_mKCaLP)

    GKCaLP = GmaxKCaLP * power(mKCaLP_next, 4) * A

    # h current
    mmaxhLP = 1 / (1 + exp((VsLP + 75) / 5.5))
    tauhLP = (2 / (exp(-(VsLP + 169.7) / 11.6) + exp((VsLP - 26.7) / 14.3))) * 1000

    mhLP_next = mmaxhLP
    if h < tauhLP:
        mhLP_next = mmaxhLP + (mhLP - mmaxhLP) * exp(- h / tauhLP)

    GhLP = GmaxhLP * mhLP_next * A

    # A current
    mmaxALP = 1 / (1 + exp(-(VsLP + 27.2) / 8.7))
    hmaxALP = 1 / (1 + exp((VsLP + 56.9) / 4.9))
    tau_mALP = (23.2 - 20.8 / (1 + exp(-(VsLP + 32.9) / 15.2))) * 1000
    tau_hALP = (77.2 - 58.4 / (1 + exp(-(VsLP + 38.9) / 26.5))) * 1000

    mALP_next = mmaxALP
    if h < tau_mALP:
        mALP_next = mmaxALP + (mALP - mmaxALP) * exp(- h / tau_mALP)

    hALP_next = hmaxALP
    if h < tau_hALP:
        hALP_next = hmaxALP + (hALP - hmaxALP) * exp(- h / tau_hALP)

    GALP = GmaxALP * power(mALP_next, 3) * hALP_next * A

    # Proctolin current
    turnoff = t < decent

    GmaxProcLP = GmaxProcLP_0 * turnoff
    mProcLP = 1 / (1 + exp(0.2 * (-55 - VsLP)))
    GProcLP = GmaxProcLP * mProcLP * A

    # update VsLP
    GmLP = GLLP + GNaLP + GKLP + GCaSLP + GCaTLP + GALP + GhLP + GKCaLP + GProcLP + gPDLP_g + gPDLP_a + gPYLP_g
    VinfLP = (GLLP * EL + GNaLP * ENa + GKLP * EK + GCaSLP * ECaLP + GCaTLP * ECaLP + GALP * EK + GhLP * Eh +
              GKCaLP * EK + GProcLP * EProc + gPDLP_g * EGlu + gPYLP_g * EGlu + gPDLP_a * EACh) / GmLP
    tauLP = Cs / GmLP
    VsLP_next = VinfLP + (VsLP - VinfLP) * exp(-h / tauLP)

    # update [Ca2+]
    Ca_infLP = -f * (ICaSLP + ICaTLP) + Ca_init
    ConCaLP_next = Ca_infLP + (ConCaLP - Ca_infLP) * exp(-h / tau_Ca)

    ## calculate PY neuron conductance
    # leak current
    GLPY = GmaxLPY * A

    # CaS current
    ECaPY = Nernstfactor * log(Ca_out / ConCaPY)

    mmaxCaSPY = 1 / (1 + exp(-(VsPY + 33) / 8.1))
    hmaxCaSPY = 1 / (1 + exp((VsPY + 60) / 6.22))
    tau_mCaSPY = (2.8 + 14 / (exp((VsPY + 27) / 10) + exp(-(VsPY + 70) / 13))) * 1000
    tau_hCaSPY = (120 + 300 / (exp((VsPY + 55) / 9) + exp(-(VsPY + 65) / 16))) * 1000

    mCaSPY_next = mmaxCaSPY
    if h < tau_mCaSPY:
        mCaSPY_next = mmaxCaSPY + (mCaSPY - mmaxCaSPY) * exp(- h / tau_mCaSPY)

    hCaSPY_next = hmaxCaSPY
    if h < tau_hCaSPY:
        hCaSPY_next = hmaxCaSPY + (hCaSPY - hmaxCaSPY) * exp(- h / tau_hCaSPY)

    GCaSPY = GmaxCaSPY * power(mCaSPY_next, 3) * hCaSPY_next * A
    ICaSPY = GCaSPY * (VsPY - ECaPY)

    # Na current
    mmaxNaPY = 1 / (1 + exp(-(VsPY + 25.5) / 5.29))
    hmaxNaPY = 1 / (1 + exp((VsPY + 48.9) / 5.18))
    tau_mNaPY = (2.64 - 2.52 / (1 + exp(-(VsPY + 120) / 25))) * 1000
    tau_hNaPY = (1.34 / (1 + exp(-(VsPY + 62.9) / 10)) * (1.5 + 1 / (1 + exp((VsPY + 34.9) / 3.6)))) * 1000

    mNaPY_next = mmaxNaPY
    if h < tau_mNaPY:
        mNaPY_next = mmaxNaPY + (mNaPY - mmaxNaPY) * exp(- h / tau_mNaPY)

    hNaPY_next = hmaxNaPY
    if h < tau_hNaPY:
        hNaPY_next = hmaxNaPY + (hNaPY - hmaxNaPY) * exp(- h / tau_hNaPY)

    GNaPY = GmaxNaPY * power(mNaPY_next, 3) * hNaPY_next * A

    # K current
    mmaxKPY = 1 / (1 + exp(-(12.3 + VsPY) / 11.8))
    tau_mKPY = (14.4 - 12.8 / (1 + exp(-(VsPY + 28.3) / 19.2))) * 1000

    mKPY_next = mmaxKPY
    if h < tau_mKPY:
        mKPY_next = mmaxKPY + (mKPY - mmaxKPY) * exp(- h / tau_mKPY)

    GKPY = GmaxKPY * power(mKPY_next, 4) * A

    # KCa current
    mmaxKCaPY = (ConCaPY / (ConCaPY + 3)) * 1 / (1 + exp(-(VsPY + 28.3) / 12.6))
    tau_mKCaPY = (180.6 - 150.2 / (1 + exp(-(VsPY + 46) / 22.7))) * 1000

    mKCaPY_next = mmaxKCaPY
    if h < tau_mKCaPY:
        mKCaPY_next = mmaxKCaPY + (mKCaPY - mmaxKCaPY) * exp(- h / tau_mKCaPY)

    GKCaPY = GmaxKCaPY * power(mKCaPY_next, 4) * A

    # h current
    mmaxhPY = 1 / (1 + exp((VsPY + 75) / 5.5))
    tauhPY = (2 / (exp(-(VsPY + 169.7) / 11.6) + exp((VsPY - 26.7) / 14.3))) * 1000

    mhPY_next = mmaxhPY
    if h < tauhPY:
        mhPY_next = mmaxhPY + (mhPY - mmaxhPY) * exp(- h / tauhPY)

    GhPY = GmaxhPY * mhPY_next * A

    # CaT current
    mmaxCaTPY = 1 / (1 + exp(-(VsPY + 27.1) / 7.2))
    hmaxCaTPY = 1 / (1 + exp((VsPY + 32.1) / 5.5))
    tau_mCaTPY = (43.4 - 42.6 / (1 + exp(-(VsPY + 68.1) / 20.5))) * 1000
    tau_hCaTPY = (210 - 179.6 / (1 + exp(-(VsPY + 55) / 16.9))) * 1000

    mCaTPY_next = mmaxCaTPY
    if h < tau_mCaTPY:
        mCaTPY_next = mmaxCaTPY + (mCaTPY - mmaxCaTPY) * exp(- h / tau_mCaTPY)

    hCaTPY_next = hmaxCaTPY
    if h < tau_hCaTPY:
        hCaTPY_next = hmaxCaTPY + (hCaTPY - hmaxCaTPY) * exp(- h / tau_hCaTPY)

    GCaTPY = GmaxCaTPY * power(mCaTPY_next, 3) * hCaTPY_next * A
    ICaTPY = GCaTPY * (VsPY - ECaPY)

    # A current
    mmaxAPY = 1 / (1 + exp(-(VsPY + 27.2) / 8.7))
    hmaxAPY = 1 / (1 + exp((VsPY + 56.9) / 4.9))
    tau_mAPY = (23.2 - 20.8 / (1 + exp(-(VsPY + 32.9) / 15.2))) * 1000
    tau_hAPY = (77.2 - 58.4 / (1 + exp(-(VsPY + 38.9) / 26.5))) * 1000

    mAPY_next = mmaxAPY
    if h < tau_mAPY:
        mAPY_next = mmaxAPY + (mAPY - mmaxAPY) * exp(- h / tau_mAPY)

    hAPY_next = hmaxAPY
    if h < tau_hAPY:
        hAPY_next = hmaxAPY + (hAPY - hmaxAPY) * exp(- h / tau_hAPY)

    GAPY = GmaxAPY * power(mAPY_next, 3) * hAPY_next * A

    # Proctolin current
    GmaxProcPY = GmaxProcPY_0 * turnoff
    mProcPY = 1 / (1 + exp(0.2 * (-55 - VsPY)))
    GProcPY = GmaxProcPY * mProcPY * A

    # update VsPY
    GmPY = GLPY + GNaPY + GKPY + GCaSPY + GCaTPY + GAPY + GhPY + GKCaPY + GProcPY + gPDPY_g + gPDPY_a + gLPPY_g
    VinfPY = (GLPY * EL + GNaPY * ENa + GKPY * EK + GKCaPY * EK + GCaSPY * ECaPY + GCaTPY * ECaPY +
              GAPY * EK + GhPY * Eh + GProcPY * EProc + gPDPY_g * EGlu + gPDPY_a * EACh + gLPPY_g * EGlu) / GmPY
    tauPY = Cs / GmPY
    VsPY_next = VinfPY + (VsPY - VinfPY) * exp(-h / tauPY)

    # update [Ca2+]
    Ca_infPY = -f * (ICaSPY + ICaTPY) + Ca_init
    ConCaPY_next = Ca_infPY + (ConCaPY - Ca_infPY) * exp(-h / tau_Ca)

    ## burst matrix
    burst = [
        [sPDLPglu_next][0], [sPDLPach_next][0], [sPDPYglu_next][0], [sPDPYach_next][0],
        [sLPPDglu_next][0], [sLPPYglu_next][0], [sPYLPglu_next][0],
        [ConCaPD_next][0], [VsPD_next][0], [mKPD_next][0], [mNaPD_next][0],
        [hNaPD_next][0], [mCaSPD_next][0], [hCaSPD_next][0], [mKCaPD_next][0],
        [mhPD_next][0], [mCaTPD_next][0], [hCaTPD_next][0], [mAPD_next][0], [hAPD_next][0],
        [ConCaLP_next][0], [VsLP_next][0],[mKLP_next][0], [mNaLP_next][0],
        [hNaLP_next][0], [mCaSLP_next][0], [hCaSLP_next][0], [mKCaLP_next][0],
        [mhLP_next][0], [mCaTLP_next][0], [hCaTLP_next][0], [mALP_next][0], [hALP_next][0],
        [ConCaPY_next][0], [VsPY_next][0], [mKPY_next][0], [mNaPY_next][0],
        [hNaPY_next][0], [mCaSPY_next][0], [hCaSPY_next][0], [mKCaPY_next][0],
        [mhPY_next][0], [mCaTPY_next][0], [hCaTPY_next][0], [mAPY_next][0], [hAPY_next][0],
    ]

    return burst
