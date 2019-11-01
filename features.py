# Author: Kun Tian
# Updated: 05/23/16

from __future__ import division
import numpy as np


def extract_circuit_features(y, tmin, tmax, trange, h):
    # time period
    start_t = int((tmax - trange) / h)
    end_t = int(tmax / h)
    tspan = np.arange(int(tmin), int(tmax + h), int(h))    # converting t to a numpy array is necessary for ISI calc     

    # spike per burst
    # spbPD_meanE = 5.7
    # spbPD_stdE = 0.58
    # spbLP_meanE = 7.9
    # spbLP_stdE = 2.05
    # spbPY_meanE = 5.7
    # spbPY_stdE = 1.145
    #
    # # burst duration in control, s
    # bdPD_meanE = 0.15
    # bdPD_stdE = 0.035
    # bdLP_meanE = 0.29
    # bdLP_stdE = 0.055
    # bdPY_meanE = 0.29
    # bdPY_stdE = 0.055
    # 
    # minimum and maximum ISI (Tomasz et al., 2009)
    # isiMin = 0.02   # s
    # isiMax = 0.045  # s

    # spike per burst
    spbPD_meanE = 5.7
    spbPD_stdE = 0.58
    spbLP_meanE = 7.9
    spbLP_stdE = 2.05
    spbPY_meanE = 5.7
    spbPY_stdE = 1.145

    # burst duration in control, s
    bdPD_meanE = 0.15
    bdPD_stdE = 0.035
    bdLP_meanE = 0.29
    bdLP_stdE = 0.055
    bdPY_meanE = 0.29
    bdPY_stdE = 0.055

    # maximum ISI, s
    isiPDMaxE = (bdPD_meanE + 2 * bdPD_stdE) / (spbPD_meanE - 2 * spbPD_stdE)
    isiLPMaxE = (bdLP_meanE + 2 * bdLP_stdE) / (spbLP_meanE - 2 * spbLP_stdE)
    isiPYMaxE = (bdPY_meanE + 2 * bdPY_stdE) / (spbPY_meanE - 2 * spbPY_stdE)
      
    # load data
    pd_trace = y[:, 8]
    lp_trace = y[:, 21]
    py_trace = y[:, 34]

    # features of single neurons: [isi, spb, peaks_std, startBstT, endBstT, totalBst, bd, ibi, totalRidge, isis_std]
    pd_features = extract_neuron_features(pd_trace, tspan, trange, h, start_t, end_t, isiPDMaxE)
    lp_features = extract_neuron_features(lp_trace, tspan, trange, h, start_t, end_t, isiLPMaxE)
    py_features = extract_neuron_features(py_trace, tspan, trange, h, start_t, end_t, isiPYMaxE)

    pd_startBstT = pd_features[3]
    lp_startBstT = lp_features[3]
    py_startBstT = py_features[3]

    pd_endBstT = pd_features[4]
    lp_endBstT = lp_features[4]
    py_endBstT = py_features[4]

    pd_totalBst = pd_features[5]
    lp_totalBst = lp_features[5]
    py_totalBst = py_features[5]

    pd_bd = pd_features[6]
    lp_bd = lp_features[6]
    py_bd = py_features[6]

    # features of the circuit: [gaps, pre_first]
    pdlp_gaps = calc_gaps(tspan, pd_startBstT, pd_endBstT, pd_totalBst, lp_startBstT, lp_endBstT, lp_totalBst)
    lppy_gaps = calc_gaps(tspan, lp_startBstT, lp_endBstT, lp_totalBst, py_startBstT, py_endBstT, py_totalBst)

    pl_gaps = pdlp_gaps[0]
    lp_gaps = lppy_gaps[0]

    pd_first = pdlp_gaps[1]
    lp_first = lppy_gaps[1]

    # features of the circuit: [period_pd, PDoff, LPon, LPoff, PYon, PYoff, pp_gaps]
    period_features = calc_period(tspan, pd_startBstT, pl_gaps, lp_gaps, pd_totalBst, lp_totalBst, py_totalBst, pd_bd, lp_bd, py_bd)
    ordered = calc_order(tspan, pd_first, lp_first, pd_startBstT, py_startBstT, pd_totalBst, lp_totalBst, py_totalBst)

    return [pd_features, lp_features, py_features, pdlp_gaps, lppy_gaps, period_features, ordered]


def extract_neuron_features(v, tspan, trange, h, start_t, end_t, isiMax):
    slope = np.sign(np.diff(v))                                                                                 # calculate deltaV; 1 if > 0, 0 if = 0, -1 if < 0
    spk_idx, totalSpk, totalRidge = find_spikes(v, slope, trange, h, start_t, end_t)                            # detect spikes
    isi = np.diff(tspan[spk_idx]) / 1e6                                                                         # calculate ISIs; unit is s
    spb, peaks_std, isis_std, startBstT, endBstT, totalBst = find_bursts(v, spk_idx, isi, isiMax)       # detect bursts
    bd, ibi = burst_features(tspan, startBstT, endBstT, totalBst)                                               # calculate burst duration and IBI; unit is s
    
    neuron_features = [isi, spb, peaks_std, startBstT, endBstT, totalBst, bd, ibi, totalRidge, isis_std]
    
    return neuron_features


def find_spikes(v, dv, trange, h, start_t, end_t):
    # initialization
    max_trange = int(trange / h)
    spkIdx = np.zeros((max_trange,), dtype=np.int)
    spkCount = 1                                   # total # of spikes; assume the first time point is a spike (make it easy to calculate ISI)
    spkIdx[0] = start_t                            # indices of spikes
    ridgeCount = 0

    # spikes defined by [-1 1] slope patten and V > 0 
    for i in range(start_t + 1, end_t):
        if dv[i - 1] == 1 and dv[i] == -1 and v[i] > 0:
            spkCount += 1
            spkIdx[spkCount] = i
        elif dv[i-1] == 1 and dv[i] == -1 and -30 < v[i] <= 0:    # a "ridge"
            ridgeCount += 1
            
    spkCount += 1                   # assume the last time point is a spike (make it easy to calculate ISI)
    spkIdx[spkCount] = end_t
    spkIdx = spkIdx[spkIdx > 0]     # truncate unfilled slots
    
    return spkIdx, spkCount, ridgeCount
            

def find_bursts(v, spk_indices, ISI, isiMaxE):
    # initialization
    spkPerBst = np.zeros((len(ISI), 1), int)         # count spikes per burst
    startBst = np.zeros((len(ISI), 1), int)          # the start time of a burst / time of the 1st spike in a burst
    endBst = np.zeros((len(ISI), 1), int)            # the end time of a burst / time of the last spike in a burst
    peaksStd = np.zeros((len(ISI), 1))               # calculate the std of the peak amplitude of spikes in a burst
    isisStd = np.zeros((len(ISI), 1))                # calculate the std of the ISI in a burst
    bstCount = 0                                     # count # of bursts
        
    # detect bursts
    startSpkIdx = 0       # indice of the 1st spike in a burst
    isiPerBst = 0         # ISI per burst
    peaks = []            # record the peak value of each spike in a burst
    isis = []             # record the duration of ISIs in a burst
    
    for i in range(1, len(ISI) + 1):
        if ISI[i - 1] > isiMaxE:
            if startSpkIdx == 0:
                startSpkIdx = i       # this could be the start of a burst
            elif isiPerBst > 0:       # burst is detected when startSpkIdx > 0 and isiPerBst > 0; this garantees that partial burst at the end will not be counted
                bstCount += 1
                spkPerBst[bstCount - 1] = isiPerBst + 1
                peaksStd[bstCount - 1] = np.std(peaks)
                isisStd[bstCount - 1] = np.std(isis)
                startBst[bstCount - 1] = spk_indices[startSpkIdx]
                endBst[bstCount - 1] = spk_indices[i - 1]
                isiPerBst = 0         # reset
                peaks = []            # reset
                isis = []             # reset
                startSpkIdx = i       # reset
        elif startSpkIdx > 0:
            isiPerBst += 1
            peaks.append(v[spk_indices[i - 1]])
            isis.append(ISI[i - 1])
        else:                         # skip partial burst (if exists) at the beginning
            pass
            
    # truncate unfilled slots
    spkPerBst = spkPerBst[spkPerBst > 0]
    peaksStd = peaksStd[peaksStd > 0]
    isisStd = isisStd[isisStd > 0]
    startBst = startBst[startBst > 0]
    endBst = endBst[endBst > 0]
    
    return spkPerBst, peaksStd, isisStd, startBst, endBst, bstCount
                

def  burst_features(tspan, startBstT, endBstT, totalBst):
    # calculate burst duration
    if totalBst > 0:
        bd = np.zeros((totalBst,1))
        for i in range(0, totalBst):
            bd[i] = (tspan[endBstT[i]] - tspan[startBstT[i]]) / 1e6     # convert the time unit from us to s
    else:
        bd = 0
        
    # calculate inter-burst interval (IBI)
    if totalBst > 1:
        ibi = np.zeros((totalBst - 1,1)) 
        for j in range(0, totalBst - 1):
            ibi[j] = (tspan[startBstT[j+1]] - tspan[endBstT[j]]) / 1e6
    else:
        ibi = 0  
        
    return bd, ibi


def calc_gaps(tspan, pre_startBstT, pre_endBstT, pre_totalBst, post_startBstT, post_endBstT, post_totalBst):
    # initialization
    gaps_len = np.minimum(pre_totalBst, post_totalBst)
    gaps = np.zeros((gaps_len, 1))
    pre_first = False
    
    # calc
    if pre_totalBst != 0 and post_totalBst != 0:
        if pre_startBstT[0] <= post_startBstT[0] and (pre_totalBst - post_totalBst == 0 or pre_totalBst - post_totalBst == 1):     # PD before LP, or LP before PY
            pre_first = True
            for i in range(0, gaps_len):
                gaps[i] = (tspan[post_startBstT[i]] - tspan[pre_endBstT[i]]) / 1e6
        elif pre_startBstT[0] > post_endBstT[0] and post_totalBst - pre_totalBst == 0:                                             # LP before PD, or PY before LP
            for i in range(0, gaps_len - 1):
                gaps[i] = (tspan[post_startBstT[i + 1]] - tspan[pre_endBstT[i]]) / 1e6
            gaps = gaps[0:gaps_len-1]    # truncate the last unfilled slot
        elif pre_startBstT[0] > post_endBstT[0] and post_totalBst - pre_totalBst == 1:                                             # LP before PD, or PY before LP
            for i in range(0, gaps_len):
                gaps[i] = (tspan[post_startBstT[i + 1]] - tspan[pre_endBstT[i]]) / 1e6
    else:
        gaps = -1

    return gaps, pre_first

def calc_period(tspan, pd_startBstT, pl_gaps, lp_gaps, pd_totalBst, lp_totalBst, py_totalBst, pd_bd, lp_bd, py_bd):
    # initialization
#    global pp_gaps
    if np.mean(pl_gaps) > -1:
        plgaps_len = len(pl_gaps)
    else:
        plgaps_len = 0

    if np.mean(lp_gaps) > -1:
        lpgaps_len = len(lp_gaps)
    else:
        lpgaps_len = 0

    period_len = np.minimum(plgaps_len, lpgaps_len)    # period_len = np.amin([pd_totalBst, lp_totalBst, py_totalBst])
    period = np.zeros((period_len, 1))

    if pd_totalBst >= 1:
        period_pd = np.zeros((pd_totalBst-1, 1))
        pp_gaps = np.zeros((pd_totalBst-1, 1))

    PDoff = np.zeros((period_len, 1))
    LPon = np.zeros((period_len, 1))
    LPoff = np.zeros((period_len, 1))
    PYon = np.zeros((period_len, 1))
    PYoff = np.zeros((period_len, 1))
    
    # calc period of the pyloric circuit
    if plgaps_len != 0 and lpgaps_len != 0:
        for i in range(0, period_len):
            period[i] = pd_bd[i] + pl_gaps[i] + lp_bd[i] + lp_gaps[i] + py_bd[i]
    else:
        period = 0

    # calc PY-to-PD gap and phases
    if np.mean(period) != 0 and pd_totalBst > 1:
        circuit_len = np.minimum(period_len, pd_totalBst-1)
        for i in range(0, circuit_len):
            period_pd[i] = (tspan[pd_startBstT[i+1]] - tspan[pd_startBstT[i]]) / 1e6
            pp_gaps[i] = period_pd[i] - period[i]
            PDoff[i] = pd_bd[i] / period_pd[i]
            LPon[i] = (pd_bd[i] + pl_gaps[i]) / period_pd[i]
            LPoff[i] = (pd_bd[i] + pl_gaps[i] + lp_bd[i]) / period_pd[i]
            PYon[i] = (period[i] - py_bd[i]) / period_pd[i]
            PYoff[i] = period[i] / period_pd[i]
    else:
        period_pd = 0
        pp_gaps = -1
        PDoff = 0
        LPon = 0
        LPoff = 0
        PYon = 0
        PYoff = 0
        
    return period_pd, PDoff, LPon, LPoff, PYon, PYoff, pp_gaps
        

def calc_order(tspan, pd_first, lp_first, pd_startBstT, py_startBstT, pd_totalBst, lp_totalBst, py_totalBst):
    ordered = False                                                # true if fire in the right order of PD-LP-PY
    if pd_totalBst != 0 and lp_totalBst != 0 and py_totalBst != 0:
        if pd_first is True and lp_first is True:
            ordered = True
        elif tspan[py_startBstT[0]] < tspan[pd_startBstT[0]] and ((pd_first is True and lp_first is False) or (pd_first is False and lp_first is True)):
            ordered = True
            
    return ordered