# 02/15/2016
# Author: Kun Tian
# A file for storing settings to run the simulation

from collections import OrderedDict
import pyloric

# odeint
func = pyloric.step
t = None
y0 = None
params = None

# t
tmin = 0
tmax = 15e6       # us
h = 50            # us
trange = 12e6

# y0 - initialization
y0_dict = OrderedDict([
    ('ds12g', 0.0474),
    ('ds12a', 0.0474),
    ('ds13g', 0.0474),
    ('ds13a', 0.0474),
    ('ds21g', 0.0474),
    ('ds23g', 0.0474),
    ('ds32g', 0.0474),
    ('ConCaPD', 0.05),
    ('VsPD', -50.0),
    ('mKPD', 0.0390),
    ('mNaPD', 0.009),
    ('hNaPD', 0.553),
    ('mCaSPD', 0.109),
    ('hCaSPD', 0.166),
    ('mKCaPD', 0.003),
    ('mhPD', 0.011),
    ('mCaTPD', 0.040),
    ('hCaTPD', 0.963),
    ('mAPD', 0.068),
    ('hAPD', 0.197),
    ('ConCaLP', 0.05),
    ('VsLP', -50.0),
    ('mKLP', 0.039),
    ('mNaLP', 0.009),
    ('hNaLP', 0.553),
    ('mCaSLP', 0.109),
    ('hCaSLP', 0.166),
    ('mKCaLP', 0.003),
    ('mhLP', 0.011),
    ('mCaTLP', 0.040),
    ('hCaTLP', 0.963),
    ('mALP', 0.068),
    ('hALP', 0.1970),
    ('ConCaPY', 0.0500),
    ('VsPY', -50.0),
    ('mKPY', 0.0390),
    ('mNaPY', 0.0090),
    ('hNaPY', 0.5530),
    ('mCaSPY', 0.1090),
    ('hCaSPY', 0.1660),
    ('mKCaPY', 0.0030),
    ('mhPY', 0.0110),
    ('mCaTPY', 0.0400),
    ('hCaTPY', 0.9630),
    ('mAPY', 0.0680),
    ('hAPY', 0.1970)
])

# params - constants
constants_dict = OrderedDict([
    ('Cs', 0.6283),                # capacitance; nF
    ('A', 0.628e-3),               # area; cm^2
    ('EL', -50),                   # revese potential for the leak current; mV
    ('EK', -80),                   # for IK, IKCa, and IA
    ('Eh', -20),
    ('ENa', 50),
    ('EProc', -10),
    ('decent', 50e6),              # time to turn on decentralization; us
    ('tau_Ca', 200e3),             # calcium update time constant; us
    ('f', 14.961e3),               # convert current to concentration; uM/uA
    ('Ca_init', 0.05),             # steady-state Ca2+ if no Ca2+ flows across the membrane; uM
    ('Ca_out', 3000),              # outer [Ca2+]; uM
    ('R', 8.31451),                # J/(mol*K)
    ('Farad', 96485.3415),         # C/mol
    ('Temp', 283.0),               # K
    ('Z_Ca', 2.0),
    ('Nernstfactor', None),
    ('EGlu', -70),
    ('EACh', -80),
    ('k_glu', 40000),              # rate constant for transmitter-receptor dissociation; Prinz et al., 2004; us
    ('k_ach', 100000),
    ('Vth', -35),                  # half-activation voltage of the synapse; Prinz et al., 2004; mV
    ('deltaV', 5)                  # determines the slope of the activation curve; Prinz et al., 2004
])


def load(individual):

    # y0
    global y0
    y0 = []
    for key in y0_dict:
        y0.append(y0_dict[key])

    # time
    global t, tmin, tmax, h, trange
    t = range(int(tmin), int(tmax + h), int(h))

    # conductances
    global params
    conductances = []
    for conductance in individual:
        conductances.append(conductance)

    # constants
    R = constants_dict["R"]
    Farad = constants_dict["Farad"]
    Temp = constants_dict["Temp"]
    Z_Ca = constants_dict["Z_Ca"]
    constants_dict["Nernstfactor"] = 1000.0 * R * Temp / (Z_Ca * Farad)   # mV; J/C = 1V; 1V = 1000 mV
    constants = []
    for key in constants_dict:
        constants.append(constants_dict[key])

    params = [h, conductances, constants]
