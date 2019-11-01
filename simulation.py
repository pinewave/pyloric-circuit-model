# 02/15/2016
# Author: Kun Tian
# Translate the conductance-based pyloric circuit model code written in MATLAB to PYTHON 2.7
# Project Replaces MATLAB files, network_RRAD_MOEA, init_netRRAD, expEuler, update_netVar, plot_neuroNet

import settings
import expEuler
import features
import objectives
import objectives2nd
#import uuid

def evaluate(individual):

    #sim_uuid = uuid.uuid4()
    #print("Beginning pyloric circuit simulation {0}...".format(sim_uuid))

    settings.load(individual)
    y = expEuler.run(settings.func, settings.y0, settings.t, settings.params)
    volts = [y[:,8], y[:,21], y[:,34]]
    featuresOf = features.extract_circuit_features(y, settings.tmin, settings.tmax, settings.trange, settings.h)
    errs = objectives.errFunc(featuresOf)
    errs2nd = objectives2nd.errFunc(featuresOf)

    return [volts, featuresOf, errs, errs2nd]
