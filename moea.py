# an example: https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
# python 2.7

from __future__ import print_function, division

import array
import random
from deap import base
from deap import creator
from deap import tools
import scipy.io as sio
from os.path import exists
from os import makedirs

import fortin2013
import simulation


NDIM = 34
BOUND_LOW = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
BOUND_UP = [400, 500, 800, 100, 100, 200, 200, 400, 200, 500, 120, 600, 100, 100, 100, 400, 600, 500, 400, 100, 100, 224, 800, 400, 200, 200, 200, 200, 200, 200, 200, 500, 500, 500]

NGEN = 60
NOBJS = 6
NPOP = 252                 # NPOP = int(factorial(NOBJS+5-1) / (factorial(5) * factorial(NOBJS-1))) = 252
CXPB = 0.8
INDPB = 1 / NDIM

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJS)
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", simulation.evaluate)     # define your file here
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=INDPB)
toolbox.register("select", fortin2013.selNSGA2)

# import multiprocessing
# pool = multiprocessing.Pool()
# toolbox.register("map",pool.map)

from scoop import futures
toolbox.register("map",futures.map)

hof = tools.ParetoFront()
the_best = tools.HallOfFame(NPOP)          # save the best individuals cumulatively
# the_best = tools.HallOfFame(2)

def save_gen(gen_num, pop):
    gen_id = 0
    hof_id = 0
    best_id = 0

    # save all individuals
    sim_dir = 'out/gen_{0}'.format(gen_num)
    if not exists(sim_dir):
        makedirs(sim_dir)

    for ind in pop:
        sio.savemat('{0}/{1}_conductances.mat'.format(sim_dir, gen_id), {'conductances': ind[:]})
        sio.savemat('{0}/{1}_errs.mat'.format(sim_dir, gen_id), {'errs': ind.fitness.values})
        sio.savemat('{0}/{1}_features.mat'.format(sim_dir, gen_id), {'features': ind.behaviors})
        gen_id += 1

    # save pareto front of each generation
    # hof_dir = 'out/hof_gen_{0}'.format(gen_num)
    # if not exists(hof_dir):
    #     makedirs(hof_dir)
    # if hof is not None:
    #     hof.update(pop)
    # for ind in hof:
    #     sio.savemat('{0}/{1}_conductances.mat'.format(hof_dir, hof_id), {'conductances': ind[:]})
    #     sio.savemat('{0}/{1}_errs.mat'.format(hof_dir, hof_id), {'errs': ind.fitness.values})
    #     sio.savemat('{0}/{1}_features.mat'.format(hof_dir, hof_id), {'features': ind.behaviors})
    #     hof_id += 1
    # if hof is not None:
    #     # clear them for next gen
    #     hof.clear()

    # save HallofFame (n = NPOP)
    the_best_dir = 'out/the_best_{0}'.format(gen_num)
    if not exists(the_best_dir):
        makedirs(the_best_dir)
    if the_best is not None:
        the_best.update(pop)
    for ind in the_best:
        sio.savemat('{0}/{1}_conductances.mat'.format(the_best_dir, best_id), {'conductances': ind[:]})
        sio.savemat('{0}/{1}_errs.mat'.format(the_best_dir, best_id), {'errs': ind.fitness.values})
        sio.savemat('{0}/{1}_features.mat'.format(the_best_dir, best_id), {'features': ind.behaviors})
        sio.savemat('{0}/{1}_errs2nd.mat'.format(the_best_dir, best_id), {'errs2nd': ind.extraErrs})
        # if gen_num+1 == NGEN:
        #     sio.savemat('{0}/{1}_volts.mat'.format(the_best_dir, best_id), {'voltages': ind.voltage})
        best_id += 1


def main():
    # init. population
    population = toolbox.population(n=NPOP)

    # evaluate every individual
    fitnesses = list(toolbox.map(toolbox.evaluate, population))
    # print (fitnesses[0][0])
    for ind, fit in zip(population, fitnesses):
        ind.voltage = fit[0]
        ind.behaviors = fit[1]
        ind.fitness.values = fit[2]
        ind.extraErrs = fit[3]

    # the size of individuals = k: assign the crowding distance; no actual selection is done
    # selNSGA2(individual, k): k is the # of individuals to select; call sortNondominated(individuals,k) and assignCrowdingDist(individuals)
    # sortNondominated(individuals, k): returns a list of Pareto fronts, the first list includes the first pareto front
    population = toolbox.select(population, len(population))
    save_gen(0, population)

    # Begin the evolution
    for gen in range(1, NGEN):
        print('Beginning generation {0}/{1}...'.format(gen, NGEN))

        # define offspring
        offspring = [toolbox.clone(ind) for ind in population]

        # crossover, mutate
        # cxSimulatedBinaryBounded(ind1, ind2, eta, low, up): returns a tuple of two individuals
        # mutPolynomialBounded(individual, eta, low, up, indpb): returns a tuple of one individual
        # eta: crowding degree of the crossover. A high eta will produce children resembling to their parents
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]    # individuals that are crossovered and/or mutated
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.voltage = fit[0]
            ind.behaviors = fit[1]
            ind.fitness.values = fit[2]
            ind.extraErrs = fit[3]

        # select the next generation
        population = toolbox.select(population + offspring, NPOP)
        save_gen(gen, population)

    # remember to return hof when hof code is turned on
    return population, hof


if __name__ == "__main__":
    main()
    print('Done!')
