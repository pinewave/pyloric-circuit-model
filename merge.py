# merge (conductance, errs) of individuals in the same generation

from os import listdir
from os.path import isdir, isfile, join
import scipy.io as sio
import numpy as np
from os.path import exists
from os import makedirs

# cat = 'cont_42'
# merged_dir = '%s/py_recovery_v2_cont/merged' % cat
# out_dir = '%s/py_recovery_v2_cont/out' % cat

merged_dir = 'merged'
out_dir = 'out'

def merge_data(data_dir):

    # input directory (e.g. 'out/gen_0')
    input_dir = join(out_dir, data_dir)

    # for a given data directory, get unique guids from conductance and err files
    guids = list(set(f.split('_')[0] for f in listdir(input_dir) if isfile(join(input_dir, f)) and f.endswith('mat')))
    print(len(guids))

    # for unique guids, load conductances and errs into corresponding numpy arrays
    individuals = np.array([])
    for j, guid in enumerate(guids):
        conductances = sio.loadmat('%s/%s_conductances.mat' % (input_dir, guid))['conductances'][0]
        errs = sio.loadmat('%s/%s_errs.mat' % (input_dir, guid))['errs'][0]
        errs2nd = sio.loadmat('%s/%s_errs2nd.mat' % (input_dir, guid))['errs2nd'][0]
        fm = sio.loadmat('%s/%s_features.mat' % (input_dir, guid))['features'][0,5]

        if len(fm[:,0]) > 1:
            period = np.mean(fm[0])
            PD_off = np.mean(fm[1])
            LP_on = np.mean(fm[2])
            LP_off = np.mean(fm[3])
            PY_on = np.mean(fm[4])
            PY_off = np.mean(fm[5])
        else:
            period = 0
            PD_off = 0
            LP_on = 0
            LP_off = 0
            PY_on = 0
            PY_off = 0
        feats = [period, PD_off, LP_on, LP_off, PY_on, PY_off]
        id = [int(guid)]

        if np.mean(errs) == 0 and errs2nd[0] <= 5 and errs2nd[1] == 0 and np.count_nonzero(feats) == 6:    # criteria for convergence satisfied
            row = np.concatenate((id, conductances, feats,  errs, errs2nd), axis=0)
            if individuals.size == 0:
                individuals = np.hstack((individuals, np.array(row)))
            else:
                individuals = np.vstack((individuals, np.array(row)))

    # output directory
    output_dir = join(merged_dir, data_dir.rsplit('_', 1)[0])
    if not exists(output_dir):
        makedirs(output_dir)

    # save numpy arrays into a merged file
    gen_num = data_dir.rsplit('_', 1)[1]
    # sio.savemat('%s/%s_guids.mat' % (output_dir, gen_num), {'guids': guids})
    sio.savemat('%s/%s_inds.mat' % (output_dir, gen_num), {'inds': individuals})


def merge_gen():

    print('Start gen data merge...')
    gen_dirs = [d for d in listdir(out_dir) if isdir(join(out_dir, d)) and d.split('_')[0] == 'gen']
    num_gens = len(gen_dirs) - 1
    for i, gen_dir in enumerate(gen_dirs):
        merge_data(gen_dir)
        print("gen %d/%d..." % (i, num_gens))
    print('Finished gen data merge...')


def merge_hof():

    print('Start hof_gen data merge...')
    hof_dirs = [d for d in listdir(out_dir) if isdir(join(out_dir, d)) and d.split('_')[0] == 'hof']
    num_gens = len(hof_dirs) - 1
    for i, hof_dir in enumerate(hof_dirs):
        merge_data(hof_dir)
        print("hof_gen %d/%d..." % (i, num_gens))
    print('Finished hof_gen data merge...')


def merge_the_best():

    print('Start the_best data merge...')
    the_best_dirs = [d for d in listdir(out_dir) if isdir(join(out_dir, d)) and d.split('_')[0] == 'the']
    num_gens = len(the_best_dirs) - 1
    for i, the_best_dir in enumerate(the_best_dirs):
        merge_data(the_best_dir)
        print("the_best %d/%d..., id is %s" % (i, num_gens, the_best_dir))
    print('Finished the_best data merge...')


if __name__ == "__main__":

    #merge_gen()
    #merge_hof()
    merge_the_best()
