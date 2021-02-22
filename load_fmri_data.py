import numpy as np
import h5py
import os
import copy

'''
def load_h5file(path, keys):
    tmp = []
    hf = h5py.File(path, 'r')
    for key in keys:
        n1 = hf.get(key)
        tmp.append(np.array(n1))
    return tmp
'''

def load_h5file(path, keys=None):
    """
    Load contents of a hdf5 file
    :param path: path to h5 file
    :param keys: keys to load
    :return:
    """
    with h5py.File(path, 'r') as h5file:
        try:
            if keys is None:
                keys = []
                h5file.visit(keys.append)
            return {k: np.array(h5file[k]) for k in keys}
        except KeyError:
            print('Existing keys are: ', h5file.keys())
            raise KeyError
            
            
def normalize_session_data(data, idx):
    mu = np.mean(data[idx, :], axis=0)
    sig = np.std(data[idx, :], axis=0)
    
    return (data - mu) / sig

def get_data(db):
    # Load all grid-search results
    all_results_dict = {}
    for e in db.find_all_entries({}):
        if len(all_results_dict) == 0:
            # initalize results dict
            for k in e.keys():
                all_results_dict[k] = [e[k]]
        else:
            for k in e.keys():
                all_results_dict[k].append(e[k])
    return all_results_dict

def load_neural_data(root_path, mask_names=None):
    normalizer_idx = {'stimgroup1':[0, 4, 99, 14, 20, 23, 33, 40, 43, 45, 49, 76, 78, 83, 86],
                      'stimgroup2': list(range(15))}
    if mask_names is None: 
        mask_names = ['leba', 'reba', 'lffa', 'rffa', 'lofa', 'rofa', 'ropa', 'lppa', 'rppa'] # 'lopa'

    # read data 
    neural_data_dict = {}
    for f in ['stimgroup1_data.h5', 'stimgroup2_data.h5']:
        g_name = f.split('_data')[0]
        tmp = load_h5file(os.path.join(root_path, f), 
                                       keys=['data/lhdata', 'data/rhdata', 
                                             # 'data/lhreliability', 'data/rhreliability', 'data'
                                            ])
        for k in tmp:
            tmp[k] = tmp[k].T

        neural_data_dict[g_name] = copy.deepcopy(tmp)

    # pool data 
    pooled_data = {}
    for k in ['data/lhdata', 'data/rhdata']: 
        g1_norm = normalize_session_data(neural_data_dict['stimgroup1'][k], normalizer_idx['stimgroup1'])
        g2_norm = normalize_session_data(neural_data_dict['stimgroup2'][k], normalizer_idx['stimgroup2'])[15:]
        pooled_data[k] = np.concatenate((g1_norm, g2_norm), axis=0)
        
    # mask data
    masks = load_h5file(os.path.join(root_path, 'handmade_rel_mask.h5'))
    for k in masks:
        if k not in ['data']:
            masks[k] = np.array(np.squeeze(masks[k] - 1), dtype=np.int)

    output = dict()
    for m in mask_names: 
        mask_label = 'data/'+m
        hem = mask_label.split('data/')[-1][0]
        DATA_SUBSET = f'data/{hem}hdata'

        Y_idx = masks[mask_label][np.isfinite(pooled_data[DATA_SUBSET][:, masks[mask_label]].mean(0))]
        output[m] = pooled_data[DATA_SUBSET][:, Y_idx] #.mean(axis=1, keepdims=True)
        
    return output, masks
