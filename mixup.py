__doc__ = """
Return mixed-up input images and labels 
"""

import numpy as np 
from random import choices, random, seed, shuffle

def mixup(data, label, size=None, valid_labels=None, alpha=1, mix=True):
    seed(42)
    
    if not size:
        size = len(data)

    if valid_labels == None:
        valid_labels = [0, 1]  # default is MNIST case 
    
    # first, filter data such that they consist of only two classes.
    mask = np.isin(label, valid_labels).squeeze(1)

    data = data[mask]
    label = label[mask]

    indices = list(range(len(data)))  


    print('Data size is %d' % size)

    ret_data_mix = []
    ret_label_mix = []
    ret_data = []
    ret_label = []

    for _ in range(size):
        ida, idb = choices(indices, k=2)
        if mix:
            lmbda = np.random.beta(alpha, alpha)
        else:
            lmbda = 0
        ret_data_mix.append(data[ida]*lmbda + data[idb]*(1-lmbda))
        ret_label_mix.append(label[ida]*lmbda + label[idb]*(1-lmbda))  # label represents the component of '1'
    
    """
    for _ in range(size):
        ida, idb = choices(indices, k=2)
        # lmbda = random()
        lmbda = 0
        ret_data.append(data[ida]*lmbda + data[idb]*(1-lmbda))
        ret_label.append(label[ida]*lmbda + label[idb]*(1-lmbda))  # label represents the component of '1'
    """
    return np.array(ret_data_mix), np.array(ret_label_mix)

def mixup_shuffle(data, label, size=None, valid_labels=None, alpha=1., mix=True):
    # seed(42)
    
    if not size:
        size = len(data)

    if valid_labels == None:
        valid_labels = [0, 1]  # default is MNIST case 
    
    # first, filter data such that they consist of only two classes.
    mask = np.isin(label, valid_labels).squeeze(1)

    data = data[mask]
    label = label[mask]
    
    data = data[:size]
    label = label[:size]
    
    indices = list(range(len(data)))  
    shuffle(indices)

    datab = data[indices]
    labelb = label[indices]
    
    
    ret_data_mix = []
    ret_label_mix = []
    # ret_data = []
    # ret_label = []

    for da, la, db, lb in zip(data, label, datab, labelb):
        if mix:
            lmbda = np.random.beta(alpha, alpha)
        else:
            lmbda = 0
        ret_data_mix.append(da*lmbda + db*(1-lmbda))
        ret_label_mix.append(la*lmbda + lb*(1-lmbda))  # label represents the component of '1'
        if len(ret_data_mix) == size: 
            break 


    print('Data size is %d' % len(ret_data_mix))

    return np.array(ret_data_mix), np.array(ret_label_mix)

if __name__ == '__main__':
    from loader import load_mnist
    data, label, _, __ = load_mnist()
    mixup(data, label)
    