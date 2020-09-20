import mixup, solver, loader
from tqdm import tqdm
import numpy as np



def filter(X, Y, valid_labels=[0,1]):
    mask = np.isin(Y, valid_labels).squeeze(1)
    return X[mask], Y[mask]

train_data, train_label, test_data, test_label = loader.load_cifar10()

train_data, train_label = filter(train_data, train_label)

test_data, test_label = filter(test_data, test_label)  # no need total 10000 

# X, Y = train_data, train_label
# X, Y = mixup.mixup_shuffle(train_data, train_label, mix=True,size=128) 

X, Y = mixup.mixup_shuffle(train_data, train_label, mix=False, size=500)

# CIFAR: 5e-2, 1e-3，100
# MNIST: 1e-2, 1e-4, 64

for lr, C,  in [(1e-1, 1e-3)]:
    print('='*81)  # a nice bar
    print('lr is %.3e, C is %.3e' %(lr, C))

    alphas = solver.SGD(X, Y, Hidden=250, lr=lr, C=C, epochs=400, batch_size=64, mix=False) 
    # print(alphas)
    # input()
    print('Start testing...')
    right = 0
    cnt = 0

    d=len(X[0])
    Hidden=250
    np.random.seed(42)
    W = np.random.randn(Hidden, d) * np.sqrt(2./(Hidden + d))

    for td, tl in tqdm(zip(test_data, test_label)):
        if tl in [0, 1]:
            if solver.predict(td, alphas, W) == tl[0]:
                right+=1
            cnt += 1
        
    print('\t --- No-mixup total accuracy: %.3f %%'%(right/cnt*100.,))

    alphas = solver.SGD(X, Y, Hidden=250, lr=lr, C=C, epochs=400, batch_size=64, mix=True) 
    # print(alphas)
    # input()
    print('Start testing...')
    right = 0
    cnt = 0
    d=len(X[0])
    Hidden=250
    np.random.seed(42)
    W = np.random.randn(Hidden, d) * np.sqrt(2./(Hidden + d))

    for td, tl in tqdm(zip(test_data, test_label)):
        if tl in [0, 1]:
            if solver.predict(td, alphas, W) == tl[0]:
                right+=1
            cnt += 1
        
    print('\t --- Mixup total accuracy: %.3f %%'%(right/cnt*100.,))
        