import mixup, solver, loader
from tqdm import tqdm
train_data, train_label, test_data, test_label = loader.load_cifar10()
X, Y = mixup.mixup_shuffle(train_data, train_label, mix=True,size=128)

Xn, Yn = mixup.mixup_shuffle(train_data, train_label, mix=False,size=128)
# CIFAR: 5e-2, 1e-3，100
# MNIST: 1e-2, 1e-4, 64

for lr, C,  in [(5e-2, 1e-3)]:
    print('='*81)  # a nice bar
    print('lr is %.3e, C is %.3e' %(lr, C))

    alphas = solver.SGD(X, Y, lr=lr, C=C, epochs=40, batch_size=128) 
    # print(alphas)
    # input()
    print('Start testing...')
    right = 0
    cnt = 0
    for td, tl in tqdm(zip(test_data, test_label)):
        if tl in [0, 1]:
            if solver.predict(X, td, alphas) == tl[0]:
                right+=1
            cnt += 1
        
    print('\t --- Mixup total accuracy: %.3f %%'%(right/cnt*100.,))

    alphas = solver.SGD(Xn, Yn, lr=lr, C=C, epochs=10, batch_size=128) 
    # print(alphas)
    # input()
    print('Start testing...')
    right = 0
    cnt = 0
    for td, tl in tqdm(zip(test_data, test_label)):
        if tl in [0, 1]:
            if solver.predict(Xn, td, alphas) == tl[0]:
                right+=1
            cnt += 1
        
    print('\t --- No-mixup total accuracy: %.3f %%'%(right/cnt*100.,))
        