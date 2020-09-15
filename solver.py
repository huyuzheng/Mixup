__doc__ = '''Implement the mathematical details. '''

from random import choices, seed 
import numpy as np
from tqdm import tqdm

def rbf(x, y, sigma=1.): 
    return np.exp(-np.linalg.norm(x-y)**2/sigma**2/100.)

def exponential(x, y, sigma=1.):
    return np.exp(-np.linalg.norm(x-y)/sigma/10)

def inner(x, y):
    return np.dot(x, y) / len(x)**.5

def polynomial(x, y, c, d=2):
    return (np.dot(x, y)+c)**d


@np.vectorize
def sigmoid(x):
    return 1/(1+np.exp(x))


def k_mat(X, kernel_func):
    """
    Assuming that kernel_func is symmetric.  
    """
    n = len(X)
    ret = np.zeros((n, n))
    for i in range(n):
        ret[i][i] = kernel_func(X[i], X[i])
        for j in range(i+1, n):
            ret[i][j] = ret[j][i] = kernel_func(X[i], X[j])

    return ret 

"""
For MNIST, we recommend:
lr=1e-2,
C=1e-4,
batch_size=None,
epoch=20,
"""

def SGD(X, Y, lr=1e-3, C=1e-4, batch_size=None, epochs=20, tolerance=1e-3, kernel_func=None):
    if kernel_func == None:
        kernel_func = rbf

    n = len(X)
    if not batch_size:
        batch_size = n 

    indices = list(range(n)) 
    # np.random.seed(42)
    # alpha = np.random.random(n) 
    alpha = np.random.randn(n) / 10.
    norm = []
    K = k_mat(X, kernel_func)
    losses = [10000] 

    print('Start training...')
    for epoch in tqdm(range(epochs)):
        batch_idx = choices(indices, k=batch_size)  # randomly choose a mini-batch
        
        K_rest = k_mat(X[np.ix_(batch_idx)], kernel_func)  # consider a restricted matrix 

        y_rest = Y[np.ix_(batch_idx)]

        alpha_rest = alpha[np.ix_(batch_idx)]
        prev_alpha = alpha.copy()

        """
        print(-(np.dot(y_rest.squeeze(), np.log(sigmoid(alpha_rest@K_rest)))+np.dot(1-y_rest.squeeze(), np.log(1-sigmoid(alpha_rest@K_rest)))))
        print(alpha_rest)
        print((sigmoid(alpha_rest@K_rest)[:,None] - y_rest).squeeze(1))
        print((K_rest@(sigmoid(alpha_rest@K_rest)[:,None] - y_rest)).squeeze(1))
        print((lr * K_rest@(sigmoid(alpha_rest@K_rest)[:,None] - y_rest)).squeeze(1)   + C*alpha_rest)
        input()
        """

        alpha[np.ix_(batch_idx)] -= -(lr * K_rest@(sigmoid(alpha_rest@K_rest)[:,None] - y_rest)).squeeze(1)   + C*alpha_rest # update alpha
        """
        alpha_rest = alpha[np.ix_(batch_idx)]

        print(-(np.dot(y_rest.squeeze(), np.log(sigmoid(alpha_rest@K_rest)))+np.dot(1-y_rest.squeeze(), np.log(1-sigmoid(alpha_rest@K_rest)))))
        """
        new_loss = -((np.dot(Y.squeeze(), np.log(sigmoid(alpha@K)))+np.dot(1-Y.squeeze(), np.log(1-sigmoid(alpha@K))))/n)

        if new_loss > losses[-1]: 
            lr /= 2.   # adaptively scale down learning rate
            alpha = prev_alpha
            continue 

        losses.append(new_loss)
        norm.append(np.linalg.norm(alpha))
        
        """        
        right = 0
        cnt = 0
        for td, tl in tqdm(zip(X, Y)):
            if tl in [0, 1]:
                if predict(X, td, alpha) == tl[0]:
                    right+=1
                cnt += 1
        
        print('Total Accuracy: %.3f %%'%(right/cnt*100.,))
        print('lr is %.1e'%lr)
        """
        
        # if losses[-2]-losses[-1] < tolerance:
        #     break  # early stop
  

    print('alpha.T K alpha is %.3f' % float(alpha[None,:]@K@alpha[:,None]))

    import matplotlib.pyplot as plt
    plt.plot(np.array(losses[1:]))
    plt.show()
    
    return alpha 


def predict(X, x, alpha, labels=None, kernel_func=None):
    """
    single image predict
    """
    if kernel_func == None:
        kernel_func = rbf
    if labels == None:
        labels = [0, 1]
    
    # print(sigmoid(alpha@np.array([kernel_func(xx, x) for xx in X])))
    if sigmoid(alpha@np.array([kernel_func(xx, x) for xx in X])) < 0.5:
        return 0 
    else:
        return 1
