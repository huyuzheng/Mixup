__doc__ = '''Implement the mathematical details. '''

from random import choices, seed 
import numpy as np
from tqdm import tqdm
import mixup


# Some choices of activation functions

def Activation_ReLU(x):
    return np.maximum(0, x)

def Activation_Cos(x):
    return np.cos(x)

# Model: Random Feature(RF), f(x) = a^T \sigma(Wx), where "W" is fixed after initialization and we only train "a" using SGD.
# a: R^{Hidden*1}, W: R^{Hidden*d}, x: R^{d*1}

def Random_Feature(alpha, x, W_mat):
    # b = np.random.rand(Hidden, 1) * 2 * np.pi    
    return alpha@Activation_ReLU(W_mat@x).squeeze()

@np.vectorize
def sigmoid(x):
    return 1/(1+np.exp(x))



"""
For MNIST, we recommend:
lr=1e-2,
C=1e-4,
batch_size=None,
epoch=20,
"""



def Update_Parameters(X, Y, alpha, C, lr, Hidden, W_mat):
    b = len(X)
    Gradient = np.zeros(Hidden)
    for i in range(b):
        Gradient += (Y[i] - sigmoid(Random_Feature(alpha, X[i], W_mat)))*np.array([Activation_ReLU(W_mat[j]@X[i]) for j in range(Hidden)])


    return alpha - lr*Gradient/b + C*alpha

def SGD(X, Y, Hidden, lr=1e-2, C=1e-4, batch_size=64, epochs=20, mix=False):

    d = len(X)

    if not batch_size:
        batch_size = d 

    indices = list(range(d)) 
    # np.random.seed(42)
    # alpha = np.random.random(n) 
    alpha = np.random.randn(Hidden) * np.sqrt(2./(1 + Hidden)) 
    losses = [10000] 

    # never calc a fixed object twice, as long as memory permits
    
    # Xavier Initialization
    
    np.random.seed(42)
    W = np.random.randn(Hidden, len(X[0])) * np.sqrt(2./(Hidden + len(X[0])))

    # make it random again
    np.random.seed()
    Qlist = [i*400//8 for i in range(1, 8)]
    print('Start training...')
    if mix == False:
        for epoch in tqdm(range(epochs)):
            batch_idx = choices(indices, k=batch_size)  # randomly choose a mini-batch
            X_rest = X[np.ix_(batch_idx)]
            Y_rest = Y[np.ix_(batch_idx)]

            prev_alpha = alpha.copy()
            alpha = Update_Parameters(X_rest, Y_rest, alpha, C, lr, Hidden, W)
            Prediction = np.array([Random_Feature(alpha, X[i], W) for i in range(d)])
            new_loss = -((np.dot(Y.squeeze(), np.log(sigmoid(Prediction)))+np.dot(np.ones(d)-Y.squeeze(), np.log(1.-sigmoid(Prediction))))/d)
            
            """
            if new_loss > losses[-1]: 
                lr /= 2.   # adaptively scale down learning rate
                alpha = prev_alpha
                continue 
            """
            
            if epoch == epochs//2:
                lr /= 10
            elif epoch == epochs*3//4:
                lr /= 10
            elif epoch == epochs*7//8:
                lr /= 10
            """
            if epoch in Qlist:
                lr /= 2
            """
            losses.append(new_loss)
            # norm.append(np.linalg.norm(alpha))

        import matplotlib.pyplot as plt
        plt.plot(np.array(losses[1:]))
        plt.show()

    if mix == True:
        for epoch in tqdm(range(epochs)):
            batch_idx = choices(indices, k=batch_size)  # randomly choose a mini-batch
            X_rest = X[np.ix_(batch_idx)]
            y_rest = Y[np.ix_(batch_idx)]
            X_mix, Y_mix = mixup.mixup_shuffle(X_rest, y_rest, mix=True, size=batch_size)
            for i in range(len(X_mix)):
                Xi, Yi = np.array([X_mix[i]]), np.array([Y_mix[i]])
                alpha = Update_Parameters(Xi, Yi, alpha, C, lr, Hidden, W)
            """
            if epoch == epochs//2:
                lr /= 10
            elif epoch == epochs*3//4:
                lr /= 10
            elif epoch == epochs*7//8:
                lr /= 10
            """
            if epoch in Qlist:
                lr /= 2
            



    # print('alpha.T K alpha is %.3f' % float(alpha[None,:]@K@alpha[:,None]))

    return alpha 



def predict(x, alpha, W, labels=None):
    """
    single image predict
    """
    if labels == None:
        labels = [0, 1]
    

    # print(sigmoid(alpha@np.array([kernel_func(xx, x) for xx in X])))
    if sigmoid(Random_Feature(alpha, x, W)) < 0.5:
        return 0 
    else:
        return 1
