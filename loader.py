import torchvision 
import torch 
import numpy as np 
from tqdm import tqdm

# torch.manual_seed(42)

# Download CIFAR 10 Dataset

def load_mnist():
    """
    Load MNIST dataset using torchvision
    """

    print('Loading train data...')
    train_data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('mnist/', 
                                    train=True, 
                                    download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor()
                                    ])),
            shuffle=True,)

    train_input = []
    train_label = []
    
    cnt = 0
    for batch, label in tqdm(train_data):
        train_input.append(batch.squeeze().numpy().reshape(784,))
        train_label.append(label.numpy())
        cnt += 1
        if cnt == 100: break

    print('Loading test data...')
    test_data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('mnist/', 
                                    train=False, 
                                    download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor()
                                    ])),
            shuffle=True,)

    test_input = []
    test_label = []
    
    for batch, label in tqdm(test_data):
        test_input.append(batch.squeeze().numpy().reshape(784,))
        test_label.append(label.numpy())

    return np.array(train_input), np.array(train_label), np.array(test_input), np.array(test_label)

def load_cifar10():
    """
    Load CIFAR-10 dataset using torchvision
    """

    print('Loading train data...')
    train_data = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('cifar/', 
                                    train=True, 
                                    download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor()
                                    ])),
            shuffle=True,)

    train_input = []
    train_label = []
    
    cnt = 0
    for batch, label in tqdm(train_data):
        train_input.append(batch.squeeze().numpy().reshape(3072,))
        train_label.append(label.numpy())
        cnt += 1
        if cnt == 2800: break

    print('Loading test data...')
    test_data = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('cifar/', 
                                    train=False, 
                                    download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor()
                                    ])),
            shuffle=True,)

    test_input = []
    test_label = []
    
    for batch, label in tqdm(test_data):
        test_input.append(batch.squeeze().numpy().reshape(3072,))
        test_label.append(label.numpy())

    return np.array(train_input), np.array(train_label), np.array(test_input), np.array(test_label)


if __name__ == '__main__':
    load_cifar10()
    train_data = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('cifar/', 
                                   train=True, 
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()
                                   ])),
        batch_size=4,
        shuffle=True,)

    for batch, label in train_data:
        print(batch[0][0].shape)
        print(label[0])
        input()
