import random
import os, copy, pickle, time
import itertools
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import utils
import gpu_utils as gu
import data_utils as du

def get_binary_mnist(y1=0, y2=1, apply_padding=True, repeat_channels=True):
    
    def _make_cifar_compatible(X):
        if apply_padding: X = np.stack([np.pad(X[i][0], 2)[None,:] for i in range(len(X))]) # pad
        if repeat_channels: X = np.repeat(X, 3, axis=1) # add channels
        return X

    binarize = lambda X,Y: du.get_binary_datasets(X, Y, y1=y1, y2=y2)
    
    tr_dl, te_dl = du.get_mnist_dl(normalize=False)
    # Xtr, Ytr = binarize(*utils.extract_numpy_from_loader(tr_dl))
    # Xte, Yte = binarize(*utils.extract_numpy_from_loader(te_dl))
    Xtr, Ytr = utils.extract_numpy_from_loader(tr_dl)
    Xte, Yte = utils.extract_numpy_from_loader(te_dl)
    Xtr, Xte = map(_make_cifar_compatible, [Xtr, Xte])
    return (Xtr, Ytr), (Xte, Yte)

def get_binary_cifar(y1=3, y2=5, c={0,1,2,3,4}, use_cifar10=True):
    # binarize = lambda X,Y: du.get_binary_datasets(X, Y, y1=y1, y2=y2)
    # binary = False if y1 is not None and y2 is not None else True
    # if binary: print ("grouping cifar classes")
    binary = False
    tr_dl, te_dl = du.get_cifar_dl(use_cifar10=use_cifar10, shuffle=False, normalize=False, binarize=binary, y0=c)

    # Xtr, Ytr = binarize(*utils.extract_numpy_from_loader(tr_dl))
    # Xte, Yte = binarize(*utils.extract_numpy_from_loader(te_dl))
    Xtr, Ytr = utils.extract_numpy_from_loader(tr_dl)
    Xte, Yte = utils.extract_numpy_from_loader(te_dl)
    return (Xtr, Ytr), (Xte, Yte)

def combine_datasets2(Xm, Ym, Xc, Yc, randomize_order=False, randomize_first_block=False, randomize_second_block=False):
    """combine two datasets"""

    def partition_cifar(X, Y):
        p0 = (Y==0).nonzero()[0]
        p1 = (Y==1).nonzero()[0]
        p2 = (Y==2).nonzero()[0]
        p3 = (Y==3).nonzero()[0]
        p4 = (Y==4).nonzero()[0]
        p5 = (Y==5).nonzero()[0]
        p6 = (Y==6).nonzero()[0]
        p7 = (Y==7).nonzero()[0]
        p8 = (Y==8).nonzero()[0]
        p9 = (Y==9).nonzero()[0]
        return X[p0], X[p1], X[p2], X[p3], X[p4], X[p5], X[p6], X[p7], X[p8], X[p9]

    def partition_mnist(X, Y):
        n = len(Y)
        p = np.random.permutation(n)    
        s = np.split(p, 10)
        return X[s[0]], X[s[1]], X[s[2]], X[s[3]], X[s[4]], X[s[5]], X[s[6]], X[s[7]], X[s[8]], X[s[9]]

    def _combine(X1, X2):
        """concatenate images from two sources"""
        X = []
        for i in range(min(len(X1), len(X2))):
            x1, x2 = X1[i], X2[i]
            # randomize order 
            if randomize_order and random.random() < 0.5:
                x1, x2 = x2, x1
            x = np.concatenate((x1,x2), axis=1)
            X.append(x)
        return np.stack(X)

    Xm0, Xm1, Xm2, Xm3, Xm4, Xm5, Xm6, Xm7, Xm8, Xm9 = partition_mnist(Xm, Ym)
    Xc0, Xc1, Xc2, Xc3, Xc4, Xc5, Xc6, Xc7, Xc8, Xc9 = partition_cifar(Xc, Yc)
    n = min(map(len, [Xm0, Xm1, Xm2, Xm3, Xm4, Xm5, Xm6, Xm7, Xm8, Xm9, Xc0, Xc1, Xc2, Xc3, Xc4, Xc5, Xc6, Xc7, Xc8, Xc9]))
    Xm0, Xm1, Xm2, Xm3, Xm4, Xm5, Xm6, Xm7, Xm8, Xm9, Xc0, Xc1, Xc2, Xc3, Xc4, Xc5, Xc6, Xc7, Xc8, Xc9 = map(lambda Z: Z[:n], [Xm0, Xm1, Xm2, Xm3, Xm4, Xm5, Xm6, Xm7, Xm8, Xm9, Xc0, Xc1, Xc2, Xc3, Xc4, Xc5, Xc6, Xc7, Xc8, Xc9])

    X0 = _combine(Xm0, Xc0)
    Y0 = np.zeros(len(X0))

    X1 = _combine(Xm1, Xc1)
    Y1 = np.full(len(X1), 1)

    X2 = _combine(Xm2, Xc2)
    Y2 = np.full(len(X2), 2)

    X3 = _combine(Xm3, Xc3)
    Y3 = np.full(len(X3), 3)

    X4 = _combine(Xm4, Xc4)
    Y4 = np.full(len(X4), 4)

    X5 = _combine(Xm5, Xc5)
    Y5 = np.full(len(X5), 5)

    X6 = _combine(Xm6, Xc6)
    Y6 = np.full(len(X6), 6)

    X7 = _combine(Xm7, Xc7)
    Y7 = np.full(len(X7), 7)

    X8 = _combine(Xm8, Xc8)
    Y8 = np.full(len(X8), 8)

    X9 = _combine(Xm9, Xc9)
    Y9 = np.full(len(X9), 9)
    
    X = np.concatenate([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9], axis=0)
    Y = np.concatenate([Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9], axis=0)
    P = np.random.permutation(len(X))
    X, Y = X[P], Y[P]
    return X, Y    


def get_mnist_cifar(mnist_classes=(0,1), cifar_classes=None, c={0,1,2,3,4}, 
                    randomize_mnist=False, randomize_cifar=False):  
        
    y1, y2 = mnist_classes
    (Xtrm, Ytrm), (Xtem, Ytem) = get_binary_mnist(y1=y1, y2=y2)
    
    y1, y2 = (None, None) if cifar_classes is None else cifar_classes
    (Xtrc, Ytrc), (Xtec, Ytec) = get_binary_cifar(c=c, y1=y1, y2=y2)
    
    Xtr, Ytr = combine_datasets2(Xtrm, Ytrm, Xtrc, Ytrc, randomize_first_block=randomize_mnist, randomize_second_block=randomize_cifar)
    Xte, Yte = combine_datasets2(Xtem, Ytem, Xtec, Ytec, randomize_first_block=randomize_mnist, randomize_second_block=randomize_cifar)

    print("Xtr : ", Xtr.shape)
    print("Xte : ", Xte.shape)
    print("Ytr : ", Ytr.shape)
    print("Yte : ", Yte.shape)
    return (Xtr, Ytr), (Xte, Yte)

def get_mnist_cifar_dl(mnist_classes=(0,1), cifar_classes=None, c={0,1,2,3,4}, bs=256, 
                       randomize_mnist=False, randomize_cifar=False):
    (Xtr, Ytr), (Xte, Yte) = get_mnist_cifar(mnist_classes=mnist_classes, cifar_classes=cifar_classes, 
                                             c=c, randomize_mnist=randomize_mnist, randomize_cifar=randomize_cifar)
    tr_dl = utils._to_dl(Xtr, Ytr, bs=bs, shuffle=True)
    te_dl = utils._to_dl(Xte, Yte, bs=bs, shuffle=False)
    return tr_dl, te_dl

# if __name__ == '__main__':
#     mnist_classes = (0, 1)
#     cifar_classes = (1, 9)
#     batch_size = 256

#     trm_dl, tem_dl = get_mnist_cifar_dl(mnist_classes=mnist_classes, cifar_classes=cifar_classes, bs=batch_size, 
#                                              randomize_mnist=True, randomize_cifar=False)    