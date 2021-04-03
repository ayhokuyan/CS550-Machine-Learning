import numpy as np
# Losses 
# author: Ayhan Okuyan
# version: 1.0
# created: 02.12.2020
class Loss(object):
    def __call__(self, y, pred):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
    
class MSE(Loss):
    def __init__(self):
        self.dif = 0
        self.samples = 0
    
    def __call__(self, y, pred):
        self.dif = y - pred
        self.samples = y.shape[0]
        #print(self.samples)
        loss = np.sum(self.dif**2, axis=0) / self.samples
        if loss.ndim <= 1:
            loss = loss[:, np.newaxis]
        return loss
    
    def backward(self):
        dL = - self.dif / self.samples
        if dL.ndim <= 1:
            dL = dL[:, np.newaxis]
        return dL

class MAE(Loss):
    def __init__(self):
        self.dif = 0
        self.samples = 0
    
    def __call__(self, y, pred):
        self.dif = y - pred
        self.samples = y.shape[0]
        loss = np.sum(np.abs(self.dif), axis=0) / (self.samples)
        if loss.ndim <= 1:
            loss = loss[:, np.newaxis]
        return loss
    
    def backward(self):
        dL = np.array(self.dif, copy=True) 
        dL[dL > 0] = -1
        dL[dL <= 0] = 1
        dL = dL #/ self.samples
        if dL.ndim <= 1:
            dL = dL[:, np.newaxis]
        return dL

#add MAE and MSE log in the future and cross-entropy, categorical cross-entropy fot classification
    
    