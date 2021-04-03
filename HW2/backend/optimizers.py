import numpy as np #
# Optimizers 
# author: Ayhan Okuyan
# version: 1.0
# created: 03.12.2020
# prepared for Dense Layers only, can be extended in the future. 

#kwargs accepts the parameters put in the layer 
class Optimizer(object):
    def step(self, weights, grad, **kwargs):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self,lr=0.001):
        self.lr = lr
    
    def step(self, weights, grad, **kwargs):
        weights -= (self.lr * grad)
        
    def paramDict(self):
        paramDict = dict()
        return paramDict
        
class Momentum(Optimizer):
    def __init__(self,lr=0.001, momentum=0.9, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
    
    def step(self, weights, grad, **kwargs):
        prevUpd = kwargs['prevUpdate']
        
        if self.nesterov:
            upd = self.momentum * prevUpd - self.lr * grad
            weights += self.momentum * upd - self.lr * grad
        else:
            upd = self.momentum * prevUpd - self.lr * grad
            weights += upd  
        
        kwargs['prevUpdate'] = upd
        
    def paramDict(self):
        paramDict = dict()
        paramDict['prevUpdate'] = 0
        return paramDict
    
class MomentumSlides(Optimizer):
    def __init__(self,lr, alpha=0.9):
        self.lr = lr
        self.alpha = alpha
    
    def step(self, weights, grad, **kwargs):
        prevUpd = kwargs['prevUpdate']
        
        upd =  self.alpha * grad + (1-self.alpha) * prevUpd
        weights -= self.lr * upd
        
        kwargs['prevUpdate'] = upd
        
    def paramDict(self):
        paramDict = dict()
        paramDict['prevUpdate'] = 0
        return paramDict
        

class Adam(Optimizer):
    def __init__(self,lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
    def step(self, weights, grad, **kwargs):
        prevM = kwargs['prevM']
        prevV = kwargs['prevV']
        epoch = kwargs['epoch']
        
        M = (1-self.beta1)* grad + self.beta1 * prevM
        V = ((1-self.beta2)* grad**2) + self.beta2* prevV
        alpha_t = self.lr * np.sqrt(1-np.power(self.beta2,epoch))/(1-np.power(self.beta1,epoch))
        weights -= alpha_t * M / (np.sqrt(V) + self.eps)
        
        kwargs['prevM'] = M
        kwargs['prevV'] = V
    
    def paramDict(self):
        paramDict = dict()
        paramDict['prevM'] = 0
        paramDict['prevV'] = 0
        paramDict['epoch'] = 0
        return paramDict
        
        
