import numpy as np
# Activations 
# author: Ayhan Okuyan
# version: 1.0
# created: 02.12.2020
# prepared for Dense Layers only, can be extended in the future. 
class Activation(object):
    
    def forward(self,x):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
    
    def __str__(self):
        raise NotImplementedError


class Relu(Activation):
    def __init__(self):
        self.name = 'relu'
        self.xCache = None
        

    def forward(self,x):
        self.xCache = np.maximum(0,x)
        return self.xCache
    
    def backward(self):
        dx = np.array(self.xCache, copy=True)
        dx[self.xCache > 0] = 1
        dx[self.xCache <= 0] = 0
        return dx
    
    def __str__(self):
        return self.name
    
        
class Sigmoid(Activation):
    def __init__(self):
        self.name = "sigmoid"
        self.xCache = None

    def forward(self,x):
        expx = np.exp(x)
        self.xCache = expx/(expx+1)
        return self.xCache
    
    def backward(self):
        dx =  self.xCache*(1-self.xCache)
        return dx
    
    def __str__(self):
        return self.name
     
class Tanh(Activation):
    def __init__(self):
        self.name = "tanh"
        self.xCache = None

    def forward(self,x):
        self.xCache = np.tanh(x)
        return self.xCache
    
    def backward(self):
        dx =  (1 - np.power(self.xCache,2))
        return dx
    
    def __str__(self):
        return self.name
       
class Linear(Activation):
    def __init__(self):
        self.name = "linear"
        self.xCache = None
        
    def forward(self,x):
        self.xCache = x
        return self.xCache
        
    def backward(self):
        return np.ones(self.xCache.shape)
    
    def __str__(self):
        return self.name
    
class Softmax(Activation):
    def __init__(self):
        self.name = "softmax"
        self.xCache = None
    
    def forward(self,x):
        expx = np.exp(x - np.max(x))
        self.xCache = expx/np.sum(expx, axis=0)
        return self.xCache
        
    def backward(self):
        dx =  self.xCache*(1-self.xCache)
        return dx
    
    def __str__(self):
        return self.name
    
class Lrelu(Activation):
    def __init__(self, alpha):
        self.name = "lrelu"
        self.alpha = alpha
        self.xCache = None
    
    def forward(self,x):
        self.xCache = np.maximum(self.alpha*x,x)
        return self.xCache
    
    def backward(self):
        dx = np.array(self.xCache, copy=True)
        dx[self.xCache > 0] = 1
        dx[self.xCache <= 0] = self.alpha
        return dx
    
    def __str__(self):
        return self.name + "(alpha=" + str(self.alpha) + ")"