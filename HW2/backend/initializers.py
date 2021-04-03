import numpy as np #
from scipy.stats import truncnorm

# Initializations 
# author: Ayhan Okuyan
# version: 1.0
# created: 02.12.2020
# prepared for Dense Layers only, can be extended in the future. 

class Initializer(object):
    def __call__(self, shape, dtype=None, **kwargs):
        raise NotImplementedError
    
class Random(Initializer):
    def __init__(self,alpha=0.001, seed=None):
        self.alpha = alpha
        self.seed = seed
        
    def __call__(self, shape, dtype=None, **kwargs):
        np.random.seed(self.seed)
        return np.random.random(size=shape) * self.alpha 


class RandomUniform(Initializer):
    def __init__(self,minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
    def __call__(self, shape, dtype=None, **kwargs):
        np.random.seed(self.seed)
        return np.random.uniform(low=self.minval, high=self.maxval, size=shape)

class TruncatedNormal(Initializer):
    def __init__(self,mean=0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
    def __call__(self, shape, dtype=None, **kwargs):
        np.random.seed(self.seed)
        a = self.mean - 2 * self.stddev
        b = self.mean + 2 * self.stddev
        out = truncnorm.rvs(a,b, loc=self.mean, scale=self.stddev, size=shape)
        return out
        
class Zeros(Initializer):
    def __init__(self):
        pass

    def __call__(self, shape, dtype=None, **kwargs):
        return np.zeros(shape)
    
class Ones(Initializer):
    def __init__(self):
        pass
    def __call__(self, shape, dtype=None, **kwargs):
        return np.ones(shape)
    
class Constant(Initializer):
    def __init__(self, value):
        self.value = value
        
    def __call__(self, shape, dtype=None, **kwargs):
        return np.ones(shape) * self.value

class GlorotUniform(Initializer):
    def __init__(self):
        pass
    
    def __call__(self, shape, dtype=None, **kwargs):
        fan_in = shape[0]
        fan_out = shape[1]
        
        return np.random.uniform(low=-np.sqrt(2/(fan_in + fan_out)),high= np.sqrt(2/(fan_in+fan_out)), size=shape)
   
class GlorotNormal(Initializer):
    def __init__(self, seed=None):
        self.seed = seed
        
    def __call__(self, shape, dtype=None, **kwargs):
        np.random.seed(self.seed)
        # set fans
        fan_in = shape[0]
        fan_out = shape[1]
        #set mean and stddev
        mean = 0
        stddev = np.sqrt(2/(fan_in+fan_out))
        #get weights
        a = mean - 2 * stddev
        b = mean + 2 * stddev
        out = truncnorm.rvs(a,b, loc=mean, scale=stddev, size=shape)
        return out
    