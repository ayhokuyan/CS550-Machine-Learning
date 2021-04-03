import numpy as np 
import backend.activations as activ
import backend.initializers as init

# Layers 
# author: Ayhan Okuyan
# version: 1.0
# created: 03.12.2020
# prepared for Dense Layer only, can be extended in the future. 

class Dense:
    def __init__(self, inDim, outDim, 
                 activation=None, 
                 kernel_initializer=init.RandomUniform(), 
                 bias_initializer=init.Zeros()):
        self.inDim = inDim
        self.outDim = outDim
        
        if activation is None:
            self.activation = activ.Sigmoid()
        else: 
            self.activation = activation
        
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        
        #w(in x out) b(1 x out) wE(in+1 * out)
        self.weights = self.kernel_initializer(shape=(self.inDim, self.outDim))
        self.biases = self.bias_initializer(shape=(1, self.outDim))
        self.wE = np.vstack((self.weights, self.biases))
        
        #backprop
        self.delta = None
        
        #for optimizer parameters and loss object
        self.output = False
        self.params = None
    
                    
    def forward(self, x):
        xTilde = np.hstack( (x, np.ones((x.shape[0],1))) )

        y = self.activation.forward(xTilde @ self.wE)
        if(y.ndim <= 1):
            y = y[:, np.newaxis]
        return y
    

    def backward(self, deltaNext, nextWE):
        #for hidden and input layers. 
        #Find the diagonal matrix of the backward activations.
        #print(self.activation.backward().shape)
        #print(np.sum(self.activation.backward(), axis=1)[:,np.newaxis].shape)
        derMatrix = np.diag(np.sum(self.activation.backward(), axis=0)[:,np.newaxis].T[0])
        #print(derMatrix.shape)
        nextW = nextWE[:-1,:]
        #print(derMatrix.shape, nextW.shape, deltaNext.shape)
        self.delta = (derMatrix @ nextW @ deltaNext.T).T
       
    def __repr__(self):
        return "Input Dim: " + str(self.inDim) +\
            ", Number of Neurons: " + str(self.outDim) +\
                " Activation: " + str(self.activation)
    
        