import numpy as np #
import backend.optimizers as optim
import backend.layers
import backend.losses as losses
import backend.batch_generator as bgen
# the ANN Regressor Model 
# author: Ayhan Okuyan
# version: 1.01
# created: 04.12.2020
# Can be converted to classification with the added loss and the proper predict function, on top of softmax activation

class ANNRegressor():
    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.epoch = 0
        self.batchSize = 1
        
    def add(self, layer):
        self.layers.append(layer)
        
    def __call__(self, lyrList):
        #list of layer objects to add 
        for layer in lyrList:
            self.layers.append(layer)
              
    def forward(self, x):
        y = x
        for lyr in self.layers:
            y = lyr.forward(y)
        return y
    
    def compile(self, optimizer=optim.SGD(), loss=losses.MSE(), metrics=[]):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        for i,layer in enumerate(self.layers):
            #set the parameters for updates for each optimizer.
            layer.params = self.optimizer.paramDict().copy()
        
        
    def backward(self,x):
        for i in reversed(range(len(self.layers))):
            lyr = self.layers[i]
            #outputLayer
            if(lyr == self.layers[-1]):
                error = self.loss.backward()
                derMatrix = np.diag(lyr.activation.backward().T[0])
                lyr.delta = derMatrix @ error
            #hiddenLayer
            else:
                lyr.backward(self.layers[i+1].delta, self.layers[i+1].wE)
        
        for i,lyr in enumerate(self.layers):
            if(x.ndim <= 1):
                x = x[:, np.newaxis]
                
            if(i == 0):
                xTilde = np.hstack( (x, np.ones((x.shape[0],1))) )
            else:
                xTilde = self.layers[i-1].activation.xCache 
                xTilde = np.hstack((xTilde, np.ones((xTilde.shape[0],1)) )) 
   
            grad = xTilde.T @ lyr.delta
           
            #update epoch number if exists in the optimizer requirements like adam.
            if 'epoch' in lyr.params:
                lyr.params['epoch'] = self.epoch
            
            
            self.optimizer.step(lyr.wE, grad, **lyr.params)
            
            
    def train(self, x, y, epochs, batch_size = 1, verbose=True, stopping_loss=None, shuffle=True):
        metricDict = dict()
        for metric in self.metrics:
            metricDict[metric] = list()
            
        generator = bgen.BatchGenerator(x,y,batch_size, shuffle=shuffle)
        
        for epoch in range(1,epochs+1):
            self.epoch = epoch
            for i in range(int(generator.batchnum)):
                xbatch, ybatch, mult = generator.getBatch()
                outbatch = self.forward(xbatch)
                self.loss(ybatch, outbatch) #* mult
                self.backward(outbatch)
                            
            train_loss = self.loss(self.forward(x),y)

            if 'train_loss' in self.metrics:
                train_loss = self.loss(self.forward(x),y)
                metricDict['train_loss'].append(train_loss[0][0])
                if verbose:
                    print('----------------' + 'epoch ' + str(epoch) + '----------------')
                    printStr = ''
                    for key,value in metricDict.items():
                        printStr += key + ':' + str(value[-1]) + ', '
                    print(printStr[:-2])
                    
            if stopping_loss is not None and train_loss <= stopping_loss:
                print('Epoch: %d' % epoch)
                print('Train loss: %.3f' % train_loss)
                print('Stopping criterion reached.')
                break;
                
                
                
        
        return metricDict
    
    def __repr__(self):
        retStr = "\n"
        for i, lyr in enumerate(self.layers):
            retStr += "Layer " + str(i) + ": " + lyr.__repr__() + "\n"
        return retStr