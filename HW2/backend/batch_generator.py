import numpy as np 
# Mini-Batch Generator 
# author: Ayhan Okuyan
# version: 1.0
# created: 03.12.2020
# repared only for numeric data at this point
class BatchGenerator():
    def __init__(self, x, y, batch_size=1, shuffle=False, seed=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        
        #shuffle data
        if self.shuffle:
            np.random.seed(self.seed)
            ind = np.random.permutation(x.shape[0])
            self.x = self.x[ind]
            self.y = self.y[ind]
        
        self.batchnum = np.ceil(x.shape[0] / self.batch_size)
        self.lastbatch_size = x.shape[0] % self.batch_size
      
        self.start = 0
    
    def getBatch(self):
        end = min(self.start + self.batch_size, self.x.shape[0])
        multiplier = 1
        
        if(end == self.x.shape[0]):
            if(self.lastbatch_size != 0):
                multiplier = self.lastbatch_size / self.batch_size 
            
            xbatch = self.x[self.start:end,:]
            ybatch = self.y[self.start:end,:]
            
            #print(self.start, end, multiplier)
            
            self.start = 0
            
            if self.shuffle:
                np.random.seed(self.seed)
                ind = np.random.permutation(self.x.shape[0])
                self.x = self.x[ind]
                self.y = self.y[ind]
        else:
            xbatch = self.x[self.start:end,:]
            ybatch = self.y[self.start:end,:]
            
            #print(self.start, end, multiplier)
            
            self.start = end
        
        
            
        return xbatch, ybatch, multiplier
        
            
        