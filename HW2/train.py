import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import backend.ann as nn 
import backend.layers as lyr
import backend.activations as activ
import backend.optimizers as optim
import backend.losses as losses
import backend.initializers as init

#read the two training datasets and compare and compare the analysis
train1 = pd.read_csv('train1.txt', delimiter='\t', names=['x','y'])
train1.reset_index(drop=True, inplace=True)
train1 = train1.to_numpy()
print(train1.shape)
train1x = train1[:,0,np.newaxis]
train1y = train1[:,1,np.newaxis]
print(train1x.shape, train1y.shape)


model = nn.ANNRegressor()
layers = []
layers.append(lyr.Dense(train1x.shape[1], 15, activation=activ.Tanh(), 
                        kernel_initializer=init.GlorotNormal(),
                         bias_initializer=init.Zeros()))
layers.append(lyr.Dense(15, 1, activation=activ.Linear(),
                        kernel_initializer=init.GlorotNormal(),
                        bias_initializer=init.Zeros()))
model(layers)

print(model)

opt = optim.Adam(lr = 0.01)
loss = losses.MSE()
metrics = ['train_loss']

model.compile(optimizer=opt, loss=loss, metrics=metrics)

print('Model Compiled')

for lyr in model.layers:
    print(lyr.params)

hist = model.train(train1x, train1y, epochs=10000, batch_size=15, verbose=True)

plt.plot(hist['train_loss'])
plt.title('Loss for Training 1')
plt.show()
plt.scatter(train1x, train1y)
plt.scatter(train1x, model.forward(train1x))
plt.title('Training Set 1')
plt.show()

#test
plt.scatter(test1xNor, test1yNor)
plt.scatter(test1xNor, forward(test1xNor, we1,we2,act1,act2))
plt.title('Test Set 1')
plt.show()

