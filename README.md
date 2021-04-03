# CS550 Machine Learning
The assignment and project implementations for CS550 Machine Learning course, Bilkent University.

## Coverage of the repository

### HW1
Decision Trees 
- Used sklearn to train an decision tree classifier to classify thyroid disease patients. 
- Implemented a decision tree classifier to build a decision tree with preprunning.  
- Extended the implementation for a Cost-Sensitive decision tree. 
- Implemented in Python (Jupyter Notebook)

### HW2
Linear Regression and Neural Networks
- Trained a neural network and a linear regressor and optimize to learn the correlation between the provided input output sets. Implemented a general purpose neural network framework under **backend**, however, this implementation is not completely accurate. I wouldn't recommend anyone to clone this repo and use that backend. The future work consists of debugging.
- Backend is implemented in Python, and the experiments are done in Python using Jupiter Notebooks

### HW3
Clustering 
- Implemented k-Means clustering algorithm to cluster the image pixels in an image. 
- Implemented Hierarchical Agglomerative Clustering (HAC) to cluster the pixels. Since this is a expensive algorithm in terms of memory and time, we have initially clustered the pixels with k-Means, then used HAC to cluster the rest. 
- Implemented in Python (Jupyter Notebook)

###  Project:
Image Domain Adaptation using Cyclic Generative Adversarial Networks 
- Used Cycle GANs to transform the images from the photograph to cartoon domain. 
- Experimented on Vanilla GAN, LSGAN and Wasserstein GAN architectures. 
- Used [this repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to train the networks. 
- Frechet Inception Distance (FID) implemented to measure performance. 

