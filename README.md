# NNforMNIST
A fully connected neural network for MNIST classification, implemented by Python 3.5 and Numpy.   
Author: Mingqi Gao, Chongqing University  
Email: gaomingqi@cqu.edu.cn, im.mingqi@gmail.com  

### Requirements: 
- Python 3.5  
- Numpy  

### Description:
`dataloader.py`: Dataloader for MNIST dataset  
`main.py`: Entry point for this project  
`network.py`: Implementation for the proposed network  
`model`: Parameters obtained by training process  
`TRAIN_DATA.npy`: Meta-data created by training process (epoch, iteration, accuracy, loss)

### Training:
Uncomment and run `train()` in `main.py`. The updated weights and parameters will be saved in `'model'` folder.

### Testing:
Uncomment and run `test()` in `main.py` to obtain the classification accuracy on test set.  
You can also run `test10RandomImgs()` to check classification results for 10 random images through graphic interface.
