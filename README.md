# NNforMNIST
A fully connected neural network for MNIST classification, implemented by Python 3.5 and Numpy. 

**Requirements:**
Python 3.5  
Numpy  

**Description:**  
dataloader.py: Dataloader for MNIST dataset  
main.py: Entry point for this project  
network.py: Implementation for the proposed network  
model: Parameters obtained by training process  
TRAIN_DATA.npy: Meta-data created by training process (epoch, iteration, accuracy, loss)

**Training:**  
Open main.py and run train(). The updated weights and parameters will be saved in 'model' folder.

**Testing:**
Open main.py and run test() to obtain the classification accuracy on test set.
You can also run test10RandomImgs() to check 10 random classification results through graphic interface.
