## Implementation Details for Capstone Project:

### Dependencies
* Eigen

### Execution 
```
git clone https://github.com/kubershahi/ashoka-capstone.git
make nn         // for Neural Network
make bb         // for SecureNN building blocks
```

### Scratch Neural Network Implementation:
* Neural Network with ReLU and Softmax 
* Total of 2 Layers: one hidden layer with 256 neurons and one output layer with 10 neurons for MNIST dataset

### SecureNN Building Blocks
* Mapping and Reverse Mapping
* Truncation
* Secret Sharing
* Matrix Multiplication (Beaver's Triplet)
* Private Compare 

SecureNN Paper: https://eprint.iacr.org/2018/442.pdf
