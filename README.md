# **Implementing AlexNet and ResNet-18 Architectures in PyTorch**

## **Project Overview**
This repository demonstrates the implementation of two powerful deep learning models, **AlexNet** and **ResNet-18**, using **PyTorch**. Both models are primarily used for image classification tasks and are based on different architectures: AlexNet, a convolutional neural network (CNN), and ResNet-18, a residual network. These architectures are popular benchmarks in computer vision and have achieved great success in various image classification challenges.

## **Architectures**
### **1. AlexNet**
AlexNet is a deep convolutional neural network architecture introduced by **Alex Krizhevsky** in 2012. It was a significant breakthrough in deep learning and achieved state-of-the-art results in the **ImageNet** competition.

#### **Key Features:**
- **Convolutional Layers:** 5 convolutional layers followed by max-pooling layers.
- **Activation Function:** **ReLU (Rectified Linear Unit)** for non-linearity.
- **Fully Connected Layers:** 3 fully connected layers, with the last one corresponding to the output classes.
- **Dropout:** A **dropout** layer is included for regularization to reduce overfitting.
  
### **2. ResNet-18**
ResNet-18 is a part of the **Residual Networks** family, introduced by **Kaiming He et al.** in 2015. This architecture uses **residual connections** to help in training deeper networks by mitigating the vanishing gradient problem.

#### **Key Features:**
- **Residual Blocks:** Use of skip connections to bypass one or more layers.
- **Batch Normalization:** Applied after each convolutional layer to stabilize training and speed up convergence.
- **Adaptive Average Pooling:** Instead of using a fixed-size output, adaptive pooling outputs a fixed-size tensor (1x1).
  
## **Implementation Details**
This project includes two files, each implementing one of the architectures:

### **1. `alexnet.py`**
- Implements the **AlexNet** architecture in PyTorch.
- Contains 5 convolutional layers, max-pooling layers, dropout, and fully connected layers.

### **2. `resnet18.py`**
- Implements the **ResNet-18** architecture in PyTorch.
- Uses **residual blocks** with batch normalization and skip connections.

### **3. `requirements.txt`**
- Lists the dependencies required to run the project, such as PyTorch and Torchvision.

## **Setup Instructions**


### **1. Install Dependencies**

Make sure Python 3.8+ is installed, then create a virtual environment (optional) and install the dependencies:
```bash
pip install -r requirements.txt
```
The requirements.txt file includes:

- PyTorch. 

- Torchvision

### **2. Running the Models**

Once the environment is set up, you can run either of the models:

For AlexNet:

```bash

python alexnet.py
```
For ResNet-18:

```bash

python resnet18.py
```
### Model Summary
#### AlexNet
AlexNet consists of:

1. 5 convolutional layers with increasing depth.

2. Max-pooling layers after certain convolutional layers.
  
3. Fully connected layers, with ReLU activations in between.
  
4. A Dropout layer after the first fully connected layer.
 
Example Output of Model Summary:


```bash
AlexNet(
  (conv1): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
  (conv2): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (conv3): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv4): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=3, stride=2, padding=0)
  (fc1): Linear(in_features=9216, out_features=4096, bias=True)
  (fc3): Linear(in_features=4096, out_features=1000, bias=True)
  (dropout): Dropout(p=0.5)
)
```
### ResNet-18
ResNet-18 is built using residual blocks and is designed to be deeper with residual connections:

- It includes 4 layers with increasing depth.
  
- Each layer consists of BasicBlocks, which are repeated twice for each group.
  
Example Output of Model Summary:

```bash
ResNet18(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
  )
  ...
)
```








