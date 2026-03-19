# Image Classification with Convolutional Neural Networks (CNN)
## Overview
Image classification is a critical task in computer vision applications, enabling machines to recognise and categorise images accurately. Over the years, there has been a remarkable evolution in image classification algorithms, transitioning from traditional feature-based methods to more advanced deep learning-based techniques. Among these, Convolutional Neural Networks (CNNs) have emerged as a standout success story, significantly improving classification accuracy.
### Project Description
This project, undertaken as part of the 3rd year of the Bachelor's degree program at SGSITS, Indore, involves implementing two popular CNN models, LeNet and VGGNet, and comparing their performance in image classification. The primary goal is to achieve high accuracy in classifying images using deep learning techniques.
#### LeNet Model
The LeNet model, pioneered by Yann LeCun, is an early and influential CNN architecture. It consists of convolutional layers with subsampling, leading to a compact yet effective model. 
#### VGGNet Model
VGGNet, developed by the Visual Graphics Group at Oxford, is known for its simplicity and depth. The model comprises stacked convolutional layers with small receptive fields, followed by max-pooling layers.
## Dataset
We use ciFAIR-10, a cleaned version of CIFAR-10 with a duplicate-free test set.
### Classes:
| airplane | automobile | bird | cat | deer |
|----------|------------|------|-----|------|
| dog      | frog       | horse| ship| truck|
### Image size:
32 x 32 x 3
### Samples
Train: 50000
Test: 10000
## Data preprocessing
### Convert to float32
```
# Connvert data type to float for computation
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
```
### Normalize (divide by 255)
```
# Normalize the data
x_train /= 255
x_test /= 255
```
### One-hot encoding of labels
```
# Convert class vectors to binary class matrices (One hot encoding)
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```
## Classification Techniques
### LeNET
#### Layers
```Conv2D(6,5x5) → MaxPooling → Conv2D(16,5x5) → MaxPooling → Flatten → Dense120 → Dense84 → Dense10```
#### Training parameters
| Parameter | Value |
|----------|--------|
| Batch size | 64 |
| Epochs | 30 |
| Optimizer | Adam |
| Loss | categorical_crossentropy |
#### Performance
![LeNet Grapt](Images/LeNet_Graph.png)
![LeNet Matrix](Images/LeNet_Matrix.png)
### VGGNet
#### Layers
```
[Input]
   ↓
[Block1: Conv32 → Conv32 → MaxPool → Dropout]
   ↓
[Block2: Conv64 → Conv64 → MaxPool → Dropout]
   ↓
[Block3: Conv128 → Conv128 → MaxPool → Dropout]
   ↓
[Flatten]
   ↓
[Dense 4096]
   ↓
[Dense 4096]
   ↓
[Dense 10]
   ↓
[Softmax]
   ↓
[Output]
```
#### Training parameters
| Parameter | Value |
|----------|--------|
| Batch size | 64 |
| Epochs | 30 |
| Optimizer | Adam |
| Loss | categorical_crossentropy |
#### Performance
![VGG Grapt](Images/VGG_Graph.png)
![VGG Matrix](Images/VGG_Matrix.png)
## Comparison
![LeNet vs VGG](Images/LeNet_vs_VGG.png)
