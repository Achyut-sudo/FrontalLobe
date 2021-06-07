# FrontalLobe

A library to create and train Artificial Nueral Network classifier and analyse their performance.

Currently, it only includes a class to implement Artificial Nueral Networks and a confusion matrix class for its analysis 
and validation named 'NueralNetwork' and 'confusionMatrix' respectively. It also include three other suppport classes namely
'NueronLayer', 'rolledVector' and its subclass 'skeletonRolledVector' for smooth and fast working of the 'NueralNetwork' class.

This is library also contains a special feature of real
time display of change in activation values of the component nuerons in the Nueral Network object while training, so that the user can understand the source of any numerical errors and also because it looks cool!

![](https://github.com/Achyut-sudo/FrontalLobe/blob/main/nn.gif)

## Iris Data set 

on testing this library on [UCI'S iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) an acuuracy of 100% was  achieved in 1000 epochs on test set of 30 examples.


The model used, contains a hidden of layer 3 nuerons with one bias so does the input layer of 5 input  nuerons and the ouput layer contains 3 nuerons  corresponding to each class label.Sigmoid activation function is used for the hidden layer as well as for the output layers with alpha value of 5. [click](https://github.com/Achyut-sudo/FrontalLobe/blob/main/driver.py) for the code used.


![](https://github.com/Achyut-sudo/FrontalLobe/blob/main/IrisTest.jpg)





## Requirements 

python > 3.6

## Installation

pip install FrontalLobe

## URLs

The url address of detailed explaination and features of component classes of this library are given below:

[NueralNetwork](https://github.com/Achyut-sudo/FrontalLobe/blob/main/NueralNetwork.md)

[confusionMatrix](https://github.com/Achyut-sudo/FrontalLobe/blob/main/confusionMatrix.md)

[NueronLayer](https://github.com/Achyut-sudo/FrontalLobe/blob/main/NueronLayer.md)

[rolledVector](https://github.com/Achyut-sudo/FrontalLobe/blob/main/rolledVector.md)

[skeletonRolledVector](https://github.com/Achyut-sudo/FrontalLobe/blob/main/skeletonRolledVector.md)






