# NueralNetwork

A class of 'FrontalLobe' library to create and train a Artificial Nueral Network and then predict on the basis of the training.

## Initialization 

A NueralNetwork object requires atmost 4 parameters (1 mandatory, 3 optional ) during initialization, they are :

- architecture : list containing details of layers or layer themselves of the NueralNetwork object (mandatory)

    - the NueralNetwork objects's architecture is made up of objects of the supporting class ['NueronLayer'](https://github.com/Achyut-sudo/FrontalLobe/blob/main/NueronLayer.md) 

    - the list for parameter 'architecture' can contain values of type:
        
        - int : width of the layer, the corresponding layer will include bias nueron by default (1 bias nueron + (passed width - 1) nuerons)


        - tuple : (int, bool) : (width of layer, bool value specifying to include bias), as opposed to the above 'int' bullet point, passing a tuple provides an extra field to specify whether to include bais nueron or not

        - 'NueronLayer' : a 'NueronLayer' object, see
[NueronLayer](https://github.com/Achyut-sudo/FrontalLobe/blob/main/NueronLayer.md) to check different ways to initialize a 'NueronLayer' object    

    - the list for parameter 'architecture' can also be empty! but before training, new layers must be added using 'append' method of the NueralNetwork
        
        
- HLAF : 'Hidden Layer Activation Function', activation function for forward propagation in hidden layers 
    - the values for parameter 'HLAF' can be :
        - int : supported int values for parmeter 'HLAF' are :
            - 1 for sigmoid activation function
            - 2 for ReLU activation Function
            - 3 for tanH activation function
            - 4 for linear activation function
        - str : supported str values for parameter 'HLAF' are :
            - 'sigmoid; for sigmoid activation function
            - 'ReLU' for ReLU activation Function
            - 'tanH' for tanH activation function
            - 'linear' for linear activation function
        - default value : 'sigmoid'

- OLAF : 'Output Layer Activation Function', activation function for forward propagation to ouput layers
    - the values for parameter 'OLAF' can be :
        - int : supported int values for parmeter 'HLAF' are :
            - 1 for sigmoid activation function
            - 2 for tanH activation function
        - str : supported str values for parameter 'OLAF' are :
            - 'sigmoid; for sigmoid activation function
            - 'tanH' for tanH activation function
        - default value : 'sigmoid'

- precision : number of decimal digits in the activation value of the nuerons and also for other internal calculation, except the 'cost' of the NueralNetwork object as restricting cost to a certain number of decimal may lead to a stationary cost and eventually hinder with workings of the gradient descent algorithm



```
>>> import FrontalLobe
>>> nn = FrontalLobe.NueralNetwork([4,(3,True),FrontalLobe.NueronLayer(3),FrontalLobe.NueronLayer(3,False),(2,False)])
>>> print(nn)
input Layer    hiddenLayer 1    hiddenLayer 2    hiddenLayer 3    output Layer
-------------  ---------------  ---------------  ---------------  --------------
1.0
               1.0              1.0              0.0
0.0                                                               0.0
               0.0              0.0              0.0
0.0                                                               0.0
               0.0              0.0              0.0
0.0
>>>
```


## Features



## Attributes 




**NOTE: PLEASE DO NOT DIRECTLY ACCESS OR MODIFY ANY ATTRIBUTES**

as in this version (0.0.1) all the attributes are supposed to be modified by the class methods only and direct modification may lead to change in the behaviour of the NueralNetwork object and eventually to an error

## Methods

- [toString](#toString)
- [layer](#layer)
- [getDim](#getDim)
- [getTrainingStatus](#getTrainingStatus)
- [append](#append)
- [pop](#pop)
- [resetLayers](#resetLayers)
- [dump](#dump)
- [load](#load)
- [copy](#copy)
- [setBiases](#setBiases)
- [layerNormalization](#layerNormalization)
- [getWeightShape](#getWeightShape)
- [sigmoid](#sigmoid)
- [derivativeSigmoid](#derivativeSigmoid)
- [ReLU](#ReLU)
- [derivativeReLU](#derivativeReLU)
- [tanH](#tanH)
- [derivativeTanH](#derivativeTanH)
- [linear](#linear)
- [derivativelinear](#derivativelinear)
- [predict](#predict)
- [validate](#validate)
- [labelExtractor](#labelExtractor)
- [train](#train)



## Class Methods
***
## load
loads locally saved images of NueralNetwork object (nueralNetwok objects can be saved locally using 'dump' method of NueralNetwork object)

Parameters

- fileName : str
        
        name of the file for the local image of the NueralNetwork obj, include the directory if the desired location of the file is same as current working directory (do os.getcwd() to get current working directory)

Examples
```
>>> import FrontalLobe
>>> nnCopy = nn.load("d:\\codes\\ml\\NueralNett\\nnCopy")
>>> print(nnCopy)
input Layer    hiddenLayer 1    hiddenLayer 2    output Layer
-------------  ---------------  ---------------  --------------
1.0

0.0            1.0              1.0              0.0

0.0            0.0              0.0              0.0

0.0            0.0              0.0              0.0

0.0
>>> nnCopy.HLAF.__name__
'linear'
>>> nnCopy.OLAF.__name__
'sigmoid'
>>>
```
***

## Instance Methods
***
## toString

returns string representation of the architecture and activation values of the component Nuerons in NueralNetwork obj, string representation can also be returned without the dashed underlines in layer headings by setting the boolean parameter 'preintUnderline' to False.

Parameters 

- printUnderline : bool

        if True string representations  with dashed underline in the layer headings is passed otherwise string without the dashed undeline is returned, by default True

Examples
```
>>> print(nn.toString())
input Layer    hiddenLayer 1    hiddenLayer 2    output Layer
-------------  ---------------  ---------------  --------------
1.0

1.0            1.0              1.0              0.31458

0.340909       -0.362999        0.208062         0.332846

0.613636       0.011057         0.666664         0.359032

0.0
>>> print(nn.toString(False))
input Layer    hiddenLayer 1    hiddenLayer 2    output Layer
1.0

1.0            1.0              1.0              0.31458

0.340909       -0.362999        0.208062         0.332846

0.613636       0.011057         0.666664         0.359032

0.0
>>>

```
***
## layer
returns layer (NueronLayer object) at the passed  parameter 'layerNumer'

**layer indexing for a NueralNetwork object starts from zero**

Parameters 
- layerNumber 

    index of the desired layer 

Example
```
>>> nn.layer(2)
FrontalLobe.NueronLayer(width=3,includeBias=True)
>>> nn.layer(2).includeBias
True
>>> nn.layer(2).layer
array([[ 1.      ],
       [-0.362999],
       [ 0.011057]])
>>>

```
***

## getDim
returns dimension of the architecture of the NueralNetwork object,
dimension of the NueralNetwork object is a tuple containing two elements, 1st the width of the layer with maximum width of the arhitecture of the NueralNetwork and second the number of layers. By default the input layer is not considered for the number of layers, but can be included by setting the parameter 'includeIL' to True.

Parameters

- inludeIl

    if True input layer is also considered for 2nd member of return tuple (number of layers in architecture), by default False

Example
```
>>> print(nn.getDim())
(5, 3)
>>> print(nn.getDim(True))
(5, 4)
>>>

```
***
## getTrainingStatus
returns True if the NueralNetwork object is not trained using [train](#train)
method of the NueralNetwork object, atleast once.

Examples
```
>>> nnCopy.getTrainingStatus()
True
>>>
```
***
## append
appends a new NueronLayer at the end of  NueralNetwork object onlhy if the object is not trained using [train](#train) method of the NueralNetwork object, atleast once.

Parameters 

- layerData : int, NueronLyer object, tuple of int and bool
        
        parameters of layer to append


***
## pop

***
## resetLayers
***

## dump

***
## copy
***

## setBiases

***
## layerNormalization

***
## getWeightShape

***
## sigmoid

***

## derivativeSigmoid

***
## ReLU

***
## derivativeReLU

***
## tanH

***
## derivativeTanH

***
## linear

***
## derivativelinear

***
## predict

***
## validate

***
## labelExtractor

***
## train
***
