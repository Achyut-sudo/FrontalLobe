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

    - the list for parameter 'architecture' can also be empty! but before training, new layers must be added using 'append' method of the NueralNetwork 'class-entropy cost' of the NueralNetwork object as it may hinder
        
        
- HLAF : 'Hidden Layer Activation Function'
    - the values for parameter 'HLAF' can be :
        - int : supported int values for parmeter 'HLAF' are :
            - 1 for sigmoid activation function
            - 2 for ReLu activation Function
            - 3 for tanH activation function
            - 4 for linear activation function
        - str : supported str values for parameter 'OLAF' are :
            

- OLAF : 'Output Layer Activation Function'
    - the values for parameter 'HLAF' can be :
        - int :
        - str :

- precision : number of decimal digits in the activation value of the nuerons and also for other internal calculation, except the 'cost' of the NueralNetwork object as restricting cost to a certain number of decimal may lead to a stationary cost and eventually hinder with workings of the gradient descent algorithm


## Features


## Attributes 




**NOTE: PLEASE DO NOT DIRECTLY ACCESS OR MODIFY ANY ATTRIBUTES**

as in this version (0.0.1) all the attributes are supposed to be modified by the class methods only and direct modification may lead to change in the behaviour of the NueralNetwork object and eventually to an error

## Methods
