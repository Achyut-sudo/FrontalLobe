# NueralNetwork

A class of 'FrontalLobe' library to create and train a Artificial Nueral Network and then predict on the basis of the training.




## Initialization 

A NueralNetwork object requires atmost 4 parameters (1 mandatory, 3 optional ) during initialization, they are :

- architecture : list containing details of layers or layer themselves of the NueralNetwork object (mandatory)

    - the NueralNetwork objects's architecture is made up of objects of the supporting class ['NueronLayer'](https://github.com/Achyut-sudo/FrontalLobe/blob/main/NueronLayer.md) 

    - the list for parameter 'architecture' can contain values of type:
        
        - int
        - tuple
        - 'NueronLayer'
    - the list for parameter 'architecture' can also be empty! but before training, new layers must be appended using 'append' method of the NueralNetwork class
        
        
- HLAF : 'Hidden Layer Activation Function'
- OLAF : 'Output Layer Activation Function'
- precision : 


## Features


## Attributes 




**NOTE: PLEASE DO NOT DIRECTLY ACCESS OR MODIFY ANY ATTRIBUTES**

as in this version (0.0.1) all the attributes are supposed to be modified by the class methods only and direct modification may lead to change in the behaviour of the NueralNetwork object and eventually to an error

## Methods
