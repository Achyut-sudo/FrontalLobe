import tkinter as tk
import numpy as np
import tabulate
import warnings
import math
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import reprint
import pickle


np.seterr(divide='raise')

showWarnings = False
warnings.simplefilter('always',UserWarning)
#import sys
#sys.tracebacklimit = 1
popRow = lambda arr,r:(arr[r],np.vstack([arr[:r,:],arr[r+1:,:]]))
popColumn = lambda arr,c:(arr[:,c],np.hstack([arr[:,:c],arr[:,c+1:]]))

class rolledVector:
	"""
	data structure to store multiply numpy.ndarrays as one dimensional members 
	arrays and implement operators/methods collectively

	...

	Attributes
	----------
	vector : list
		list storing all reshaped numpy.ndarray
	shapeData : list
		list storing all original shapes of memeber
		numpy.ndarrays
	size : int
		sum of size of all numpy.ndarrays
	indexType : int
		0 if indexing starts from 0 else 1 

	Methods
	-------
	addMatrix(matrix : numpy.ndarray)
		adds a new numpy matrix to the rolledVector object
	update(updateValue : [list,numpy.ndarray])
		updates attribute 'vector' 
	getRolledIndex(pos1: int,pos2: int)
		returns index of a elemnt according to the original 
		numpy.ndarray from element's index in 'vector' attribute
		of rolledVector obj
	addUpdate(updateVal : [rolledVector,skeletonRolledVector])
		inplace element wise addition of the rolledVector object 
		with argument updateVal
	pow(exponent : int)
		raises elech element of rolledVector object to exponent 
		and returns it
	reverse()
		inplace reversal of rolledVector object
	isEmpty()
		return True if attributes vector and shapeData is Empty
		else returns False
	scalarMultiply(multiplicant : [int,float])
		inplace element wise multiplication of rolledVector object
		with argument multiplicant
	copy()
		returns a copy of rolledVector object
	sum(sumType :  int)
		return overall sum,member array sum,row wise sum of member arrays
		and column wise sum of member arrays

	Generators
	----------
	unrolledIteration()
		generator for iterating over element wise rather than member wise 
		of the rolledVector obj  as in __iter__ method 

	** for detailed explanation see help(FrontalLobe.rolledVector.<method Name/generator Name>)
	The docstring examples assume that `FrontalLobe` has been imported as `FrontalLobe`
	
	"""

	def __init__(self,zeroIndexed = False):
		"""

		Parameters
		----------
		zeroIndexed : bool, optional
			indexing starts from 0 if True else 1, by default False

		Raises
		------
		TypeError
			if parameter 'zeroIndexed' is of any type other than bool
		"""

		if(type(zeroIndexed) != bool):
			raise TypeError("argument 'zeroIndexed' must be of type bool")
		self.vector = []
		self.shapeData = []
		self.size = 0
		self.indexType = 0 if zeroIndexed == True else 1

	def addMatrix(self,matrix):
		"""adds a new numpy.ndarray to the rolledVector object

		Parameters
		----------
		matrix : numpy.ndarray
			array to be added to the rooledVector object

		Raises
		------
		ValueError
			if parameter 'matrix' is of any type other than numpy.ndarray
		"""
		if type(matrix) != np.ndarray :
			raise ValueError("argument 'matrix' must be of type list or numpy.ndarray")
		self.shapeData.append(matrix.shape)
		self.vector.append(matrix.reshape(1,matrix.size)[0])
		self.size += matrix.size

	def __repr__(self):
		return self.vector.__repr__()

	def __getitem__(self,index):
		if type(index) in [int,tuple] :
			if type(index) == int :
				try:
					return self.vector[index - self.indexType].reshape(self.shapeData[index-self.indexType])
				except IndexError :
					raise IndexError("rolledVector index out of range") from None
			else:
				if len(index) == 1:
					return self.__getitem__(index[0])
				if len(index) == 2:
					return self.__getitem__(index[0])[index[1]]
				if len(index) == 3:
					return self.__getitem__(index[:2])[index[2]]
				if len(index) > 3:
					raise IndexError("rolledVector index out of range") from None
		else:
			raise TypeError("rolledVector indices must be int or tuple") from None

	def __iter__(self):
		self.start = 0
		return self

	def __next__(self):
		self.start += 1
		try:
			return self.__getitem__(self.start)
		except IndexError:
			raise StopIteration

	def __len__(self):
		return len(self.shapeData)

	def __add__(self,other):
		if self.shapeData != other.shapeData:
			raise AttributeError(f"attribute 'shapeData' of both addends must be same") from None
		retVal = rolledVector()
		for i,j in enumerate(self):
			retVal.addMatrix(j+other[i + other.indexType])
		return retVal

	def __sub__(self,other):
		if self.shapeData != other.shapeData:
			raise AttributeError(f"attribute 'shapeData' of both addends must be same") from None
		retVal = rolledVector()
		for i,j in enumerate(self):
			retVal.addMatrix(j-other[i + other.indexType])
		return retVal

	def __eq__(self,other):
		if type(other) not  in [rolledVector,skeletonRolledVector]:
			return False
		else:
			if other.shapeData != self.shapeData :
				return False
			else:
				for i,ind,val in self.unrolledIteration():
					if other[i][ind] != val:
						return False
			return True

	def update(self,updateVal):
		"""updates rolledVector obj with parameter updateVal

		updates each element to parameter updateVal,if type(updateVal) 
		is int or float,else updates member wise according to
		the parameter updateVal

		Parameters
		----------
		updateVal : int,float,list,rolledVector,skeletonRolledVector
			if type of parameter updateVal is in [list,rolledVector,
			skeletonRolledVector] then it must match the attribute 
			'shapeData' of the rolledVector obj

		Raises
		------
		TypeError
			if type of parameter updateVal is not in [int,float,list,
			rolledVector,skeletonRolledVector]
		ValueError
			if type of parameter updateVal is in [rolledVector,
			skeletonRolledVector] and it's size doesn't match with size 
			of rolledVector obj
		ValueError
			if type of parameter updateVal is in [rolledVector,
			skeletonRolledVector] and it's 'shapeData' attribute doesn't 
			match with 'shapeData' attribute of rolledVector obj
		ValueError
			if type of parameter updateVal is list and no of numpy.ndarray 
			memebers of updateVal doesnot match number of array members of 
			rolledVector obj 
		TypeError
			if type of parameter updateVal is list and atleast one of it's
			memberss is of type other tha numpy.ndarray
		ValueError
			if type of parameter updateVal is list and atleast one of its member 
			numpy.ndarrays is compatible with 'shapeData' attribute of rolledVector 
			obj 
		"""
		if type(updateVal) not in [int,float,list,rolledVector,skeletonRolledVector]:
			raise TypeError(f"argument 'updateVal' must of type int,float, list, rolledVector or skeletonRolledVector")
		else:
			if type(updateVal) in [rolledVector,skeletonRolledVector]:
				if updateVal.size != self.size:
					raise ValueError(f"could not broadcast input rolledVector updateVal of  size {updateVal.size} into size {self.size}") from None
				if(updateVal.shapeData != self.shapeData):
					raise ValueError(f"input rolledVector updateVal's attribute 'shapeData' is not compatible ") from None
				self.vector = updateVal.vector.copy()
			if type(updateVal) == list:
				if len(updateVal) != len(self.shapeData):
					raise ValueError(f"could not broadcast input list updateVal of {len(updateVal)} members into {len(self.shapeData)} members") from None
				for i,j in enumerate(updateVal):
					if type(j) != np.ndarray:
						raise TypeError(f"argument 'updateVal' must contain members of type of numpy.ndarray") from None
					if j.shape != self.shapeData[i] :
						raise ValueError(f"could not broadcast member numpy.ndarray at index {j} of argument 'updateVal' of shape {j.shape} into {self.shapeData[i]} ") from None
				self.vector = updateVal
			if type(updateVal) in [int,float] :
				for i,j in enumerate(self.vector):
					self.vector[i] = np.full(j.shape,updateVal)
	
	def getRolledIndex(self,pos1,pos2): 
		"""return index of element in original array

		indexing of elements of rolledVector in attribute 'vector'
		where they are stored, is different from their index in 
		original numpy.ndarray, this method returns index in original
		nump.ndarray fom their index in 'vector' attribute of 
		rolledVector obj

		Parameters
		----------
		pos1 : int
			index of single dimensional member array in which element is stored
			**indexing of rolledVector obj must be considered, i.e. for 
			  1st array use 1 if indexing of rolledVector starts from 1 
			  else 0 and so forth for further indices, to know indexing of 
			  rolledVector obj check indexType attribute of the rolledVector
			  obj
		pos2 : int
			index of element in the single dimensional array, indexing starts
			from 0 

		Returns
		-------
		tuple : (int,int)
			index of element at passed positions in original numpy.ndarray 
		"""
		return (int(pos2/self.shapeData[pos1-self.indexType][1]),pos2%self.shapeData[pos1-self.indexType][1])
	
	def unrolledIteration(self):
		"""iterates over each element of rolledVector object

		yields index of numpy.ndarray , index of element in the 
		numy.ndarray of which it is a member of, along with element

		Yields
		-------
		int,tuple: (int,int),[int,float]
			int : index of numpy.ndarray
			tuple : (int,int) : index of element in the numpy.ndarray
			[int,float] : element of roilledVector object

		"""
		for i,j in enumerate(self.vector):
			for k in range(j.size):
				yield (i+self.indexType ,self.getRolledIndex(i if self.indexType == 0 else i+1,k),j[k])

	def addUpdate(self,updateVal):
		"""updates rolledVector object by inplace element wise 
		   addition with updateVal

		Parameters
		----------
		updateVal : rolledVector,skeletonRolledVector
			[description]

		Raises
		------
		TypeError
			if type of parameter 'updateVal' is not in 
			[rolledVector,skeletonRolledVector]
		AttributeError
			if attribute 'shapeData' does not match with that 
			of the rolledVector upon which the method is called
		"""

		if type(updateVal) not in [rolledVector,skeletonRolledVector]:
			raise TypeError("argument 'updateVal' must be rolledVector or skeletonRolledVector")
		if updateVal.shapeData != self.shapeData:
			raise AttributeError(f"attribute 'shapeData' of updateVal is not compatible")
		self.vector = (self + updateVal).vector 

	def __round__(self,precison=None):
		if precison is not None:
			for i in range(self.__len__()):
				for j,k in enumerate(self.vector[i]):
					self.vector[i][j] = round(k,precison)

	def pow(self,exponent):
		"""raises each member element to 'exponent' and returns it

		each element of member array is raised to value of parameter
		'exponent' and the rolledVector object of it is returned 

		Parameters
		----------
		exponent : int
			exponent to which the member elements are to be raised

		Returns
		-------
		obj : rolledVector

		Raises
		------
		TypeError
			if type of parameter 'exponent' is not int
		"""
		if type(exponent) != int:
			raise TypeError("exponent must be an integer, but exponent of type (type(exponent)) was given")
		retVal = rolledVector()
		for i in self:
			retVal.addMatrix(i**exponent)
		return retVal

	def reverse(self):
		"""inplace reversal of rolledVector object
		"""
		self.vector.reverse()
		self.shapeData.reverse()

	def isEmpty(self):
		"""returns True if rolledVector is empty else False

		checks if rolledVector object is empty by checking 
		if attributes 'vector' and 'shapedata' is empty.

		Returns
		-------
		bool
			True if attributes 'vector' and shapeData' is empty
		"""
		return (self.vector == [] and self.shapeData == [])

	def scalarMultiply(self,multiplicant):
		"""inplace scalar multiplication of member arrays with multiplicant

		Parameters
		----------
		multiplicant : int,float
			int or float value to be multiplied to member arrays

		Raises
		------
		TypeError
			if type of multiplicant is not in [int,float]
		"""
		if type(multiplicant) not in [int,float]:
			raise TypeError("multiplicant must be scalar")
		for i,j in enumerate(self.vector):
			self.vector[i] = multiplicant*j

	def copy(self):
		"""returns copy of the rolledVector object 

		Returns
		-------
		rolledVector
			copy of rolledVector object upon which 'copy'
			method is called
		"""
		retVal = rolledVector()
		for i in self:
			retVal.addMatrix(i.copy())
		return retVal

	def sum(self,sumType=0):
		"""return memeber/dimension wise sum of member arrays

		returns (float) sum of all elements of all member arrays,
		if no parameter is passed

		Parameters
		----------
		sumType : int, optional
			indicates type os sum required, by default 0

		Returns
		-------
		float
			sum of all elements of all members,if parameter
			sumType == 0
		rolledVector
			rolledVector obj of member wise sum,if parameter
			sumType == 1
			rolledVectror obj of row wise sum of each member 
			arrays,if parameter sumType == 2
			rolledVector obj of column wise of each member 
			arrays,if parameter sumType == 3

		Raises
		------
		ValueError
			if sumType is not in [0,1,2,3]
		"""
		if sumType not in [0,1,2,3]:
			raise ValueError(f"{sumType} is not a valid value for sumType, supported values are 0, 1, 2 and 3")
		if sumType == 0 :
			retVal = 0 
			for i in self.__iter__():
				retVal += np.sum(i)
			return retVal
		else:
			retVal = rolledVector()
			for i in self.__iter__():
				if sumType == 1:
					s = np.array([np.sum(i)])
				if sumType == 2:
					s = i.sum(axis=0)
				if sumType == 3:
					s = i.sum(axis=1,keepdims =True)
				retVal.addMatrix(s)
			return retVal

class skeletonRolledVector(rolledVector):
	"""
	subclass of rolledVector class,with shapeData passed as
	parameter during initialization

	** for detailed explanation see help(FrontalLobe.skeletonRolledVector.<method Name/generator Name>)
	The docstring examples assume that `FrontalLobe` has been imported as `FrontalLobe

	"""

	def __init__(self,shapeData,zeroIndexed = False,dummyVals = None):
		"""

		Parameters
		----------
		shapeData : list :[int,tuple: (int,int)]
			list of shapes of member numpy.ndarray
		zeroIndexed : bool, optional
			indexing starts from 0 if True else from 1, by default False
		dummyVals : [int,float], optional
			dummy value,sets each element to value passed to dummyVals, 
			by default None

		Raises
		------
		TypeError
			if type of atleast one of members of parameter 'shapeData' is not in 
			[int,tuple]
		"""
		super().__init__(zeroIndexed)
		for i in shapeData :
			if type(i) not in [tuple,int]:
				raise TypeError("elemnts of arguments 'shapeData' must be tuple or int")
			shape = i if type(i) == tuple else (i,)
			self.addMatrix(np.full(shape,dummyVals))

	def __setitem__(self,index,value):
		if type(index) != int:
			raise TypeError("skeletonRolledVector index must be int (while assigning value)") from None
		if type(value) != np.ndarray:
			raise TypeError("input value must be np.ndarray") from None
		if value.shape != self.__getitem__(index).shape :
			raise ValueError(f"could not broadcast input array from shape {value.shape} into shape {self.__getitem__(index).shape}") from None
		self.vector[index -self.indexType] = value

class confusionMatrix:
	u"""
	class for describing  performance of a classification
	model

	Example
	-------
	>>> cm = nn.validate() # nn -> FrontalLobe.NueralNetwork obj  
	>>> print(cm)
	actual Values→       class0    class1    class2
	predict Values↓
	-----------------  --------  --------  --------
	class0                   15         7         4
	class1                   13         8         9
	class2                    7         6        11
	>>>

	**in the above example, row 'class0' contains all cases
	where the predicted class was 'class0' and column 'class1' 
	represents all cases where expected/actual class was 'class1' 

	...

	Attributes
	----------
	NClasses : int
		no of classes to classify
	matrix : np.ndarray
		actual confusion matrix that stores no. of class 
		wise predicted cases  along the rows and no. of 
		class wise expected/actual cases along the columns. 
	className: list,numpy.ndarray
		names of class
		if not set,will be set to 'class0', 'class1' an so on if 
		NClasses > 2, otherwise 'Negative', 'positive'

	Methods
	-------
	update(predictedVal: [int,list,numpy.ndarray],expectedVal: [int,list,numpy.ndarray])
		updates value attribute 'matrix' according to class represented by 
		predictedVal and expectedVal
	toString(printUnderLine: bool)
		returns string representation of confusion matrix
	heatmap(colorscheme : str,optional, textColor : str,optional, fontsize : int)
		returns heatmap of the confusion matrix
	getTN(*args)
		returns True Negatives of confusion matrix
	getTP(*args)
		returns True positives of confusion matrix 
	getFN(*args)
		returns False Negatives of confusion matrix
	getFP(*args)
		returns False positives of confusion matrix
	accuracy(**kwargs)
		returns accuracy score of confusion matrix
	missclassificationRate(**kwargs)
		returns missclassificationRate of confusion matrix
	precision(**kwargs)
		returns precison score of confusion matrix
	recall(**kwargs)
		returns recall score of confusion matrix
	specificity(**kwargs)
		returns specificity score of confusion matrix
	FIScore(**kwargs)
		returns F1Score of confusion matrix
	microF1(roundTo : int,optional)
		return microF1 of the confusion matrix
	macroF1(roundTo : int,optional)
		returns macroF1 of the confusion matrix
	weightedF1(roundTo : int,optional)
		returns weightedF1 of the confusion matrix
	report(roundTo : int,optional, format : str,optional)
		report of confusion matrix

	** for detailed explanation see help(FrontalLobe.confusionMatrix.<method Name/generator Name>)
	The docstring examples assume that `FrontalLobe` has been imported as `FrontalLobe'

	"""
	def __init__(self,NClasses,classNames = None):
		"""
		Parameters
		----------
		NClasses : int
			no of classes to classify
		classNames : str, optional
			names of class to classify, by default None

		Raises
		------
		TypeError
			if type of parameter 'NClasses' is other than int
		TypeError
			if type of parameter 'classNames' other than 
			NoneType, list, numpy.ndarray
		TypeError
			if type of parameter 'classNames' is np.ndarray and it's
			ndim > 1 
		ValueError
			if parameter 'NClasses' is less than 2
		ValueError
			if no. of member elements of parameter 'classNames' does
			not match attribute 'NClasses' of the confusionMatrix obj

		"""
		if type(NClasses) != int:
			raise TypeError("argument 'NClasses' must be of type int")
		if classNames is not None :
			if type(classNames) not in [list,np.ndarray]:
				raise TypeError("argument 'classNames' must be of type list or numpy.ndarray")
			if type(classNames)  == np.ndarray:
				if classNames.ndim != 1:
					raise TypeError("argument 'classNames' can be only one dimensional")
				else:
					classNames = list(classNames)
		if NClasses < 2:
			raise ValueError("argument'NClasses' can only be >= 2")
		if classNames is not None:
			if len(classNames) != NClasses :
				raise ValueError("size of argument 'classNames' does not match argument 'NClasses'")
		self.NClasses = NClasses
		self.matrix = np.zeros((NClasses,NClasses))
		if self.NClasses > 2:
			if classNames is not None:
				self.classNames = list(classNames) if type(classNames) == np.ndarray else classNames
			if classNames is None:
				self.classNames = [f"class{i}" for i in range(NClasses)]
		if self.NClasses == 2:
			self.classNames=["Negative","positive"]

	def __intLE(self,data,argName):
		"""extracts class label form parameter 'data'

		this method is used in other methods of class
		confusionMatrix to extract class label from a 
		int parameter 

		Parameters
		----------
		data : int
			value of parameter 'argName' passed to method
			which implements this method
		argName : str
			name of parameter passed to method which 
			implements this method

		Returns
		-------
		int
			class label

		Raises
		------
		ValueError
			if value of parameter 'data' is < 0 or > 'Nclasses' 
			attribute of confusionMatrix

		"""
		if data < 0 or data >= self.NClasses:
			raise ValueError(f"argument '{argName}' must be in [0,{self.NClasses-1}]")
		else:
			return data

	def __listLE(self,data,argName):
		"""extracts class label form parameter 'data'

		this method is used in other methods of class
		confusionMatrix to extract class label from a 
		list parameter 

		Parameters
		----------
		data : list
			value of parameter 'argName' passed to method
			which implements this method
		argName : str
			name of parameter passed to method which 
			implements this method

		Returns
		-------
		int
			class label

		Raises
		------
		TypeError
			if value of parameter 'data' is multi dimensional list
		ValueError
			if attribute 'NClasses' == 2 and len of parameter 'data' 
			is > 1
		ValueError
			if atleast one member of parameter 'data' is not in [0,1]
		ValueError
			if len of parameter 'data' is not equal to attribute 'NClasses'
		ValueError
			if atleast one member of parameter 'data' is other than 0 or 1
		ValueError
			if more tha one member of parameter 'data' is 1
		ValueError
			if no member of parameter 'data' is 1

		"""
		if np.array(data).ndim > 1:
			raise TypeError(f"argument '{argName}' can only be one dimensional list ")
		if self.NClasses == 2 :
			if len(data) > 1:
				raise ValueError(f"update() takes argument '{argName}' of size 1, but argument '{argNam}' of size {len(data)}  was given ")
			if data[0,0] not in [0,1] :
				raise ValueError(f"{argName} can only contain values 0 or 1")
			return data[0]
		if self.NClasses > 2:
			if len(data) != self.NClasses :
				raise ValueError(f"update() takes argument '{argName}' of size {self.Nclasses}, but argument '{argNam}' of size {len(data)}  was given ")
			retVal = None 
			for j,i in enumerate(data):
				if i not in [0,1]:
					raise ValueError(f"argument '{argName}' must contain 0 or 1 only")
				if i in [1]:
					if retVal is not None:
						raise ValueError(f"argument'{argName}' can not contain multiple 1 ")
					retVal = j
			if retVal is None :
				raise ValueError(f"argument '{argName}' must contain a single 1")
			return retVal

	def __ndarrayLE(self,data,argName):
		"""extracts class label form parameter 'data'

		this method is used in other methods of class
		confusionMatrix to extract class label from a 
		numpy.ndarray parameter 

		Parameters
		----------
		data : numpy.ndarray
			value of parameter 'argName' passed to method
			which implements this method
		argName : str
			name of parameter passed to method which 
			implements this method

		Returns
		-------
		int
			class label

		Raises
		------
		TypeError
			if value of parameter 'data' is not two dimensional list
		ValueError
			if attribute 'NClasses' == 2 and shape of parameter 'data' 
			is not (1,1)
		ValueError
			if atleast one member of parameter 'data' is not in [0,1]
		ValueError
			if shape of parameter 'data' doesnot match the required shape
			**rows of parameter 'data' must be equal to attribute 'NClasses'
			**column of parameter 'data' must be 1

		**this method uses _confusionMatrix__listLE of class confusionMatrix. 
		check help(FrontalLobe.confusionMatrix._confusionMatrix__listLE) for 
		other errors that might be raised

		The docstring examples assume that `FrontalLobe` has been imported as `FrontalLobe'

		"""
		if data.ndim != 2:
			raise TypeError(f"argument '{argName}' can only be 2 dimensional numpy.ndarray")
		if self.NClasses == 2:
			if data.shape != (1,1):
				raise ValueError(f"update() takes argument '{argName}' of size (1,1), but argument '{argName}' of shape {data.shape}  was given ")
			if data[0,0] not in [0,1] :
				raise ValueError(f"{argName} can only contain values 0 or 1")
			return data[0,0]
		if self.NClasses > 2:
			if data.shape != (self.NClasses,1):
				raise ValueError(f"update() takes argument '{argName}' of shape ({self.Nclasses,1}), but argument '{argName}' of shape {data.shape}  was given ")
			return self.__listLE(list(data.reshape(1,data.size)[0]),argName)

	def update(self,predictedVal,expectedVal):
		"""updates confusion matrix according to the expectedVal and predictedVal

		updates row,column of 'matrix' attribute of confusionMatrix obj according 
		to class represented by parameters 'predictedVal' and 'expectedVal'
		respectively

		Parameters
		----------
		predictedVal : [int,list,np.ndarray]
			parameter to represent predicted class
		expectedVal : [type]
			parameter to represent expected/actual class

		Raises
		------
		TypeError
			if type of any one of parameters 'predictedVal' and expectedVal' is
			other than [int,list,np.ndarray]

		**this method uses _confusionMatrix__intLE,_confusionMatrix__listLE,
		_confusionMatrix__ndarrayLE of class confusionMatrix to extract class 
		from parameters 'predictedVal and 'expectedVal', these methods 
		also raise errors. check 
		help(FrontalLobe.confusionMatrix._confusionMatrix__intLE) ,
		help(FrontalLobe.confusionMatrix._confusionMatrix__listLE), 
		help(FrontalLobe.confusionMatrix._confusionMatrix__ndarrayLE) for these errors.

		The docstring examples assume that `FrontalLobe` has been imported as `FrontalLobe'

		"""
		if type(predictedVal) not in [int,list,np.ndarray] or type(expectedVal) not in [int,list,np.ndarray] :
			errorArg,errorType = ("predictedVal",type(predictedVal)) if type(predictedVal) not in [int,list,np.ndarray] else ("expectedVal",type(expectedVal))
			raise TypeError(f"{errorType} is not valid type for {errorArg}, supported type are: 'int', 'list',np.ndarray'")
		extractorFuncs = [self.__intLE,self.__listLE,self.__ndarrayLE]
		predictedClass = extractorFuncs[[int,list,np.ndarray].index(type(predictedVal))](predictedVal,"predictedVal")		
		expectedClass = extractorFuncs[[int,list,np.ndarray].index(type(expectedVal))](expectedVal,"expectedVal")
		self. matrix[predictedClass,expectedClass] += 1
				
	def toString(self,printUnderline = True):
		"""returns string repersentation of confusionMatrix obj

		Parameters
		----------
		printUnderline : bool, optional
			True: adds underline to the row and column heads
			False: returns string without any underline, by default True

		Returns
		-------
		str
			string representation 

		"""
		cornerStr = "actual Values"+u"\u2192"+"\n"+"predict Values" + u"\u2193"
		columnHeads = np.array([[f"{className}"] for className in self.classNames])
		temp = np.hstack([columnHeads,self.matrix])
		table = tabulate.tabulate(temp,[cornerStr]+self.classNames)
		if printUnderline == True:
			return table
		else:
			table = table.split("\n")
			table.pop(2)
			return "\n".join(table)

	def __repr__(self):
		return f"FrontalLobe.confusionMatrix({self.NClasses},{self.classNames})"

	def __str__(self):
		return self.toString()

	def heatmap(self,colorScheme = "YlGn",textColor="black",fontSize = 10):
		"""plots heatmap of the confusionMatrix obj

		Parameters
		----------
		colorScheme : str, optional
			sets color scheme of the heatmap plot, by 
			default "YlGn"
		textColor : str, optional
			sets color of text of the  heatmap plot, by 
			default "black"
		fontSize : int, optional
			sets font size of text of the heatmap plot, by 
			default 10

		"""
		fig, ax = plt.subplots()
		im = ax.imshow(self.matrix,cmap=colorScheme)
		ax.set_xticks(np.arange(self.NClasses))
		ax.set_yticks(np.arange(self.NClasses))
		ax.set_xticklabels(self.classNames)
		ax.set_yticklabels(self.classNames)
		plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
		for i in range(self.NClasses):
			for j in range(self.NClasses):
				text = ax.text(j,i,self.matrix[i,j],ha="center",va="center",color=textColor,size=fontSize)
		ax.set_title(f"total data size: {self.matrix.sum(axis=0).sum()}")
		fig.tight_layout()
		fig.colorbar(im)
		plt.show()

	def __iter__(self):
		self.start = -1
		return self

	def __next__(self):
		self.start += 1
		try:
			return (self.classNames[self.start],dict(zip(self.classNames,self.matrix[self.start,:])))
		except IndexError :
			raise StopIteration

	def getTN(self,*args):
		"""returns class wise True Negatives score of confusionMatrix object

		Parameters
		----------
		*args : iterable
			if attribute 'Nclasses' == 2, 
				no parameters required
			if attribute 'NClasses' > 2, 
				a str or int representing required class 

		Returns
		-------
		float
			TN score of passed class

		Raises
		------
		TypeError
			if attribute 'Nclasses' == 2 and any parameter is passed
		TypeError
			if attribute 'Nclasses' > 2 and more than one or no 
			parameter is passed
		TypeError
			if attribute 'NClasses' > 2 and type of parameter passed is
			not in [int,str]
		IndexError
			if attribute 'NClasses' > 2 and type of parameter is int, 
			but passed parameter is < 0 or >= 'NClasses' attribute of 
			confusionMatrix obj
		ValueError
			if attribute 'NClasses'> 2 and type of parameter is str, 
			but passed parameter is not in attribute 'classNames'

		**'attribute 'NClasses' > 2' represents confusionMatrix obj for
		  multi-class classification and 'attribute 'NClasses' == 2' 
		  represents confusionMatrix obj for binary-class classification

		"""
		if self.NClasses == 2:
			if len(args)> 0:
				raise TypeError(f"getTN() takes 1 argument but {1+len(args)} were given (in case of binary class confusionMatrix)") from None
			else:
				return self.matrix[0,0]        	                 
		else:
			if len(args) == 0 or len(args) > 1:
				raise TypeError(f"getTN() takes 2 arguments but {1+len(args)} were given (in case of multi class confusionMatrix)") from None
			else:
				args = args[0]
				if type(args) not in [str,int]:
					raise TypeError(f"argument can only be of type str or int") from None
				if type(args) == int:
					if args >= self.NClasses or args < 0:
						raise IndexError("class index out of range") from None
					classIndex = args
				if type(args) == str:
					if args not in self.classNames:
						raise ValueError(f"'{args}' is not a valid argument; supported values are {self.classNames.__str__()[1:-1]}") from None
					classIndex = self.classNames.index(args)
				gar,temp = popRow(self.matrix,classIndex)
				gar,temp = popColumn(temp,classIndex)
				gar = None
				return temp.sum(axis=0).sum()

	def getTP(self,*args):
		"""returns class wise True Positives score of confusionMatrix object

		Parameters
		----------
		*args : iterable
			if attribute 'Nclasses' == 2, 
				no parameters required
			if attribute 'NClasses' > 2, 
				a str or int representing required class

		Returns
		-------
		float
			TP score of passed class

		Raises
		------
		TypeError
			if attribute 'Nclasses' == 2 and any parameter is passed
		TypeError
			if attribute 'Nclasses' > 2 and more than one or no 
			parameter is passed
		TypeError
			if attribute 'NClasses' > 2 and type of parameter passed is
			not in [int,str]
		IndexError
			if attribute 'NClasses' > 2 and type of parameter is int, 
			but passed parameter is < 0 or >= 'NClasses' attribute of 
			confusionMatrix obj
		ValueError
			if attribute 'NClasses'> 2 and type of parameter is str, 
			but passed parameter is not in attribute 'classNames'

		**'attribute 'NClasses' > 2' represents confusionMatrix obj for
		  multi-class classification and 'attribute 'NClasses' == 2' 
		  represents confusionMatrix obj for binary-class classification
			
		"""
		if self.NClasses == 2:
			if len(args)> 0:
				raise TypeError(f"getTP() takes 1 positional argument but {1+len(args)} were given (in case of binary class confusionMatrix)") from None
			else:
				return self.matrix[1,1]
		else:
			if len(args) != 1:
				raise TypeError(f"getTP() takes 2 positional arguments but {1+len(args)} were given(in case of multi class confusionMatrix)") from None
			else:
				args = args[0]
				if type(args) not in [str,int]:
					raise TypeError(f"argument can only be of type str or int") from None
				if type(args) == int:
					if args >= self.NClasses or args < 0 :
						raise IndexError("class index out of range") from None
					classIndex = args
				if type(args) == str:
					if args not in self.classNames:
						raise ValueError(f"'{args}' is not a valid argument; supported values are {self.classNames.__str__()[1:-1]}") from None
					classIndex = self.classNames.index(args)
				return self.matrix[classIndex,classIndex]

	def getFN(self,*args):
		"""returns class wise False Negatives score of confusionMatrix object

		Parameters
		----------
		*args : iterable
			if attribute 'Nclasses' == 2, 
				no parameters required
			if attribute 'NClasses' > 2, 
				a str or int representing required class 

		Returns
		-------
		float
			FN score of cpassed class

		Raises
		------
		TypeError
			if attribute 'Nclasses' == 2 and any parameter is passed
		TypeError
			if attribute 'Nclasses' > 2 and more than one or no 
			parameter is passed
		TypeError
			if attribute 'NClasses' > 2 and type of parameter passed is
			not in [int,str]
		IndexError
			if attribute 'NClasses' > 2 and type of parameter is int, 
			but passed parameter is < 0 or >= 'NClasses' attribute of 
			confusionMatrix obj
		ValueError
			if attribute 'NClasses'> 2 and type of parameter is str, 
			but passed parameter is not in attribute 'classNames'
			
		**'attribute 'NClasses' > 2' represents confusionMatrix obj for
		  multi-class classification and 'attribute 'NClasses' == 2' 
		  represents confusionMatrix obj for binary-class classification

		"""
		if self.NClasses == 2:
			if len(args)> 0:
				raise TypeError(f"getFN() takes 1 positional argument but {1+len(args)} were given (in case of binary class confusionMatrix)") from None
			else:
				return self.matrix[0,1]
		else:
			if len(args) == 0 or len(args) > 1:
				raise TypeError(f"getFN() takes 2 positional arguments but {1+len(args)} were given(in case of multi class confusionMatrix)") from None
			else:
				args= args[0]
				if type(args) not in [str,int]:
					raise TypeError(f"argument can only be of type str or int") from None
				if type(args) == int:
					if args >= self.NClasses or args < 0 :
						raise IndexError("class index out of range") from None
					classIndex = args
				if type(args) == str:
					if args not in self.classNames:
						raise ValueError(f"'{args}' is not a valid argument; supported values are {self.classNames.__str__()[1:-1]}") from None
					classIndex = self.classNames.index(args)
				temp = np.hstack([self.matrix[:classIndex,classIndex],self.matrix[classIndex+1:,classIndex]])
				return temp.sum(axis=0).sum()

	def getFP(self,*args):
		"""returns class wise True Negatives score of confusionMatrix object

		Parameters
		----------
		*args : iterable
			if attribute 'Nclasses' == 2, 
				no parameters required
			if attribute 'NClasses' > 2, 
				a str or int representing required class 

		Returns
		-------
		float
			FP score of passed class

		Raises
		------
		TypeError
			if attribute 'Nclasses' == 2 and any parameter is passed
		TypeError
			if attribute 'Nclasses' > 2 and more than one or no 
			parameter is passed
		TypeError
			if attribute 'NClasses' > 2 and type of parameter passed is
			not in [int,str]
		IndexError
			if attribute 'NClasses' > 2 and type of parameter is int, 
			but passed parameter is < 0 or >= 'NClasses' attribute of 
			confusionMatrix obj
		ValueError
			if attribute 'NClasses'> 2 and type of parameter is str, 
			but passed parameter is not in attribute 'classNames'

		**'attribute 'NClasses' > 2' represents confusionMatrix obj for
		  multi-class classification and 'attribute 'NClasses' == 2' 
		  represents confusionMatrix obj for binary-class classification
			
		"""
		if self.NClasses == 2:
			if len(args)> 0:
				raise TypeError(f"getFP() takes 1 argument but {1+len(args)} were given (in case of binary class confusionMatrix)") from None
			else:
				return self.matrix[1,0]
		else:
			if len(args) == 0 or len(args) > 1:
				raise TypeError(f"getFP() takes 2 arguments but {1+len(args)} were given(in case of multi class confusionMatrix)") from None
			else:
				args =args[0]
				if type(args) not in [str,int]:
					raise TypeError(f"argument can only be of type str or int") from None
				if type(args) == int:
					if args >= self.NClasses or args < 0 :
						raise IndexError("class index out of range") from None
					classIndex = args
				if type(args) == str:
					if args not in self.classNames:
						raise ValueError(f"'{args}' is not a valid argument; supported values are {self.classNames.__str__()[1:-1]}") from None
					classIndex = self.classNames.index(args)
				temp = np.hstack([self.matrix[classIndex,:classIndex],self.matrix[classIndex,classIndex+1:]])
				return temp.sum(axis=0).sum()

	def accuracy(self,**kwargs):
		"""returns class wise accuray score of confusionMatrix obj

		accuracy = (TN + TP)/(TN + TP + FP + FN )

		Parameters
		----------
		*kwargs: roundTo : int,optional classRef : [int,str],optional
			roundTo : decimal places to round return value 
			classRef : represents required class
			if attribute 'Nclasses' == 2, 
				only one optional parameter roundTo is required
			if attribute 'NClasses' > 2, 
				one paramater 'classRef' is mandatory ,
				parameter 'roundTo' is optional

		Returns
		-------
		float
			accuracy score of passed class

		Raises
		------
		TypeError
			if attribute 'NClasses' == 2 and more than one 
			parameter is passed
		ValueError
			if attribute 'NClasses' == 2 and parameter other
			than 'roundTo' is passed
		TypeError
			if type of parameter roundTo is not int
		ValueError
			if paramter 'roundTo' is int, but < 1 
		TypeError
			if attribute 'NClasses' > 2 and 0 or more than
			2 parameters are passed
		KeyError
			if attribute 'NClasses' > 2 and parameter 'classRef'
			is not passed in kwargs
		KeyError
			if attribute 'Nclasses' > 2 and kwargs is of len 2,
			but does not contain parameters other than 'classRef' 
			or 'roundTo'

		**'attribute 'NClasses' > 2' represents confusionMatrix obj for
		  multi-class classification and 'attribute 'NClasses' == 2' 
		  represents confusionMatrix obj for binary-class classification 

		"""
		if self.NClasses == 2:
			if len(kwargs) > 1:
				raise TypeError(f"accuracy() takes 1 to 2 positional arguments but {1+len(kwargs)} were given (in case of binary class confusionMatrix)") from None
			if len(kwargs) == 0:
				return (self.getTN()+self.getTP())/(self.getTN()+self.getTP()+self.getFP()+self.getFN())
			if len(kwargs) == 1:
				if "roundTo" not in kwargs.keys():
					raise ValueError(f"accuracy() takes argument 'roundTo' , but {kwargs.keys()} were given (in case of binary class confusionMatrix)") from  None
				roundTo = kwargs["roundTo"]
				if roundTo is not None:
					if type(roundTo) != int :
						raise TypeError(f"argument 'roundTo' must be int") from None
					if type(roundTo) == int and roundTo < 1:
						raise ValueError(f"argument 'roundTo' must be int > = 1") from None
					return round(((self.getTN()+self.getTP())/(self.getTN()+self.getTP()+self.getFP()+self.getFN())),roundTo)
				if roundTo  is None :
					return (self.getTN()+self.getTP())/(self.getTN()+self.getTP()+self.getFP()+self.getFN())
		else:
			if len(kwargs) > 2 or len(kwargs) == 0:
				raise TypeError(f"accuracy() takes 2 to 3 positional arguments but {1+len(kwargs)} were given (in case of multi class confusionMatrix)") from None
			else:
				try:
					classRef = kwargs["classRef"]
				except KeyError:
					raise KeyError(f"accuracy() takes arguments 'classRef' (mandatory) and 'roundTo' (optional), but {kwargs.keys()} were given (in case of multi class confusionMatrix)") from None
				try:
					roundTo = kwargs["roundTo"]
				except KeyError :
					if len(kwargs) == 1:
						roundTo = None
					else:
						raise KeyError(f"accuracy() takes argument 'classRef' and sometimes 'roundTo' , but {kwargs.keys()} were given (in case of multi class confusionMatrix)") from None
				retVal = (self.getTN(classRef)+self.getTP(classRef))/(self.getTN(classRef)+self.getTP(classRef)+self.getFP(classRef)+self.getFN(classRef))
				if roundTo is not None:
					if type(roundTo) != int :
						raise TypeError(f"argument 'roundTo' must be int ") from None
					if type(roundTo) == int and roundTo < 1:
						raise ValueError(f"argument 'roundTo' must be int > = 1") from None
					return round(retVal,roundTo)
				if roundTo  is None :
					return retVal

	def misclassificationRate(self,**kwargs):
		"""returns class wise missclassificationRate of confusionMatrix obj

		misclassificationRate = (FN + FP)/(TN + TP + FP + FN)

		Parameters
		----------
		*kwargs: roundTo : int,optional classRef : [int,str],optional
			roundTo : decimal places to round return value 
			classRef : represents required class
			if attribute 'Nclasses' == 2, 
				only one optional parameter roundTo is required
			if attribute 'NClasses' > 2, 
				one paramater 'classRef' is mandatory ,
				parameter 'roundTo' is optional

		Returns
		-------
		float
			missclassificationRate score of passed class

		Raises
		------
		TypeError
			if attribute 'NClasses' == 2 and more than one 
			parameter is passed
		ValueError
			if attribute 'NClasses' == 2 and parameter other
			than 'roundTo' is passed
		TypeError
			if type of parameter roundTo is not int
		ValueError
			if paramter 'roundTo' is int, but < 1 
		TypeError
			if attribute 'NClasses' > 2 and 0 or more than
			2 parameters are passed
		KeyError
			if attribute 'NClasses' > 2 and parameter 'classRef'
			is not passed in kwargs
		KeyError
			if attribute 'Nclasses' > 2 and kwargs is of len 2,
			but does not contain parameters other than 'classRef' 
			or 'roundTo'

		**'attribute 'NClasses' > 2' represents confusionMatrix obj for
		  multi-class classification and 'attribute 'NClasses' == 2' 
		  represents confusionMatrix obj for binary-class classification

		"""
		if self.NClasses == 2:
			if len(kwargs) > 1:
				raise TypeError(f"misclassificationRate() takes 1 to 2 positional arguments but {1+len(kwargs)} were given (in case of binary class confusionMatrix)") from None
			if len(kwargs) == 0:
				return (self.getFN()+self.getFP())/(self.getTN()+self.getTP()+self.getFP()+self.getFN())
			if len(kwargs) == 1:
				if "roundTo" not in kwargs.keys():
					raise ValueError(f"misclassificationRate() takes argument 'roundTo' , but {kwargs.keys()} were given (in case of binary class confusionMatrix)") from  None
				roundTo = kwargs["roundTo"]
				if roundTo is not None:
					if type(roundTo) != int :
						raise TypeError(f"argument 'roundTo' must be int") from None
					if type(roundTo) == int and roundTo < 1:
						raise ValueError(f"argument 'roundTo' must be int > = 1") from None
					return round(((self.getFN()+self.getFP())/(self.getTN()+self.getTP()+self.getFP()+self.getFN())),roundTo)
				if roundTo  is None :
					return ((self.getFN()+self.getFP())/(self.getTN()+self.getTP()+self.getFP()+self.getFN()))
		else:
			if len(kwargs) > 2 or len(kwargs) == 0:
				raise TypeError(f"misclassificationRate() takes 2 to 3 positional arguments but {1+len(kwargs)} were given (in case of multi class confusionMatrix)") from None
			else:
				try:
					classRef = kwargs["classRef"]
				except KeyError:
					raise KeyError(f"misclassificationRate() takes arguments 'classRef' (mandatory) and 'roundTo' (optional), but {kwargs.keys()} were given (in case of multi class confusionMatrix)") from None
				try:
					roundTo = kwargs["roundTo"]
				except KeyError :
					if len(kwargs) == 1:
						roundTo = None
					else:
						raise KeyError(f"misclassificationRate() takes argument 'classRef' and sometimes 'roundTo' , but {kwargs.keys()} were given (in case of multi class confusionMatrix)") from None
				retVal = (self.getFN(classRef)+self.getFP(classRef))/(self.getTN(classRef)+self.getTP(classRef)+self.getFP(classRef)+self.getFN(classRef))
				if roundTo is not None:
					if type(roundTo) != int :
						raise TypeError(f"argument 'roundTo' must be int ") from None
					if type(roundTo) == int and roundTo < 1:
						raise ValueError(f"argument 'roundTo' must be int > = 1") from None
					return round(retVal,roundTo)
				if roundTo  is None :
					return retVal

	def precision(self,**kwargs):
		"""returns class wise precision score of confusionMatrix obj

		precision = TP /( TP + FP )

		Parameters
		----------
		*kwargs: roundTo : int,optional classRef : [int,str],optional
			roundTo : decimal places to round return value 
			classRef : represents required class
			if attribute 'Nclasses' == 2, 
				only one optional parameter roundTo is required
			if attribute 'NClasses' > 2, 
				one paramater 'classRef' is mandatory ,
				parameter 'roundTo' is optional

		Returns
		-------
		float
			precision score of passed class

		Raises
		------
		TypeError
			if attribute 'NClasses' == 2 and more than one 
			parameter is passed
		ValueError
			if attribute 'NClasses' == 2 and parameter other
			than 'roundTo' is passed
		TypeError
			if type of parameter roundTo is not int
		ValueError
			if paramter 'roundTo' is int, but < 1 
		TypeError
			if attribute 'NClasses' > 2 and 0 or more than
			2 parameters are passed
		KeyError
			if attribute 'NClasses' > 2 and parameter 'classRef'
			is not passed in kwargs
		KeyError
			if attribute 'Nclasses' > 2 and kwargs is of len 2,
			but does not contain parameters other than 'classRef' 
			or 'roundTo'

		**'attribute 'NClasses' > 2' represents confusionMatrix obj for
		  multi-class classification and 'attribute 'NClasses' == 2' 
		  represents confusionMatrix obj for binary-class classification

		"""
		if self.NClasses == 2:
			if len(kwargs) > 1:
				raise TypeError(f"precision() takes 1 to 2 positional arguments but {1+len(kwargs)} were given (in case of binary class confusionMatrix)") from None
			if len(kwargs) == 0:
				return self.getTP()/(self.getTP()+self.getFP())
			if len(kwargs) == 1:
				if "roundTo" not in kwargs.keys():
					raise ValueError(f"precision() takes argument 'roundTo' , but {kwargs.keys()} were given (in case of binary class confusionMatrix)") from  None
				roundTo = kwargs["roundTo"]
				if roundTo is not None:
					if type(roundTo) != int :
						raise TypeError(f"argument 'roundTo' must be int") from None
					if type(roundTo) == int and roundTo < 1:
						raise ValueError(f"argument 'roundTo' must be int > = 1") from None
					return round((self.getTP()/(self.getTP()+self.getFP())),roundTo)
				if roundTo  is None :
					return self.getTP()/(self.getTP()+self.getFP())
		else:
			if len(kwargs) > 2 or len(kwargs) == 0:
				raise TypeError(f"precision() takes 2 to 3 positional arguments but {1+len(kwargs)} were given (in case of multi class confusionMatrix)") from None
			else:
				try:
					classRef = kwargs["classRef"]
				except KeyError:
					raise KeyError(f"precision() takes arguments 'classRef' (mandatory) and 'roundTo' (optional), but {kwargs.keys()} were given (in case of multi class confusionMatrix)") from None
				try:
					roundTo = kwargs["roundTo"]
				except KeyError :
					if len(kwargs) == 1:
						roundTo = None
					else:
						raise KeyError(f"precision() takes argument 'classRef' and sometimes 'roundTo' , but {kwargs.keys()} were given (in case of multi class confusionMatrix)") from None
				retVal = self.getTP(classRef)/(self.getTP(classRef)+self.getFP(classRef))
				if roundTo is not None:
					if type(roundTo) != int :
						raise TypeError(f"argument 'roundTo' must be int ") from None
					if type(roundTo) == int and roundTo < 1:
						raise ValueError(f"argument 'roundTo' must be int > = 1") from None
					return round(retVal,roundTo)
				if roundTo  is None :
					return retVal

	def recall(self,**kwargs):
		"""returns class wise recall score of confusionMatrix obj

		recall =  TP /( TP + FN )

		Parameters
		----------
		*kwargs: roundTo : int,optional classRef : [int,str],optional
			roundTo : decimal places to round return value 
			classRef : represents required class
			if attribute 'Nclasses' == 2, 
				only one optional parameter roundTo is required
			if attribute 'NClasses' > 2, 
				one paramater 'classRef' is mandatory ,
				parameter 'roundTo' is optional

		Returns
		-------
		float
			recall score of passed class

		Raises
		------
		TypeError
			if attribute 'NClasses' == 2 and more than one 
			parameter is passed
		ValueError
			if attribute 'NClasses' == 2 and parameter other
			than 'roundTo' is passed
		TypeError
			if type of parameter roundTo is not int
		ValueError
			if paramter 'roundTo' is int, but < 1 
		TypeError
			if attribute 'NClasses' > 2 and 0 or more than
			2 parameters are passed
		KeyError
			if attribute 'NClasses' > 2 and parameter 'classRef'
			is not passed in kwargs
		KeyError
			if attribute 'Nclasses' > 2 and kwargs is of len 2,
			but does not contain parameters other than 'classRef' 
			or 'roundTo'

		**'attribute 'NClasses' > 2' represents confusionMatrix obj for
		  multi-class classification and 'attribute 'NClasses' == 2' 
		  represents confusionMatrix obj for binary-class classification

		"""
		if self.NClasses == 2:
			if len(kwargs) > 1:
				raise TypeError(f"recall() takes 1 to 2 positional arguments but {1+len(kwargs)} were given (in case of binary class confusionMatrix)") from None
			if len(kwargs) == 0:
				return self.getTP()/(self.getTP()+self.getFN())
			if len(kwargs) == 1:
				if "roundTo" not in kwargs.keys():
					raise ValueError(f"recall() takes argument 'roundTo' , but {kwargs.keys()} were given (in case of binary class confusionMatrix)") from  None
				roundTo = kwargs["roundTo"]
				if roundTo is not None:
					if type(roundTo) != int :
						raise TypeError(f"argument 'roundTo' must be int") from None
					if type(roundTo) == int and roundTo < 1:
						raise ValueError(f"argument 'roundTo' must be int > = 1") from None
					return round((self.getTP()/(self.getTP()+self.getFN())),roundTo)
				if roundTo  is None :
					return self.getTP()/(self.getTP()+self.getFN())
		else:
			if len(kwargs) > 2 or len(kwargs) == 0:
				raise TypeError(f"recall() takes 2 to 3 positional arguments but {1+len(kwargs)} were given (in case of multi class confusionMatrix)") from None
			else:
				try:
					classRef = kwargs["classRef"]
				except KeyError:
					raise KeyError(f"recall() takes arguments 'classRef' (mandatory) and 'roundTo' (optional), but {kwargs.keys()} were given (in case of multi class confusionMatrix)") from None
				try:
					roundTo = kwargs["roundTo"]
				except KeyError :
					if len(kwargs) == 1:
						roundTo = None
					else:
						raise KeyError(f"recall() takes argument 'classRef' and sometimes 'roundTo' , but {kwargs.keys()} were given (in case of multi class confusionMatrix)") from None
				retVal = self.getTP(classRef)/(self.getTP(classRef)+self.getFN(classRef))
				if roundTo is not None:
					if type(roundTo) != int :
						raise TypeError(f"argument 'roundTo' must be int ") from None
					if type(roundTo) == int and roundTo < 1:
						raise ValueError(f"argument 'roundTo' must be int > = 1") from None
					return round(retVal,roundTo)
				if roundTo  is None :
					return retVal

	def specificity(self,**kwargs):
		"""returns class wise specificity score of confusionMatrix obj

		specificity = TN / ( TN + FP )

		Parameters
		----------
		*kwargs: roundTo : int,optional classRef : [int,str],optional
			roundTo : decimal places to round return value 
			classRef : represents required class
			if attribute 'Nclasses' == 2, 
				only one optional parameter roundTo is required
			if attribute 'NClasses' > 2, 
				one paramater 'classRef' is mandatory ,
				parameter 'roundTo' is optional

		Returns
		-------
		float
			specificity score of passed class

		Raises
		------
		TypeError
			if attribute 'NClasses' == 2 and more than one 
			parameter is passed
		ValueError
			if attribute 'NClasses' == 2 and parameter other
			than 'roundTo' is passed
		TypeError
			if type of parameter roundTo is not int
		ValueError
			if paramter 'roundTo' is int, but < 1 
		TypeError
			if attribute 'NClasses' > 2 and 0 or more than
			2 parameters are passed
		KeyError
			if attribute 'NClasses' > 2 and parameter 'classRef'
			is not passed in kwargs
		KeyError
			if attribute 'Nclasses' > 2 and kwargs is of len 2,
			but does not contain parameters other than 'classRef' 
			or 'roundTo'

		**'attribute 'NClasses' > 2' represents confusionMatrix obj for
		  multi-class classification and 'attribute 'NClasses' == 2' 
		  represents confusionMatrix obj for binary-class classification

		"""
		if self.NClasses == 2:
			if len(kwargs) > 1:
				raise TypeError(f"specificity() takes 1 to 2 positional arguments but {1+len(kwargs)} were given (in case of binary class confusionMatrix)") from None
			if len(kwargs) == 0:
				return self.getTN()/(self.getTN()+self.getFP())
			if len(kwargs) == 1:
				if "roundTo" not in kwargs.keys():
					raise ValueError(f"specificity() takes argument 'roundTo' , but {kwargs.keys()} were given (in case of binary class confusionMatrix)") from  None
				roundTo = kwargs["roundTo"]
				if roundTo is not None:
					if type(roundTo) != int :
						raise TypeError(f"argument 'roundTo' must be int") from None
					if type(roundTo) == int and roundTo < 1:
						raise ValueError(f"argument 'roundTo' must be int > = 1") from None
					return round((self.getTN()/(self.getTN()+self.getFP())),roundTo)
				if roundTo  is None :
					return self.getTN()/(self.getTN()+self.getFP())
		else:
			if len(kwargs) > 2 or len(kwargs) == 0:
				raise TypeError(f"specificity() takes 2 to 3 positional arguments but {1+len(kwargs)} were given (in case of multi class confusionMatrix)") from None
			else:
				try:
					classRef = kwargs["classRef"]
				except KeyError:
					raise KeyError(f"specificity() takes arguments 'classRef' (mandatory) and 'roundTo' (optional), but {kwargs.keys()} were given (in case of multi class confusionMatrix)") from None
				try:
					roundTo = kwargs["roundTo"]
				except KeyError :
					if len(kwargs) == 1:
						roundTo = None
					else:
						raise KeyError(f"specificity() takes argument 'classRef' and sometimes 'roundTo' , but {kwargs.keys()} were given (in case of multi class confusionMatrix)") from None
				retVal = self.getTN(classRef)/(self.getTN(classRef)+self.getFP(classRef))
				if roundTo is not None:
					if type(roundTo) != int :
						raise TypeError(f"argument 'roundTo' must be int ") from None
					if type(roundTo) == int and roundTo < 1:
						raise ValueError(f"argument 'roundTo' must be int > = 1") from None
					return round(retVal,roundTo)
				if roundTo  is None :
					return retVal

	def F1Score(self,**kwargs):
		"""returns class wise F1Score of confusionMatrix obj

		F1Score = ( 2 * TP ) / ( ( 2 * TP ) + FP + FN )

		Parameters
		----------
		*kwargs: roundTo : int,optional classRef : [int,str],optional
			roundTo : decimal places to round return value 
			classRef : represents required class
			if attribute 'Nclasses' == 2, 
				only one optional parameter roundTo is required
			if attribute 'NClasses' > 2, 
				one paramater 'classRef' is mandatory ,
				parameter 'roundTo' is optional

		Returns
		-------
		float
			F1Score of passed class

		Raises
		------
		TypeError
			if attribute 'NClasses' == 2 and more than one 
			parameter is passed
		ValueError
			if attribute 'NClasses' == 2 and parameter other
			than 'roundTo' is passed
		TypeError
			if type of parameter roundTo is not int
		ValueError
			if paramter 'roundTo' is int, but < 1 
		TypeError
			if attribute 'NClasses' > 2 and 0 or more than
			2 parameters are passed
		KeyError
			if attribute 'NClasses' > 2 and parameter 'classRef'
			is not passed in kwargs
		KeyError
			if attribute 'Nclasses' > 2 and kwargs is of len 2,
			but does not contain parameters other than 'classRef' 
			or 'roundTo'

		**'attribute 'NClasses' > 2' represents confusionMatrix obj for
		  multi-class classification and 'attribute 'NClasses' == 2' 
		  represents confusionMatrix obj for binary-class classification

		"""
		if self.NClasses == 2:
			if len(kwargs) > 1:
				raise TypeError(f"F1Score() takes 1 to 2 positional arguments but {1+len(kwargs)} were given (in case of binary class confusionMatrix)") from None
			if len(kwargs) == 0:
				return (2*self.getTP())/((2*self.getTP())+self.getFP()+self.getFN())
			if len(kwargs) == 1:
				if "roundTo" not in kwargs.keys():
					raise ValueError(f"F1Score() takes argument 'roundTo' , but {kwargs.keys()} were given (in case of binary class confusionMatrix)") from  None
				roundTo = kwargs["roundTo"]
				if roundTo is not None:
					if type(roundTo) != int :
						raise TypeError(f"argument 'roundTo' must be int") from None
					if type(roundTo) == int and roundTo < 1:
						raise ValueError(f"argument 'roundTo' must be int > = 1") from None
					return round(((2*self.getTP())/((2*self.getTP())+self.getFP()+self.getFN())),roundTo)
				if roundTo  is None :
					return (2*self.getTP())/((2*self.getTP())+self.getFP()+self.getFN())
		else:
			if len(kwargs) > 2 or len(kwargs) == 0:
				raise TypeError(f"F1Score() takes 2 to 3 positional arguments but {1+len(kwargs)} were given (in case of multi class confusionMatrix)") from None
			else:
				try:
					classRef = kwargs["classRef"]
				except KeyError:
					raise KeyError(f"F1Score() takes arguments 'classRef' (mandatory) and 'roundTo' (optional), but {kwargs.keys()} were given (in case of multi class confusionMatrix)") from None
				try:
					roundTo = kwargs["roundTo"]
				except KeyError :
					if len(kwargs) == 1:
						roundTo = None
					else:
						raise KeyError(f"F1Score() takes argument 'classRef' and sometimes 'roundTo' , but {kwargs.keys()} were given (in case of multi class confusionMatrix)") from None
				retVal = (2*self.getTP(classRef))/((2*self.getTP(classRef))+self.getFP(classRef)+self.getFN(classRef))
				if roundTo is not None:
					if type(roundTo) != int :
						raise TypeError(f"argument 'roundTo' must be int ") from None
					if type(roundTo) == int and roundTo < 1:
						raise ValueError(f"argument 'roundTo' must be int > = 1") from None
					return round(retVal,roundTo)
				if roundTo  is None :
					return retVal

	def microF1(self,roundTo= None):
		"""returns microF1 score of confsuionMatrix obj

		F1Score = ( 2 * sum(TP) ) / ( ( 2 * sum(TP) ) + sum(FP) + sum(FN) )
		**sum(TP) -> sum of TP values of all classes and same goes for 
		  other values

		Parameters
		----------
		roundTo : int, optional
			rounds return value, by default None
			**setting it to None will not round return value

		Returns
		-------
		float
			microF1 score of confusionMatrix obj

		Raises
		------
		TypeError
			if type of parameter 'roundTo' is not int
		ValueError
			if parameter 'roundTo' is int < 1
		"""
		if roundTo is not None:
			if type(roundTo) != int:
				raise TypeError(f"argument 'roundTo' must be of type int") from None
			if type(roundTo) == int and roundTo < 1:
				raise ValueError(f"argument 'roundTo' must be int >= 1") from None
		if self.NClasses == 2:
			return self.F1Score(roundTo = roundTo)
		if self.NClasses > 2:
			tempTP,tempFP,tempFN = 0,0,0
			for i in range(self.NClasses):
				tempTP += self.getTP(i)
				tempFP += self.getFP(i)
				tempFN += self.getFN(i)
			retVal = (2*tempTP)/((2*tempTP)+tempFP+tempFN) 
			return round(retVal,roundTo) if roundTo is not None else retVal

	def macroF1(self,roundTo= None):
		"""returns macroF1 score of confsuionMatrix obj

		macroF1 = sum(F1Score)/NClasses
		**sum(F1Score) -> sum of F1Score for all classes and
		  NClasses -> attribute 'NClasses' of confusionMatrix
		  obj

		Parameters
		----------
		roundTo : int, optional
			rounds return value, by default None
			**setting it to None will not round return value

		Returns
		-------
		float
			macroF1 score of confusionMatrix obj

		Raises
		------
		TypeError
			if type of parameter 'roundTo' is not int
		ValueError
			if parameter 'roundTo' is int < 1
		"""
		if roundTo is not None:
			if type(roundTo) != int:
				raise TypeError(f"argument 'roundTo' must be of type int") from None
			if type(roundTo) == int and roundTo < 1:
				raise ValueError(f"argument 'roundTo' must be int >= 1") from None
		if self.NClasses == 2:
			return self.F1Score(roundTo = roundTo)
		if self.NClasses > 2:
			retVal = (sum([self.F1Score(classRef = i) for i in range(self.NClasses)]))/self.NClasses
			return round(retVal,roundTo) if roundTo is not None else retVal

	def weightedF1(self,roundTo= None):
		"""returns weightedF1 score of confsuionMatrix obj

		weightedF1 = sum(F1Score * classCount)/sum(classCount)
		**classCount -> column sum of attribute 'matrix' and
		  sum(F1Score * classCount) sum of (F1SCore * classcount)
		  for each class in confusionMatrix and 
		  sum(classCount) -> sum of classCount for each class of 
		  confusionMatrix obj

		Parameters
		----------
		roundTo : int, optional
			rounds return value, by default None
			**setting it to None will not round return value

		Returns
		-------
		float
			weightedF1 score of confusionMatrix obj

		Raises
		------
		TypeError
			if type of parameter 'roundTo' is not int
		ValueError
			if parameter 'roundTo' is int < 1
		"""
		if roundTo is not None:
			if type(roundTo) != int:
				raise TypeError(f"argument 'roundTo' must be of type int") from None
			if type(roundTo) == int and roundTo < 1:
				raise ValueError(f"argument 'roundTo' must be int >= 1") from None
		if self.NClasses == 2:
			return self.F1Score(roundTo = roundTo)
		if self.NClasses> 2:
			num,denom = 0,0
			for i in range(self.NClasses):
				classCount = self.matrix[:,i].sum(axis=0)
				num += (self.F1Score(classRef = i)*classCount)
				denom  += classCount
			return (num/denom) if roundTo == None else round((num/denom),roundTo)

	def report(self,roundTo = None,format = 'str'):
		"""returns string/pandas.DataFrame reportr of confusionMatrix obj

		returns string containing representation of attribute 'matrix'
		of confusionMatrix followed by all different class wise 
		characteristic value of the confusionMatrix obj or returns only 
		a pd.Dataframe containing all the class wise characteristic 
		values of the confusionMatrix obj

		**'class wise characteristic values' -> TP, TN, FP, FN, accuracy, 
		precision, misclassificationRate, recall, specificity, F1Score 

		Parameters
		----------
		roundTo : int, optional
			rounds return vlaue, by default None
			**setting it to None will not round return value

		format : str, optional
			specifies format of return value, it's vlaues can 
			be 'str' or 'dataframe', by default 'str'

		Returns
		-------
		str
			string of attribute 'matrix' followed by all class
			wise charateristic values, if value of parameter
			'format' is 'str'
		pandas.DataFrame
			pandas.DataFrame of all class wise charateristic 
			values, if  value of parameter 'format' is 
			'dataframe'

		Raises
		------
		TypeError
			if type of parameter 'roundTo' is not int
		ValueError
			if parameter 'roundTo' is int < 1
		ValueError
			if parameter is not in ['str','dataframe']

		"""
		if roundTo is not None:
			if type(roundTo) != int:
				raise TypeError(f"argument 'roundTo' must be of type int") from None
			if type(roundTo) == int and roundTo < 1:
				raise ValueError(f"argument 'roundTo' must be int >= 1") from None
		if format not in ['dataframe','str']:
			raise ValueError(f"{format} is not a valid value for format, supported vlaues are 'str' , 'dataframe'") from None
		metrics1 = [self.getTP,self.getTN,self.getFP,self.getFN]
		metrics2 = [self.accuracy,self.precision,self.misclassificationRate,self.recall,self.specificity,self.F1Score]
		df = pd.DataFrame()
		df["metrics"] = ["TP","TN","FP","FN"] +[func.__name__ for func in metrics2 ]
		if self.NClasses == 2:
			df[""] = [func() for func in metrics1] + [func(roundTo=roundTo) for func in metrics2]
		if self.NClasses > 2:
			for className in self.classNames:
				df[f"{className}"] = [func(className) for func in metrics1] + [func(classRef = className,roundTo=roundTo) for func in metrics2]
		#df.reset_index(drop=True, inplace=True)
		return df if format != 'str' else self.toString() + "\n\n\n" + df.to_string(index=False)
		
class NueronLayer:
	"""
	class for creating single layer of nuerons

	...

	Attributes
	----------

	layer : np.ndarray
		array containing activation values of nuerons
	bias : [float,int]
		value of bias of layer
	includeBias : bool
		True if layer has a bias otherwise False
	width : int
		number of nuerons in layer

	Methods
	-------
	setBias(value: [int,float])
		sets layer's Bias to parameter 'value'

	** for detailed explanation see help(FrontalLobe.NueronLayer.<method Name/generator Name>)
	The docstring examples assume that `FrontalLobe` has been imported as `FrontalLobe

	"""

	def __init__(self,width,includeBias=True):
		self.layer = np.zeros((width,1),dtype=float)
		self.bias = self.layer[0,0] if includeBias == True else None
		self.layer[0,0] = 1.0 if includeBias == True else 0
		self.includeBias = includeBias
		self.width = len(self.layer)

	def __repr__(self):
		return f"FrontalLobe.NueronLayer(width={len(self.layer)},includeBias={self.includeBias})"

	def __str__(self):
		return f"<class NueronLayer> includeBias = {self.includeBias} {self.layer.__str__()}"

	def setBias(self,value):
		"""sets bias of layer to parameter 'value'

		Parameters
		----------
		value : [int,flaot]
			parameter to set bias of layer

		Raises
		------
		AttributeError
			if layer does not contain bias nueron
		ValueError
			if type of parameter 'value' is not in [int,float]
		"""
		if self.includeBias == False:
			raise AttributeError(f"layer does not contain bias")
		else:
			if type(value) not in [int,float]:
				raise ValueError(f"argument 'value' must be of type int or float")
			self.layer[0][0] = value

	def __getitem__(self,index):
		try:
			return self.layer[index][0]
		except IndexError :
			raise IndexError("unit index out of range") from None

	def __setitem__(self,index,value):
		if type(index) == int:
			try:
				self.layer[index][0] = value
			except IndexError:
				raise IndexError("unit index out of range") from None
		elif index == "all":
			if type(value) != np.ndarray:
				raise TypeError("input value must be np.ndarray") from None
			reqShape = self.layer.shape if self.includeBias == False else (self.layer.shape[0] -1,1)
			if value.shape != reqShape:
				raise ValueError(f"could not broadcast input array from shape {value.shape} into shape {reqShape}")
			if self.includeBias == True:
				self.layer[1:,:] = value
			if self.includeBias == False :
				self.layer = value
		else:
			raise TypeError("unitIndex must be int or str ('all') only")

	def __iter__(self):
		self.start = 0
		return self

	def __next__(self):
		self.start += 1
		try:
			return self.layer[self.start - 1,0]
		except IndexError:
			raise StopIteration

	def __len__(self):
		return self.width

	def __round__(self,precision=None):
		if precision is not None:
			for i in range(self.width):
				self.__setitem__(i,round(self.__getitem__(i),precision))

class NueralNetwork:
	
	def __init__(self,architecture,HLAF="sigmoid",OLAF= "sigmoid",precision = 4):
		"""[summary]

		Parameters
		----------
		architecture : [int,list]
			specifies architecture of nueralNetwork obj
		HLAF : [str,int] optional
			sctivation function for hidden layers
			supported values are: 
			str value : 'sigmoid'  or int : 1
			str value : 'ReLU' or int : 2  
			str value : 'tanH' or int : 3 
			str value : 'linear' or int : 4, by default 'sigmoid'
		OLAF : str, optional
			activation function for output layer
			supported values are:
			str value : 'sigmoid' or int : 1 
			str value : 'tanH' or int : 2, by default 'sigmoid'
		precision : int, optional
			specifies decimal paces for calculations and 
			activation values , by default 4
			**cost values of the NueralNetwork obj are not
			  rounded 

		Raises
		------
		TypeError
			if type of parameter 'architecture' is other than int or list
		TypeError
			if atleat one of the member of the parameter 'architecture' is 
			not of type tuple, int or a NueronLayer obj
		TypeError
			if parameter 'precision' is not of type int
		ValueError
			if parameter 'precision; is ot type int, but less < 1
		ValueError
			if passed value of parameter 'HLAF'  or 'OLAF' is not 
			in thier respective supported values 

		"""
		self.architecture = []
		if type(architecture) not in [int, list,NueronLayer] :
			raise TypeError("argument 'architecture' must be int, list or NueronLayer object")
		architecture = [architecture] if type(architecture) == int else architecture
		for layerData in architecture :
			if type(layerData) in [int,tuple,NueronLayer] :
				if type(layerData) == int : 
					includeBias = True
					self.architecture.append(NueronLayer(layerData,includeBias))
				if type(layerData) == tuple:
					includeBias = layerData[1]
					self.architecture.append(NueronLayer(layerData[0],includeBias))
				if type(layerData) == NueronLayer :
					self.architecture.append(layerData)
			else:
				raise TypeError("argument 'architecture' must ony contain data of type <class int> , <class tuple> or <class NueronLayer>")
		if precision is not None:
			if type(precision) != int:
				raise TypeError("argument 'precision' must be int")
			if type(precision) == int and precision <= 1:
				raise ValueError("argument 'precision' must be int >= 1")
		self.precision = precision
		self.activationFuncs = [self.sigmoid,self.ReLU,self.tanH,self.linear]
		AAFA = [funcs.__name__ for funcs in self.activationFuncs] + list(range(1,5))
		if HLAF not in AAFA or OLAF not in ["sigmoid","tanH",1,2]:
			errorArg,errorVal,validArgs  = ("HLAF",HLAF,"'sigmoid' or 1, 'ReLU' or 2, 'tanH' or 3, 'linear' or 4") if HLAF not in AAFA else ("OLAF",OLAF,"'sigmoid' or 1, 'tanH' or 2")
			raise ValueError(f"{errorVal} is not a valid value for {errorArg}, supported values are {validArgs}") 
		self.weights,self.DVec = rolledVector(),None
		AFs = [self.sigmoid,self.ReLU,self.tanH]
		self.HLAF = self.activationFuncs[AAFA.index(HLAF)] if type(HLAF) == str else self.activationFuncs[HLAF - 1] 
		self.OLAF = (self.sigmoid if OLAF == "sigmoid" else self.tanH) if type(OLAF) == str else [self.sigmoid,self.tanH][OLAF - 1]
		self.X,self.Y,self.XTest,self.YTest,self.trainingSize = None,None,None,None,None
		if len(self.architecture) >= 1:
			self.width,self.depth = self.getDim() 
		self.cost,self.__preTrained,self.classes = None,False,None

	def __repr__(self):
		return f"FrontalLobe.NueralNetwork(architecture={self.architecture.__repr__()})"

	def toString(self,printUnderline = True):
		"""return string representation of NueralNetwork obj

		returns string representation only if the number if 
		the NueralNetwork obj contains atleast one layer 
		otherwise return None

		Parameters
		----------
		printUnderline : bool, optional
			if True adds underline to the layer heads in the 
			string representation of NueralNetwork obj, by 
			default True

		Returns
		-------
		str
			string represeentation of NueralNetwork obj

		"""
		if len(self.architecture) >= 1 :
			maxRows = max(self.architecture,key = lambda layer: layer.width).width
			temp = [[" " for i in range(len(self.architecture))] for j in range((2*maxRows)-1)]
			for j in range(len(temp[0])):
				ind = maxRows - self.architecture[j].width
				for i in self.__getitem__(j+1)[:,0]:
					(temp[ind])[j] = i 
					ind +=  2
			table =  tabulate.tabulate(temp,headers=["input Layer"] + [f"hiddenLayer {i+1}" for i in range(len(temp[0]) - 2)] +["output Layer"])
			if printUnderline == False :
				temp2 = table.split("\n")
				temp2.pop(1)
				return "\n".join(temp2)
			return table
		else:
			return None

	def __str__(self):
		return self.toString()

	def __getitem__(self,pos):
		try:
			(layerNumber,unitNumber) = pos
		except TypeError:
			layerNumber = pos
			unitNumber=None
		if unitNumber == None:
			try:
				return self.architecture[layerNumber - 1].layer
			except IndexError:
				raise IndexError("layer index out of range") from None
		try:
			return self.architecture[layerNumber - 1][unitNumber]
		except IndexError:
			raise IndexError("layer index out of range") from None

	def __setitem__(self,pos,value):
		try:
			(layerNumber,unitNumber) = pos
		except TypeError:
			layerNumber = pos
			unitNumber = None
		if unitNumber == None:
			try:
				if type(value) in [str,int,float]:
					raise ValueError("layer can be of type 'numpy.ndarray' or 'list' only!") from None
				else:
					if type(value) == list:
						for i in value:
							if type(i) in [list,np.ndarray]:
								raise ValueError("layer can not multi dimensional list")
							if type(i) == str :
								raise ValueError("layer can not contain values of <class str>")
						if len(value) != self.__getitem__(layerNumber).size :
							raise ValueError(f"could not broadcast input list from len {len(value)} into shape {self.__getitem__(layerNumber).shape}") from None
						else:
							value = np.array(value).reshape(self.__getitem__(layerNumber).shape)
							self.architecture[layerNumber].layer = value
					if type(value) == np.ndarray:
						if value.shape != self.__getitem__(layerNumber).shape:
							raise ValueError(f"could not broadcast input array from shape {value.shape} into shape {self.__getitem__(layerNumber).shape}") from None
						else:
							self.architecture[layerNumber - 1].layer = value
			except IndexError:
				raise IndexError("layer index out of range") from None
		else:
			layer = self.__getitem__(layerNumber)
			try:
				layer[unitNumber] = value
			except IndexError:
				raise IndexError("unit index out of range") from None

	def __iter__(self):
		self.start = 0
		return self

	def __next__(self):
		self.start += 1
		try:
			return self.architecture[self.start-1]
		except IndexError :
			raise StopIteration

	def __len__(self):
		return len(self.architecture)

	#layer = lambda self,layerNumber : self.architecture[layerNumber - 1]

	def layer(self,layerNumber):
		"""return NueronLayer at index 'layerNumber'

		Parameters
		----------
		layerNumber : int
			index of layer

		Returns
		-------
		NueronLayer
			layer at index 'layerNumber'

		Raises
		------
		TypeError
			if type of parameter 'layerNumber'
			is not int
		IndexError
			if parameter 'layerNumber' is < 1 or greater 
			than the number of layers in the NueralNetwork 
			obj

		"""
		if type(layerNumber) != int:
			raise TypeError("type of argument 'layerNumber' must be int")
		if layerNumber <= 0 or layerNumber > len(self.architecture) :
			raise IndexError("passed index argument 'layerNumber' is out of bound")
		return self.architecture[layerNumber - 1]

	def getDim(self,includeIL = False): 
		"""returns dimensions of NueralNetwork obj

		returns tuple containing width of NueronLayer with maximum 
		width and depth (no. of layers in NueralNetwork obj)
		**only considers the input layer for depth, if parameter
		  'includeIL' is True

		Parameters
		----------
		includeIL : bool, optional
			specifies to include to input layer for depth, by 
			default False

		Returns
		-------
		tuple : (int,int)
			returns (width of layer with maximum width, depth )

		Raises
		------
		TypeError
			if parameter 'TypeError' is not a bool Value

		"""
		if includeIL not in [False,True]:
			raise TypeError("argument 'includeIL' can only be bool value")
		return max(self.architecture,key= lambda layer: layer.width).width,(len(self.architecture) - 1 if includeIL == False else len(self.architecture)) 

	def getTrainingStatus(self):
		""" returns True if the NuerralNetwork obj is trained atleast once

		Returns
		-------
		bool
			return True if the NueralNetwork obj \
			has been trained for atkeast once

		"""
		return self.__preTrained

	def append(self,layerData) :
		"""adds new layer at the end of a non pre-trained NueralNetwork obj

		**new layer can only be added if the NueralNetwork obj is not already 
		trained atleast once 

		Parameters
		----------
		layerData : int, NueronLyer obj, tuple
			parameters of layer to append

		Raises
		------
		TypeError
			if tuple parameter 'layerData' contains member ot type
			other than (int,bool)
		TypeError
			if type of paramater 'layerData' is not in [int,bool]
		AttributeError
			if NueralNetwork obj is already trained 
			
		"""
		if self.__preTrained == False:
			if type(layerData) == int:
				self.architecture.append(NueronLayer(layerData))
			if type(layerData) == NueronLayer :
				self.architecture.append(layerData)
			if type(layerData) == tuple:
				if type(layerData[0]) == int and type(layerData[1]) == bool :
					self.architecture.append(NueronLayer(layerData[0],layerData[1]))
				else:
					raise TypeError(f"tuple argument 'layerData' must only contain int and bool values")
			self.width,self.depth = self.getDim()
			if type(args[0]) not in [int ,NueronLayer,tuple] :
				raise TypeError("append() takes exactly one tuple (int,bool)  or pair of int,bool or one NueronLayer") from None
		else:
			raise AttributeError("can not add new layer to a pre-trained NueralNetwork model")

	def pop(self,layerNumber):
		"""removes and returns layer of NueralNetwork obj at index 'layerrNumber'.

		removes and returns NueronLayer of the NueralNetwork at
		value of passed parameter 'layerNumber'
		**layer can only be poped if the NueralNetwork obj 
		  is not already trained atleast once aand it will also
		  not be poped if 'weights' attribute of the NueralNetwork 
		  is already initialized

		Parameters
		----------
		layerNumber : int
			index of layer to be poped

		Returns
		-------
		NueronLayer
			layer at index 'layerNumber'

		Raises
		------
		AttributeError
			if 'weights' attribute of the NueralNeetwork is already
			initialized
		IndexError
			if parameter 'layerNumber' is < 1 or greater 
			than the number of layers in the NueralNetwork 
			obj
		AttributeError
			if NueralNetwork obj is already trained atleast once

		"""
		if self.__preTrained == False:
			if layerNumber >= 1 and layerNumber <= self.depth :
				if len(self.weights) == self.depth:
					raise AttributeError("weights already initialized! can not remove layer after weight initialization") from None
				layerNumber -= 1
				temp = self.architecture[layerNumber]
				self.architecture = self.architecture[:layerNumber] + self.architecture[layerNumber+1: ]
				self.width,self.depth = self.getDim()
				return temp
			else:
				raise IndexError("passed index argument 'layerNumber' is out of bound") from None
		else:
			raise AttributeError("can not pop alayer from a pre-trained NueralNetwork model")

	def resetLayers(self,resetVal = 0,resetBiases = True):
		"""resets activation values of nuerons of NueralNetwork obj

		sets acctivation vlauea of all nuerons of Nueral netowk
		obj to value of parameter 'resetVal' except biases
		**reser biases only if the parameter 'resetBiases' is 
		  True

		Parameters
		----------
		resetVal : int, optional
			sets activation value of all nuerons, by default 0
		resetBiases : bool, optional
			True if reset of biases are required otherwise 
			False, by default True

		Raises
		------
		TypeError
			if type of parameter 'resetVal' is not in [int,float]
		TypeError
			if passed value for parameter 'resetBiases' is not bool

		"""
		if type(resetVal) not in [int,float]:
			raise TypeError("argument 'resetVal' can only be int or float")
		if resetBiases not in [True,False]:
			raise TypeError("argument 'resetBiases' can only be bool value")
		for i in self.architecture:
			if resetBiases == False :
				i["all"] = np.full((i.width,1) if i.includeBias == False else (i.width -1 ,1),resetVal,dtype=float)
			else:
				i.layer = np.full((i.width,1),resetVal,dtype=float)

	def resetWeights(self,resetVal = None):
		"""resets attribute 'weights' of the NueralNerwork obj

		resets weights of NueralNetwork obj to value of
		parameter 'resetVal'

		Parameters
		----------
		resetVal : [rolledVector,skeletonRolledVector,None], optional
			valure to reset weights of a NueralNetwork, by default None
			**if parameter'resetVal' == None , then the attribute 
			  'weights' is set to  a rolledVector obj

		Raises
		------
		TypeError
			if parameter 'resetVal' is not in [rolledVector,
			skeletonRolledVector,None]

		"""
		if resetVal is not None:
			if type(resetVal) in [rolledVector,skeletonRolledVector]:
				raise TypeError("argument 'resetVal' can only be rolledVector, skeletonVector obj  or keyword 'None'")
		if resetVal is not None :
			self.weights.update(resetVal)
		if resetVal == None:
			self.weights = rolledVector()

	def dump(self,fileName,trainingStatus=True):
		"""saves NueralNetwork obj locally

		saves binary image of the NueralNetwork obj 
		locally

		Parameters
		----------
		fileName : str
			name of binary image file of NueralNetwork 
			obj
			**it must contain the directory name if the
			  the binary file is to be saved is in a 
			  directory other than current working directory
		trainingStatus : bool
			if True saves the NueralNetwork obj as a pre-trained 
			model other wise saves it as a non pretrained model,
			by default False

		Raises
		------
		TypeError
			if tyep of parameter 'trainingStatus'
			is not bool
		FileNotFoundError
			if no such directory exists as passed
			parameter 'fileName'
		
		"""
		if type(trainingStatus) != bool :
			raise TypeError("type of argument 'trainingStatus' must be int")
		with open(fileName,"wb+") as wf:
			temp = self.copy()
			temp.__preTrained = self.__preTrained if trainingStatus == True else False
			pickle.dump(temp,wf)
		temp=None

	@staticmethod
	def load(fileName):
		"""loads a NueralNetwork obj image from local directory

		loads a NueralNetwork image form a local file 
		and retuens a NueralNetwork obj


		Parameters
		----------
		fileName : str
			name of binary image file of NueralNetwork 
			obj
			**it must contain the directory name if the
			  the binary file is to be loaded is in a 
			  directory other than current working directory

		Returns
		-------
		NueralNetwork obj
			NueralNetwork obj of the binary image file 
			at passed parameter 'fileName' location

		Raises
		------
		FileNotFoundError
			if no such directory or file exists as passed
			parameter 'fileName'
		"""
		with open(fileName,"rb") as rf:
			return pickle.load(rf)

	def copy(self):
		"""returns a copy of the nueralNetwork obj

		Returns
		-------
		NueralNetwork obj
			copy NueralNetwork obj 
		"""
		retVal = NueralNetwork([(i.width,i.includeBias) for i in self],HLAF=self.HLAF.__name__,OLAF=self.OLAF.__name__,precision=self.precision)
		if self.weights.isEmpty() == False:
			retVal.weights = self.weights.copy()
		return retVal

	def setBiases(self,biasVal):
		"""set biases of all layera of the NueralNetwork obj

		Parameters
		----------
		biasVal : [int,float,numpy.ndarray,list]
			value to set biases of layers of the NueralNetwork obj
			**if type (biasVal) is in [int,float], then all biases 
			  of layers are set to same value of parameter 'biasVal'
			  if type(biasVal) is in [list,numpy.ndarray], then 
			  all biases are set index wise and parameter 'biasval'
			  must be single dimensional, if they are to be passed 
			  as a list or a numpy.ndarray

		Raises
		------
		ValueError
			if type of parameter 'biasVal' is in [list,numpy.ndarray]
			and no. of member elemnts is less then the no. of layers 
			with baises in NueralNetwork obj
		TypeError
			if type of parameter 'biasVal' is in [list,numpy.ndarray]
			and type of atleast one of the member elements of parmaeter 
			'biasVal' is not in [int,float]
		TypeError
			if parameter 'biasVal' is multi-dimensional list or
			numpy.ndarray
			
		Warns
		-----
		if the no. of memeber elements is greater than than the
		no. of layers with bias in NueralNetwork obj 
		**even though the warning is raised, but the first 'x'
		  members are used to set bias of layers with boias in
		  NueralNetwork obj
		  'x' : no .of layers with bias in the NueralNetwork Obj

		"""
		if (type(biasVal) == list and np.array(biasVal).ndim == 1) or (type(biasVal) == np.ndarray and biasVal.ndim == 1) or (type(biasVal) in[int,float]):
			reqBiases = sum([1 if layer.includeBias == True else 0 for layer in self.__iter__() ])
			if type(biasVal) not in [int,float] :
				biasVal = list(biasVal)
				if len(biasVal) < reqBiases:
					raise ValueError("argument 'biasVal' underflow! , number of layers with bias is greater than the the size of argument 'biasVal' passed") from None
				else:
					for i in biasVal :
						if type(i) not in [int,float]:
							raise TypeError("argument 'biasVal' must only contain int,float values") from None
					for layer in self.__iter__():
						if layer.includeBias == True: 
							layer[0] = biasVal.pop(0)
					if len(biasVal) > 0 :
						warnings.warn("size of passed argument 'biasVal' is greater than number of layers with bias")
			else:
				for layer in self.__iter__() :
					if layer.includeBias == True:
						layer[0] = biasVal
		else:
			raise TypeError("argument 'biasVal' can only be a single dimensional list or numpy.ndarray") from None

	def layerNormalization(self,layerNumber,normalizeBias = False):
		"""normalizes layer of the NueralNetwork obj

		normalizes layer of the NueralNetwork obj at 
		index 'layerNumber'

		Parameters
		----------
		layerNumber : int
			index of layer to be normalized
		normalizeBias : bool, optional
			if True, the biases of the layer is also 
			included in normalization, by default False

		Raises
		------
		TypeError
			if type of parameter 'layerNumber' is not int
		IndexError
			if parameter 'layerNumber' is < 1 or greater 
			than the number of layers in the NueralNetwork 
			obj
		TypeError
			if type of parameter 'layerNumber' is not bool

		"""
		if type(layerNumber) != int :
			raise TypeError("type of argument 'layerNumber' must int")
		if layerNumber < 1 or layerNumber > len(self.architecture) + 1 :
			raise IndexError("passed index argument 'layerNumber' is out of bound")
		if type(normalizeBias) != bool :
			raise TypeError("type of argument 'layerNumber' must be bool")
		if normalizeBias == False:
			layerMax = np.max(self.layer(layerNumber).layer[(1 if self.layer(layerNumber).includeBias == True else 0):,:])
			layerMin = np.min(self.layer(layerNumber).layer[(1 if self.layer(layerNumber).includeBias == True else 0):,:])
			self.layer(layerNumber)["all"] = (self.layer(layerNumber).layer[(1 if self.layer(layerNumber).includeBias == True else 0):,:] - layerMin)/(layerMax-layerMin)
		if normalizeBias == True:
			layerMax = np.max(self.layer(layerNumber).layer)
			layerMin = np.max(self.layer(layerNumber).layer)
			self.layer(layerNumber)["all"] = (self.layer(layerNumber).layer[(1 if self.layer(layerNumber).includeBias == True else 0):,:] - layerMin)/(layerMax-layerMin)
		round(self.layer(layerNumber),self.precision) if self.precision is not None else None

	def getWeightShape(self,layerNumber):
		"""returns shape of weight matrix for forward propagation

		return the shape of weight matrix required to forward 
		propagate from layer index 'layerNumber' to layer
		index 'layerNumber' + 1

		Parameters
		----------
		layerNumber : int
			index of layer 

		Returns
		-------
		tuple : (int,int)
			required shape of the weight matrix

		Raises
		------
		IndexError
			if parameter 'layerNumber' is < 1 or greater 
			than the number of layers in the NueralNetwork 
			obj
			
		"""
		if layerNumber >= 1  and layerNumber <= self.depth:
			return (self.architecture[layerNumber].width - (0 if self.architecture[layerNumber].includeBias == False else 1),self.architecture[layerNumber-1].width)
		else:
			raise IndexError("passed index argument 'layerNumber' is out of bound")

	def RWInitialization(self,e,weightType=float):
		"""random weight initialization 

		initializes all weight matrix (stored in 
		attribute 'weights') for all layers to
		random int/float values between value of
		parameter 'e' to its negative.(e to -e)

		Parameters
		----------
		e : int,float
			limit for random weight initialization
		weightType : int,float, optional
			specifies the type of memebers of attribute
			'weights',supported values are int (not 'int'
			or any other int values) and float (not 'float'
			or any other float value), by default float

		Raises
		------
		TypeError
			if type of parameter 'e' is not in [int,float]
		TypeError
			if passed value for parameter 'weightType' is not
			in [int,float]

		"""
		if type(e) not in [int,float]:
			raise TypeError("argument 'e' can only be int or float")
		if weightType not in [int,float]:
			raise TypeError(F"{weightType} is not a valid value for weightType, supported types are 'int', 'float'")
		lowerLimit,upperLimit = min(-1*e,e),max(-1*e,e)
		temp = rolledVector()
		for layerIndex in range(1,self.depth + 1):
			weightShape = self.getWeightShape(layerIndex)
			temp.addMatrix(np.random.uniform(lowerLimit,upperLimit,weightShape) if weightType == float else np.random.randint(lowerLimit,upperLimit,weightShape))
		self.weights = temp
		if self.precision is not None:
			round(self.weights,self.precision)

	def sigmoid(self,layer):
		"""sigmoid activation function

		Parameters
		----------
		layer : numpy.ndarray
			values to apply sigmoid activation 
			function 

		Returns
		-------
		numpy.ndarray
			sigmoid values for passed parameter
			'layer'
			
		Raises
		------
		TypeError
			if type of parameter 'layer' is not 
			numpy.ndarray

		"""
		if type(layer) != np.ndarray :
			raise TypeError("argument 'layer' must be of type numpy.ndarray")
		return np.vectorize(lambda x: round(1/(1+np.exp(-1*round(x,self.precision + 1))),self.precision) if self.precision is not None else 1/(1+np.exp(-1*x)))(layer)
	
	def derivativeSigmoid(self,layer):
		"""return derivative of sigmoid values

		Parameters
		----------
		layer : numpy.ndarray
			sigmoid values to calculate 
			derivatives

		Returns
		-------
		numpy.ndarray
			derivative values of passed 
			sigmoid values parameter 
			'layer'

		Raises
		------
		TypeError
			if type of parameter 'layer' is not 
			numpy.ndarray

		"""
		if type(layer) != np.ndarray :
			raise TypeError("argument 'layer' must be of type numpy.ndarray")
		return np.multiply(layer,1-layer)

	def ReLU(self,layer):
		"""ReLU activation function

		ReLU -> rectified linear unit

		Parameters
		----------
		layer : numpy.ndarray
			values to apply ReLU activation 
			function 

		Returns
		-------
		numpy.ndarray
			ReLU values for passed parameter
			'layer'
			
		Raises
		------
		TypeError
			if type of parameter 'layer' is not 
			numpy.ndarray 

		"""
		if type(layer) != np.ndarray :
			raise TypeError("argument 'layer' must be of type numpy.ndarray")
		return np.vectorize(lambda x: round(max(0,x),self.precision) if self.precision is not None else max(0,x))(layer)

	def derivativeReLU(self,layer):
		"""return derivative of ReLU values

		Parameters
		----------
		layer : numpy.ndarray
			ReLU values to calculate 
			derivatives

		Returns
		-------
		numpy.ndarray
			derivative values of passed 
			ReLU values parameter 
			'layer'

		Raises
		------
		TypeError
			if type of parameter 'layer' is not 
			numpy.ndarray
			
		"""
		if type(layer) != np.ndarray :
			raise TypeError("argument 'layer' must be of type numpy.ndarray")
		return np.vectorize(lambda x: 0 if x<= 0 else 1)(layer)

	def tanH(self,layer):
		"""tanH activation function

		tanH -> hyperbolic tangent 

		Parameters
		----------
		layer : numpy.ndarray
			values to apply tanH activation 
			function 

		Returns
		-------
		numpy.ndarray
			tanH values for passed parameter
			'layer'
			
		Raises
		------
		TypeError
			if type of parameter 'layer' is not 
			numpy.ndarray 

		"""
		if type(layer) != np.ndarray :
			raise TypeError("argument 'layer' must be of type numpy.ndarray")
		expTerm = lambda x:round(np.exp(round(x,self.precision + 1))**2,self.precision + 1) if self.precision is not None else (np.exp(x))**2
		return np.vectorize(lambda x: round((expTerm(x)-1)/(expTerm(x) + 1),self.precision) if self.precision is not None else (expTerm(x)-1)/(expTerm(x) + 1))(layer)

	def derivativetanH(self,layer):
		"""return derivative of tanH values

		Parameters
		----------
		layer : numpy.ndarray
			tanH values to calculate 
			derivatives

		Returns
		-------
		numpy.ndarray
			derivative values of passed 
			tanH values parameter 'layer'

		Raises
		------
		TypeError
			if type of parameter 'layer' is not 
			numpy.ndarray
			
		"""
		if type(layer) != np.ndarray :
			raise TypeError("argument 'layer' must be of type numpy.ndarray")
		return (1 - (layer**2))

	def linear(self,layer):
		"""linear activation function

		Parameters
		----------
		layer : numpy.ndarray
			values to apply linear activation 
			function 

		Returns
		-------
		numpy.ndarray
			linear values for passed parameter
			'layer'
			
		Raises
		------
		TypeError
			if type of parameter 'layer' is not 
			numpy.ndarray 

		"""
		if type(layer) != np.ndarray :
			raise TypeError("argument 'layer' must be of type numpy.ndarray")
		return np.vectorize(lambda x:round(x,self.precision) if self.precision is not None else x)(layer)

	def derivativelinear(self,layer):
		"""return derivative of linear values

		Parameters
		----------
		layer : numpy.ndarray
			linear values to calculate 
			derivatives

		Returns
		-------
		numpy.ndarray
			derivative values of passed 
			linear values parameter 
			'layer'

		Raises
		------
		TypeError
			if type of parameter 'layer' is not 
			numpy.ndarray
			
		"""
		if type(layer) != np.ndarray :
			raise TypeError("argument 'layer' must be of type numpy.ndarray")
		return np.ones(layer.shape)

	def predict(self,x,normalizeInput = True,resetNetwork = True):
		"""predicts on 'x' based on current training of NueralNetwork obj

		Parameters
		----------
		x : numpy.ndarray
			input for NueralNetwork to predict
		normalizeInput : bool, optional
			if True normalizes input of the
			NueralNetwork obj, by default True
		resetNetwork : bool, optional
			if True resets NueralNetwork after 
			prediction, by default True

		Returns
		-------
		numpy.ndarray
			predicted values on parameter 'x'

		Raises
		------
		AttributeError
			if attribute 'weights' of the NUeralNetwork is 
			empty
		TypeError
			if type of parameter 'x' is not numpy.ndarray
		TypeError
			if either type of parameter 'normalizeInput' or
			'resetNetwork' is not bool 
		TypeError
			if ndim of parameter is == 1 or > 3
		ValueError
			if shape of parameter 'x' (if'x' is 2 dimensional)
			or shape of member array of 'x' (if 'x' is 3 
			dimensional) is not equal to the reqiured input
			array shape of the NUeralNetwork obj

		"""
		if self.weights.isEmpty == True:
			raise AttributeError("attribute 'weights' is empty, can not predict on basis of emty 'weights' attribute")
		if type(x)!= np.ndarray:
			raise TypeError("argument 'x' can only be numpy.ndarray")
		if normalizeInput not in [True,False] or resetNetwork not in [True,False]:
			errorArg = "normalizeInput"  if normalizeInput not in [True,False] else "resetNetwork"
			raise TypeError(f"argument '{errorArg}' can only be bool Value")
		inputShape = (self.architecture[0].width - (1 if self.architecture[0].includeBias == True else 0),1)
		if x.ndim not in [2,3] :
			raise TypeError("argument 'x' can only be 2 or 3 dimensional numpy.ndarray")
		if x.ndim == 2:
			if x.shape != inputShape :
				raise ValueError("shape of argument 'x' is not compatible with input of NUeralNetwork")
			self.__SUFP(x,normalizeInput)
			retVal = self.labelExtractor()
			self.resetLayer(resetBiases = False) if resetNetwork == True else None
			return retVal
		if x.ndim == 3:
			retVal =  np.array([self.predict(inpt,normalizeInput,resetNetwork = False) for inpt in x])
			#self.resetLayers(resetBiases = False) if resetNetwork == True else None
			return retVal

	def validate(self,normalizeInput = True,resetNetwork =	True,classNames=None):
		"""returns a confusionMatrix obj by predicting on test examples

		when the NueralNet obj is trained using 'train' method
		of the NueralNetwork obj the passed training set is split into 
		2 parts one part for training and one part for testing, this 
		methods predictas and creates a confusionMatrix obj of the 
		predictions and returns it 

		Parameters
		----------
		normalizeInput : bool, optional
			if True, the input from test examples are
			normalized, by default True
		resetNetwork : bool, optional
			if True normalizes input of the
			NueralNetwork obj, by default True
		classNames : [list,numpy.ndarray], optional
			names of classification classes, by 
			default None
		**if parameter 'classNames' is passed None
		class in confusionMatrix obj is set as 
		given in 'Attributes' section of 
		help(FrontalLobe.confusionMatrix)
		
		Returns
		-------
		confusionMatrix
			confusionMatrix obj base of prediction of
			test examples and test sets

		Raises
		------
		TypeError
			if either type of parameter 'normalizeInput'
			or parameter 'resetMetwork' is not bool
		AttributeError
			if training data attribute 'X' and labels 
			attribute 'Y' is not set
		AttributeError
			if attribute 'weights' is not initlalized
		TypeError
			it type of parameter 'classNmaes' is not in 
			[list,numpy.ndarray]
		TypeError
			if parameter 'classNames' is multi-dimensional 
			numpy.ndarray
		ValueError
			if no. of member elements is not same as 
			no. of classes for which the NueralNetworrk 
			obj is trained for

		"""
		if type(normalizeInput)!= bool or type(resetNetwork) != bool :
			errorArg = 'normalIzeInput' if type(normalizeInput) != bool else 'resetNetwork'
			raise TypeError(f" type of argument '{errorArg}' must be bool")
		if self.X is None or self.testIndices is None or self.Y is None:
			raise AttributeError("input examples 'X' or oyutput labels 'Y' is not defined yet")
		if self.weights.isEmpty() == True:
			raise AttributeError("attribute 'weights' is empty, can not validate on basis of emty 'weights' attribute")
		if classNames is not None :	
			if type(classNames) not in [list,np.array]:
				raise TypeError(f"argument 'classNames' can only be list and nump.ndarray")
			if type(classNames) == np.ndarray :
				if classNames.ndim != 1:
					raise TypeError("argument 'classNames' can only be one dimensional numpy.ndarray")
				classNames = list(classNames)
			if len(classNames) != len(self.classNames):
				raise ValueError("argument 'classNames' must contain the same number of classes as the classes in labels for which NueralNetwork object is trained ")
		retMat = confusionMatrix(len(self.classes),self.classes if classNames is not None else classNames)
		for i in self.testIndices:
			self.__SUFP(self.X[i],normalizeInput)
			retMat.update(predictedVal = self.labelExtractor(),expectedVal=self.Y[i])
		return retMat

	def __transformY(self,Y):
		classes,retVal = list(np.unique(Y)),[]
		classes.sort()
		NClasses = len(classes)
		if NClasses == 2 :
			CMap = lambda x: np.array([[classes.index(x)]])
		else:
			def CMap(x):
				retVal = np.zeros((NClasses,1))
				retVal[classes.index(x),0] = 1
				return retVal
		for val in Y:
			if type(val) == np.ndarray:
				label = CMap(val[0])
			else:
				label = CMap(val)
			retVal.append(label)
		retVal = np.array(retVal)			
		return classes,retVal
			
	def __transformX(self,X):
		return np.array([x.reshape(x.size,1) for x in X])

	def __randomSplit(self,trainSize):
		self.trainingSize = int(self.X.shape[0]*trainSize)
		self.testSize = self.X.shape[0] - self.trainingSize
		X1,Y1 = [],[]
		if trainSize >= 0.5:
			self.testIndices = list(range(self.X.shape[0]))
			self.trainIndices = random.sample(self.testIndices,self.trainingSize)
			for i in self.trainIndices:
				self.testIndices.remove(i)
		else:
			self.trainIndices = list(range(self.X.shape[0]))
			self.testIndices = random.sample(self.trainIndices,self.testSize)
			for i in self.testIndices:
				self.trainIndices.remove(i)

	def __continuousSplit(self,trainSize):
		self.trainingSize = int(X.shape[0]*trainSize)
		self.testSize = self.X.shape[0] - self.trainingSize
		self.trainIndices = list(range(self.X.shape[0]))[:self.trainingSize]
		self.testIndices = list(range(self.X.shape[0]))[self.trainingSize:]
		
	def labelExtractor(self):
		"""extracts labels from output layerr of the NueralNetwork obj

		Returns
		-------
		int
			class Label of output of Nueral Network

		"""
		if self.architecture[-1].width == 1:
			return 0 if self.architecture[-1][0] <= 0.5 else 1
		else:
			maxProb = np.amax(self.architecture[-1].layer)
			#return np.array([[0] if i != maxProb else [1] for i in self.architecture[-1]])
			for i,j in enumerate(self.architecture[-1]) :
				if j == maxProb :
					return i

	def __ZERemover(self,h):
		replacer = 10**(-(self.precision+2))
		return np.vectorize(lambda x: replacer if x <= 0 else x )(h)

	def __cost(self,h,y,m):
		h2 = 1-h
		h1,h2 = self.__ZERemover(h),self.__ZERemover(h2)
		#temp = np.multiply(-np.log(h),y) + np.multiply(-np.log(1-h),1-y)
		temp = np.multiply(-np.log(h1),y) + np.multiply(-np.log(h2),1-y)
		temp = (1/m)*np.sum(temp)
		return temp

	def __SUFP(self,x,normalizeInput=True):
		self.architecture[0]["all"] = x
		if normalizeInput == True:
			self.layerNormalization(1)
		else:
			round(self.layer(1),self.precision)
		self.depth = len(self.architecture) - 1
		for i in range(1,self.depth + 1):
			z = np.matmul(self.weights[i],self.layer(i).layer)
			a = self.OLAF(z) if i+1 == self.depth +1  else self.HLAF(z)
			self.architecture[i]["all"] = a.copy()
		return a

	def __SUBP(self,ind,m,normalizeInput = True):
		#print(self.X[ind])
		#print(self.Y[ind])
		h = self.__SUFP(self.X[ind],normalizeInput)
		l =(h - self.Y[ind])
		self.cost += self.__cost(h,self.Y[ind],m)
		if self.OLAF.__name__ != "sigmoid" :
			if self.OLAF.__name__ == "tanH" :
				l = l/(self.derivativeSigmoid(h))
				l = np.multiply(l,self.derivativetanH(h))
		self.depth = len(self.architecture) - 1
		delta = rolledVector()
		for i in range(self.depth,0,-1):
			a = self.layer(i).layer.copy()
			temp = (1/m)*np.matmul(l,a.T)
			delta.addMatrix(temp)
			#print(f"layerNum: {i} temp: {temp}")
			if i > 1 :
				#prime = np.multiply(a,(1-a))
				#prime = self.derivativeSigmoid(a)
				#prime = self.derivativetanH(a)
				if self.HLAF.__name__ == "sigmoid" :
					prime = self.derivativeSigmoid(a)
				if self.HLAF.__name__ == "tanH" :
					prime = self.derivativetanH(a)
				if self.HLAF.__name__ == "ReLU" :
					prime = self.derivativeReLU(a)
				if self.HLAF.__name__ == "softmax" :
					prime = self.derivativesoftmax(a)
				if self.HLAF.__name__ == "linear":
					prime = self.derivativelinear(a)
				l = np.matmul(self.weights[i].T,l)
				l = np.multiply(l,prime)
				l = l[1:,:] if self.layer(i).includeBias == True else l
		delta.reverse()
		return delta

	def train(self,X,Y,alpha,biasVal = 1,trainSize = 0.8,RWILimit = 10,weightType =float,
		replaceIL = True,changeILBias = False,replaceOL = True,normalizeInput = True,splitData = 'random',iterationSize = 100):
		"""trains NueralNetwork obj

		Parameters
		----------
		X : [numpy.ndarray,pandas.core.frame.DataFrame,pandas.core.series.Series]
			training examples
		Y : [numpy.ndarray,pandas.core.frame.DataFrame,pandas.core.series.Series]
			labels 
		alpha : int,float
			learning rate
		biasVal : int, optional
			sets bias values of all layer with biases
			, by default 1 see 
			help(FrontalLobe.NueralNetwork.setBiases)
			for more info
		trainSize : float, optional
			ratio of size of examples and labels used for 
			training to overal size of training set, by 
			default 0.8
		RWILimit : int,float, optional
			limit for random weight initialization, by 
			default 10 see
			help(FrontalLobe.NueralNetwork.RWInitialization)
		weightType : int,float, optional
			specifies the type of memebers of attribute
			'weights', by default float see
			help(FrontalLobe.NueralNetwork.RWInitialization)
		replaceIL : bool, optional
			specifies to change the width of input layer  or 
			not, if width of input layer is not compatible 
			with shape of memebers of training set 'X' 
			if width of input layer is not compatible, by 
			default True
		changeILBias : bool, optional
			if True and if width of input layer is not 
			compatible with shape of memebers of example set
			'X', then the bias nueron will be removed if it was
			there previously otherwise will be added, by default 
			False
		replaceOL : bool, optional
			species to change the width of ouput layer or 
			not, if width of output layer is not compatible
			with no. of class in label set 'Y', by default 
			True
		normalizeInput : bool, optional
			if True normalizes input of the
			NueralNetwork obj, by default True
		splitData : [str,int], optional
			specifies to split training set randomly 
			or continously, supported valueas are:
			'random' or 1, 'continuous' or 2,by 
			default 'random'
		iterationSize : int, optional
			specifies no.of epochs, by default 100

		Raises
		------
		AttributeError
			if architecture of nueralNetwork obj set before 
			training contains only 1 layer (a minimum of
			2 layer is required )
		TypeError
			if either type of parameter 'X' or 'y'
			is not in [numpyp.ndarray,
			pandas.core.frame.DataFrame,pandas.core.series.Series]
		TypeError
			if type of parameter 'X' is numpy.ndarray
			and it's ndim is not 2
		TypeError
			if type of parameter 'X' is numpy.ndarray
			and it's ndim is not in [1,2]
		ValueError
			if parameter 'X' and 'Y' do not match in 
			shape (no of rows in 'X' != no of memebers/
			rows in 'Y')
		ValueError
			if parameter 'trainSize' is not in [0,1]
		TypeError
			if type of atleast one of the parameters
			in ['replaceIL','changeILBias','replaceOL',
			'normalizeInput'] is not bool
		ValueError
			if passed value for parameter 'spliData' is 
			not in ['random', 1, 'continuous', 2]
		ValueError
			if the NueralNetwork obj is already trained 
			atleast once and passed parameter 'X' is not 
			compatible with input layer of NueralNetwork 
			obj
		ValueError
			if parmeter 'replaceIL' is False and NueralNetwork
			is not trained before, but passed parameter 'X' is 
			not compatible with input layer of NueralNetwork obj
		ValueError
			if the NueralNetwork obj is already trained atleast 
			once and class labels inpassed parameter 'Y' is not 
			compatible with output layer of NueralNetwork obj
		ValueError
			if the NueralNetwork obj is already trained 
			atleast once and new training set parameters 
			'X' and 'Y' is passed,but new parameters 'X'
			and 'Y' is  not compatible with the Nueralnett 
			obj
		TypeError
			if type of parameter 'iterationSize' is not int
		ValueError
			if parameter 'iterationSize' is int < 0

		**this method uses setBiases, RWInitialization
		  of class NueralNetwork, these methods also raise
		  errors. check 
		  help(FrontalLobe.NueralNetwork.setBiases) ,
		  help(FrontalLobe.ueralNetwork.RWInitialization)
		  for these errors.

		  The docstring examples assume that `FrontalLobe` has been imported as `FrontalLobe'
		
		"""
		if len(self.architecture) < 2:
			raise AttributeError("NueralNetwork must contain atleast 2 layer")
		allowedDTypes = [np.ndarray,pd.core.frame.DataFrame,pd.core.series.Series]
		if type(X) not in allowedDTypes or type(Y) not in allowedDTypes :
			errorArg,errorType = ("X",type(X)) if X not in allowedDTypes  else ("Y",type(Y)) 
			raise TypeError(f" {errorType} is not valid type for {errorArg}, supported types are {allowedDTypes.__str__()[1:-1]}")
		if type(X) == np.ndarray:
			if X.dim != 2 :
				raise TypeError(f"training examples (X) can only be 2 dimensional numpy.ndarray")
		if type(Y) == np.ndarray:
			if Y.dim not in [1,2] :
				raise TypeError(f"training labels (X) can only be 2 dimensional numpy.ndarray")
		X = np.array(X) if type(X) != np.ndarray else X
		Y = np.array(Y) if type(Y) != np.ndarray else Y
		if X.shape[0] != Y.shape[0]:
			raise ValueError("argument X and Y do not amtch in shape (no of rows in X != no of rows in Y")
		if trainSize <= 0 or trainSize > 1:
			raise ValueError("argument 'trainSize' must be in range (0,1] ")
		for args in [replaceIL,changeILBias,replaceOL,normalizeInput]:
			if args not in [True,False]:
				raise TypeError(f"{args} can only be bool value")
		if splitData not in ["random","continuous",1 ,2]:
			raise ValueError(f"{splitData} is not a valid value for splitData, supported values are 'random' or 1, 'continuous' or 2")
		reqShape = self.layer(1).width -1 if self.layer(1).includeBias == True else self.layer(1).width
		if X[0].size != reqShape:
			if self.__preTrained == True :
				raise ValueError("examples in argument 'X' is not compatible with the input layer of the pre-trained NueralNetwork obj")
			else:
				if replaceIL == True :
					prevWidth,prevBias  = self.layer(1).width,self.layer(1).includeBias
					bias = (not prevBias) if changeILBias == True else prevBias
					width = X[0].size if bias == False else X[0].size + 1
					self.architecture[0] =NueronLayer(width,bias)
					if showWarnings == True:
						warnings.warn(f"input layer changed from width {prevWidth} to {width}" + f" and 'includeBias' attribute  from {prevBias} to {bias}" if changeILBias == True else None)
				if replaceIL == False:
					raise ValueError("examples in argument 'X' is not compatible with the width of input layer of NueralNetwork obj")
		classes,tempY = self.__transformY(Y)
		reqOutWidth = len(classes) if len(classes) > 2 else 1
		if reqOutWidth != self.layer(self.depth +1).width :
			if self.__preTrained == True:
				raise ValueError("labels in expected output 'Y' is not compatible with the width of output layer of pre-trained NueralNetwork")
			else:
				if replaceOL == True:
					prevWidth,prevBias = self.architecture[-1].width,self.architecture[-1].includeBias
					self.architecture[-1] = NueronLayer(reqOutWidth,False)
					if showWarnings == True:
						warnings.warn(f"output layer changer from width {prevWidth} to {self.architecture[-1].width}" + f" and 'includeBias' attribute  from {prevBias} to False" if prevBias == True else None)
				else:
					self.architecture.append(NueronLayer(reqOutWidth,False))
					if showWarnings == True:
						warnings.warn(f"a new layer (output layer) is added to the NueralNetwork (numbers of layers in the Networks is increased from {len(self.architecture) - 1} tp {len(self.architecture)}")
		self.setBiases(biasVal)
		self.X,self.Y = self.__transformX(X),tempY
		if self.__preTrained == True:
			if self.isEmpty == True:
				self.RWInitialization(RWILimit,weightType)
			if self.classes != classes:
				self.X,self.Y = None,None
				raise ValueError(f"new training set (X,Y) do not contain same classes as the pre-trained NueralNetwork was previously trained on ")
		self.classes = classes
		self.width,self.depth, = self.getDim()
		self.__randomSplit(trainSize) if splitData in ["random",1] else self.__continuousSplit(trainSize)
		if self.__preTrained == False:
			self.RWInitialization(RWILimit,weightType)
			self.confusionMat = confusionMatrix(len(self.classes),self.classes)
		self.costs = []
		self.alpha = alpha
		self.cost = 0
		if type(iterationSize) != int:
			raise TypeError(f"type of argument 'iterationSize' must be int")
		if iterationSize < 0 :
			raise ValueError("argument 'iterationSize' must >= 0")
		with reprint.output(output_type="list",initial_len = 2*self.width + 3) as output:
			for i in range(iterationSize):
				delta = None
				self.cost  = 0
				for idx in self.trainIndices:
					if delta is None :
						delta = self.__SUBP(idx,self.trainingSize,normalizeInput)
					else:
						delta.addUpdate(self.__SUBP(idx,self.trainingSize,normalizeInput))
					strg = self.toString().split("\n")
					output[0] =f"epoch: {i}"
					for j,line in enumerate(strg):
						output[j+1] = line
					output[j+2] = f"cost: {self.cost}"
				self.costs.append(self.cost)
				delta.scalarMultiply((-1*self.alpha))
				self.weights.addUpdate(delta)
			output[0] = f"epoch: {iterationSize}"
		if iterationSize > 0: 
			plt.plot(list(range(len(self.costs))),self.costs)
			plt.show()
		self.__preTrained = True
	
	



	
