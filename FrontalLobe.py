import numpy as np
import tabulate

class rolledVector:

	def __init__(self,zeroIndexed = False):
		self.vector = []
		self.shapeData = []
		self.size = 0
		self.indexType = 0 if zeroIndexed == True else 1


	def addMatrix(self,matrix):
		self.shapeData.append(matrix.shape)
		self.vector.append(matrix.reshape(1,matrix.size)[0])
		self.size += matrix.size

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

	def update(self,updateValue):
		if type(updateValue) in [list,np.ndarray] :
			if type(updateValue) == list:
				updateValue = np.array(updateValue)
				if updateValue.ndim == 1:
					if updateValue.size == self.size :
						temp,i = [],0
						for j,k in self.shapeData:
							temp.append(updateValue[i:i+(j*k)])
							i += j*k
						self.vector = temp
					else:
						raise ValueError(f"could not broadcast input rolledVector.vector from size {self.size} into len {updateValue.size}") from None
				else:
					raise TypeError("update value must be a single dimensional list or (2 or 1 dimensional) nump.ndarray ") from None
			else:
				if updateValue.ndim in [2,1]:	
					if updateValue.size == self.size :
						if updateValue.ndim == 2:
							self.layer =  list(updateValue)
						else:
							temp,i = [],0
							for j,k in self.shapeData:
								temp.append(updateValue[i:i+(j*k)])
								i += j*k
							self.layer = temp
					else:
						raise ValueError(f"could not broadcast input rolledVector.vector from size {self.size} into len {updateValue.size}") from None
				else:
					raise TypeError("update value must be a single dimensional list or (2 or 1 dimensional) nump.ndarray ") from None

		else:
			raise TypeError("update value must be a single dimensional list or (2 or 1 dimensional) nump.ndarray ") from None




class NueronLayer:

	def __init__(self,width,includeBias=True):
		self.layer = np.zeros((width,1))
		self.bias = self.layer[0][0] if includeBias == True else None
		self.includeBias = includeBias
		self.width = len(self.layer)

	def __repr__(self):
		return f"FrontalLobe.NueronLayer(width={len(self.layer)},includeBias={self.includeBias})"

	def __str__(self):
		return f"<class NueronLayer> includeBias = {self.includeBias} {self.layer.__str__()}"

	def setBias(self,value):
		if self.includeBias == False:
			raise ValueError(f"layer does not contain bias")
		else:
			self.layer[0][0] = value

	def __getitem__(self,index):
		try:
			return self.layer[index][0]
		except IndexError :
			raise IndexError("unit index out of range") from None

	def __setitem__(self,index,value):
		try:
			self.layer[index][0] = value
		except IndexError:
			raise IndexError("unit index out of range") from None


class NueralNetwork:

	def __init__(self,architecture):
		self.architecture = []
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
				raise TypeError("argument \'architecture\' must ony contain data of type <class int> , <class tuple> or <class NueronLayer>")
		self.thetaVec = rolledVector()
		self.Dvec = rolledVector()

	def __repr__(self):
		return f"FrontalLobe.NueralNetwork(architecture={self.architecture.__repr__()})"

	def toString(self,printUnderline = True):
		maxRows = max(self.architecture,key = lambda layer: layer.width).width
		temp = [[" " for i in range(len(self.architecture))] for j in range((2*maxRows)-1)]
		for j in range(len(temp[0])):
			ind = maxRows - self.architecture[j].width
			for i in self.__getitem__(j)[:,0]:
				(temp[ind])[j] = i 
				ind +=  2
		table =  tabulate.tabulate(temp,headers=["input Layer"] + [f"hiddenLayer {i+1}" for i in range(len(temp[0]) - 2)] +["output Layer"])
		if printUnderline == False :
			temp2 = table.split("\n")
			temp2.pop(1)
			return "\n".join(temp2)
		return table

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
				return self.architecture[layerNumber].layer
			except IndexError:
				raise IndexError("layer index out of range") from None
		try:
			return self.architecture[layerNumber][unitNumber]
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
					raise ValueError("layer can be of type \'numpy.ndarray\' or \'list\' only!") from None
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
							raise ValueError(f"could not broadcast input array from shape {self.__getitem__(layerNumber).shape} into shape {self.__getitem__(layerNumber).shape}") from None
						else:
							self.architecture[layerNumber].layer = value
			except IndexError:
				raise IndexError("layer index out of range") from None
		else:
			layer = self.__getitem__(layerNumber)
			try:
				layer[unitNumber] = value
			except IndexError:
				raise IndexError("unit index out of range") from None





