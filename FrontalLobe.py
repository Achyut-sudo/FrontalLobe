import numpy as np
import tabulate

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
			if type(layerData) == int :
				includeBias = True
				self.architecture.append(NueronLayer(layerData,includeBias))
			if type(layerData) == tuple:
				includeBias = layerData[1]
				self.architecture.append(NueronLayer(layerData[0],includeBias))
			if type(layerData) == NueronLayer :
				self.architecture.append(layerData)

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
			(layerNumber,unitNumer) = pos
		except TypeError:
			layerNumber = pos
			unitNumer=None
		if unitNumer == None:
			try:
				return self.architecture[layerNumber].layer
			except IndexError:
				raise IndexError("layer index out of range") from None
		try:
			return self.architecture[layerNumber][unitNumer]
		except IndexError:
			raise IndexError("layer index out of range") from None

	def __setitem__(self,pos,value):
		try:
			layer = self.architecture[pos[0]]
		except IndexError:
			raise IndexError("layer index out of range") from None
		layer[pos[1]] = value



		# retStr = ""
		# for i in range(temp.shape[0]):
		# 	tempStr = ""
		# 	for j in range(temp.shape[1]):
		# 		tempStr += (" "*3 if temp[i][j] == None else f"{temp[i][j]:4.6}") + " "
		# 	tempStr +="\n"
		# 	retStr += tempStr
		#print(np.array_str(temp, precision=2))



#print ("\033[A                             \033[A",end="" )


