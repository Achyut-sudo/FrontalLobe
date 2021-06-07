import os
import pandas as pd
from FrontalLobe import NueralNetwork as nn

data = pd.read_csv("iris2.csv")
X = data[["d1","d2","d3","d4"]]
Y = data["mapped"]

model = nn([5,3,(3,False)],precision = 7)
model.train(X,Y,5,replaceOL = True,RWILimit = 10,normalizeInput=True,iterationSize = 1000)

