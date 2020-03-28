import pandas as pd
import numpy as np 
import time
from simulate import generateData
from linear_programs import checkMleExistence

n=40000
p=200
mu = 0
stdev = 1.0 #signal strength

X, b, y, means = generateData(dist = 'bernoulli', n = n, p = p, mu = mu, stdev = stdev) 

#df = pd.DataFrame({'X':X, 'y':y})
df = pd.DataFrame(X)
#df.column=('X')
df['y'] = pd.Series(y) 

kappaGrid = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
kappaProportion = []

d1 = time.time()
for kappa in kappaGrid:
    sampleSize = round(p/kappa)
        
    nMleExist = 0
    for i in range(5):
        subSample = df.sample(n=sampleSize)
        subSample =subSample.reset_index()
        mleStatus = checkMleExistence(subSample)
        if mleStatus == 3:
            nMleExist +=1
    propMleExist = nMleExist/len(range(5))
    kappaProportion.append(propMleExist)
    
    #uIdx = next(i for i,v in enumerate(kappaProportion) if v >= 0.5)
    #uKappa = kappaGrid[uIdx]
    #lKappa = kappaGrid[uIdx-1]

d2 = time.time()
delta = round(d2 - d1)
print(kappaProportion, delta)


