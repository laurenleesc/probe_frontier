import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import random


def generateNormalVariables(n, p):

    X = {} # initialize dictionary to hold values

    for j in range(p):
        #np.random.seed(seed = j)   #seed for distribution
        #random.seed(j)   #seed for random number generator
        X[j] = np.random.normal(0, 1/np.sqrt(n), n) # NOTE: ADDED 1/SQRT(N) SCALING 3/31
    return X

def generateX(n,p):
    
    # Initialize dataframe with an intercept column
    #intercept = np.repeat(1,n) # Create n 1's for intercept column
    #X = pd.DataFrame(intercept.reshape(n,1))
    # COMMENTED OUT INTERCEPT FOR TIME BEING. CAUSING ISSUES WITH SIGNAL STRENGTH SCALING
    X = pd.DataFrame()
    
    # Generate data
    if p != 0:
        Xnorm = generateNormalVariables(n, p)
        Xnorm = pd.DataFrame.from_dict(Xnorm)
        X = pd.concat([X, Xnorm],ignore_index=True,axis=1)
    
    return X

def generateRandomBeta(q, mu, stdev):
    beta = {}
    for j in range(q):
        #random.seed(j)#seed for random number generator
        beta[j] = np.random.normal(mu,stdev)
        #beta[j] = np.random.poisson(stdev)-3 # Checking to see what happens when betas come from noncentral poisson
        #beta[j] = np.random.uniform(mu,stdev)
    
    beta = pd.DataFrame(list(beta.items()))
    beta = beta.drop([0],axis=1)
    
    return beta

def sigmoid(z):
    return 1/(1+np.exp(-z))

def generateResponseVariable(X, beta, dist):
    
    beta = np.squeeze(beta)
    
    if dist == 'bernoulli':
        meanValues = sigmoid(X.dot(beta)) 
    elif dist == 'poisson':
        meanValues = np.exp(X.dot(beta))
    elif dist == 'exponential':
        meanValues = 1/(X.dot(beta))
    else:
        print('please spell check distribution name, all lowercase: bernoulli, poisson or exponential')
        
    y = []
    
    #np.random.seed(123)
    for eachMean in np.squeeze(meanValues.values):
        if dist == 'bernoulli':
            randomPrediction = np.random.binomial(1,eachMean)
        elif dist == 'poisson':
            randomPrediction = np.random.poisson(eachMean)
        elif dist == 'exponential':
            randomPrediction = np.random.exponential(eachMean)
        y.append(randomPrediction)
        #print(randomPrediction, eachMean)
    
    return meanValues, y

def generateData(dist, n, p, mu, stdev, signal_strength):
    
    # Generate Data
    X = generateX(n=n, p=p)

    # Generate Betas
    q = X.shape[1]
    beta = generateRandomBeta(q=q, mu=mu, stdev=stdev)
    beta_norm_squared = beta.T.dot(beta).values[0,0]
    beta_scaled = beta * np.sqrt((n * signal_strength) / beta_norm_squared)

    # Generate Response Variable (and associated means - what goes into the link fn)
    means, y = generateResponseVariable(X=X, beta=beta_scaled, dist=dist) ### dist means pass distribution name as string
    
    # Make sure we return numpy arrays (easier to work with later and all names are useless here anyway)
    beta_scaled = np.squeeze(np.array(beta_scaled))
    X = np.array(X)
    y = np.array(y)    
    
    return X, beta_scaled, y, means