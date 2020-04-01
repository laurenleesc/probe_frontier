import numpy as np
import pandas as pd
from gurobipy import *
#import gurobipy as GRB

def checkMleExistence(data):
    X = data.iloc[:,:-1] 
    y= data.iloc[:,-1]
    n,p = X.shape

    
    m = Model()
    v = []
    # Define variables and add to objective function
    for i in range(p):
        v+= [m.addVar(-GRB.INFINITY,GRB.INFINITY,0,GRB.CONTINUOUS,"v"+str(i))]
    m.update()
    
    
    
    float32_epsilon = (np.finfo(np.float32).eps)*10 
            ### We wanted strict inequality, but Gurobi can not handle strict inequality
            # So, Had to use machine epsilon as reference. However, machine epsilon turned out to be too small
            # So, we used ten times machine epsilon
            # Machine epsilon can be considered as the smallest number greater than zero.
            # https://kite.com/python/answers/how-to-find-machine-epsilon-using-numpy-in-python
            
            
            
    # Constraints
    for i in range(n):
        m.addConstr(y[i]*(X.iloc[i].dot(v))>=float32_epsilon)
    m.update()
    
    
    m.Params.OutputFlag = 0 # To avoid verbose output of m.optimize()
    m.optimize() # Run the model
    
    
    return GRB.OPTIMAL # Gurobi status: 3=LP is infeasible, 2 = Optimul solution reached, and many others

  