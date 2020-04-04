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

    v+= [m.addVar(-GRB.INFINITY,GRB.INFINITY,0,GRB.CONTINUOUS,"c")]# This is the variable, that will hold the constant c
    m.update()
    
    
    X['ones'] = np.ones(n)
    
    
    float32_epsilon = (np.finfo(np.float32).eps)*10 
            ### We wanted strict inequality, but Gurobi can not handle strict inequality
            # So, Had to use machine epsilon as reference. However, machine epsilon turned out to be too small
            # So, we used ten times machine epsilon
            # Machine epsilon can be considered as the smallest number greater than zero.
            # https://kite.com/python/answers/how-to-find-machine-epsilon-using-numpy-in-python
            
            
            
    # Constraints
    # We want complete separation, this is the reason we are keeping a buffer zone of float epsilon (a very small number)
    # But the epsilon keeps two groups completely separated.
    for i in range(n):
        if y[i]>0:
            m.addConstr(((X.iloc[i]).dot(v))>= 0)
        else:
            m.addConstr(((X.iloc[i]).dot(v))<= -1*float32_epsilon)
    

    m.update()
    
    
    # m.ModelSense = -1 # To Maximize instead of Minimize
    m.Params.OutputFlag = 0 # To avoid verbose output of m.optimize()
    m.optimize() # Run the model
    return m.Status # Gurobi status: 3=LP is infeasible, 2 = Optimul solution reached, and many others

  