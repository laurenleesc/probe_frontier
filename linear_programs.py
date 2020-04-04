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
            
            
    # Constraints
    # The right hand side of the constraints does not matter, because of the constant c on the LHS. They adjsust
    # However, these constraints find complete separability
    for i in range(n):
        if y[i]>0:
            m.addConstr(((X.iloc[i]).dot(v))>= 1)
        else:
            m.addConstr(((X.iloc[i]).dot(v))<= -1)
    

    m.update()
    
    
#     m.ModelSense = -1 # To Maximize instead of Minimize
    m.Params.OutputFlag = 0 # To avoid verbose output of m.optimize()
    m.optimize() # Run the model
    return m.Status # Gurobi status: 3=LP is infeasible, 2 = Optimul solution reached, and many others

  