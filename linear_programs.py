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
    
    # Constraints
    for i in range(n):
        m.addConstr(y[i]*(X.iloc[i].dot(v))>=0)
    m.update()
    
    
    m.Params.OutputFlag = 0 # To avoid verbose output of m.optimize()
    m.optimize() # Run the model
    
    
    return GRB.OPTIMAL # Gurobi status: 3=LP is infeasible, 2 = Optimul solution reached, and many others

  