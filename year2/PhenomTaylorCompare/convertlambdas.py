import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

class LambdasInversion:

    def __init__(self, mr, LambdaT, DLambdaT):

        self.mr = mr
        self.LambdaTs = LambdaT
        self.DLambdaTs = DLambdaT

    def LambdaSolving(self, Ls):
        # Expression providing function for fsolve call

        Lambda1 = Ls[0]
        Lambda2 = Ls[1]
        expressions = np.zeros(2)
        expressions[0] = (8/13)*((1+7*self.mr-31*self.mr**2)*(Lambda1+Lambda2)+((1-4*self.mr)**.5)*(1+9*self.mr-11*self.mr**2)*(Lambda1-Lambda2))-self.LambdaT
        expressions[1] = .5*(((1-4*self.mr)**.5)*(1-(13272/1319)*self.mr+(8944/1319)*self.mr**2)*(Lambda1+Lambda2)+(1-(15910/1319)*self.mr+(32850/1319)*self.mr**2+(3380/1319)*self.mr**3)*(Lambda1-Lambda2))-self.DLambdaT
        return expressions

    def solve_system(self):

        Lambdas1 = []
        Lambdas2 = []
        for self.mr, self.LambdaT, self.DLambdaT in zip(self.mr,self.LambdaTs,self.DLambdaTs):

            Lambda1, Lambda2 = fsolve(self.LambdaSolving,[1.0,1.0])
            Lambdas1.append(Lambda1)
            Lambdas2.append(Lambda2)

        return(Lambdas1,Lambdas2)
