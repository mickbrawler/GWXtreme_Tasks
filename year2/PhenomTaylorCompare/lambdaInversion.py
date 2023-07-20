import numpy as np
import json
import seaborn as sns
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from GWXtreme import eos_model_selection as ems

class TildeInvertLambda:

    def __init__(self,q,DLT,LT):
        # Requires symmetric mass-ratio, delta-lambda-tilda and lambda-tilda to 
        # invert to lambda1 and lambda2.

        self.qs = q
        self.DLTs = DLT
        self.LTs = LT

    def LambdaSolving(self,Ls):
        # Expression providing function for fsolve call

        Lambda1 = Ls[0]
        Lambda2 = Ls[1]
        expressions = np.zeros(2)
        expressions[0] = (8/13)*((1+7*self.q-31*self.q**2)*(Lambda1+Lambda2)+((1-4*self.q)**.5)*(1+9*self.q-11*self.q**2)*(Lambda1-Lambda2))-self.LT
        expressions[1] = .5*(((1-4*self.q)**.5)*(1-(13272/1319)*self.q+(8944/1319)*self.q**2)*(Lambda1+Lambda2)+(1-(15910/1319)*self.q+(32850/1319)*self.q**2+(3380/1319)*self.q**3)*(Lambda1-Lambda2))-self.DLT
        return expressions

    def solve_system(self):

        Lambdas1 = []
        Lambdas2 = []
        for self.q, self.LT, self.DLT in zip(self.qs, self.LTs, self.DLTs):

            Lambda1, Lambda2 = fsolve(self.LambdaSolving,[1.0,1.0])
            Lambdas1.append(Lambda1)
            Lambdas2.append(Lambda2)

        return Lambdas1, Lambdas2

def LambdaInvertTilde(q,lambda1s,lambda2s):
        # Requires symmetric mass-ratio, lambda1, lambda2 to invert to 
        # delta-lambda-tilda and lambda-tilda.

        LambdaT = (8/13)*((1+7*q-31*q**2)*(lambda1s+lambda2s)+((1-4*q)**.5)*(1+9*q-11*q**2)*(lambda1s-lambda2s))
        DLambdaT = .5*(((1-4*q)**.5)*(1-(13272/1319)*q+(8944/1319)*q**2)*(lambda1s+lambda2s)+(1-(15910/1319)*q+(32850/1319)*q**2+(3380/1319)*q**3)*(lambda1s-lambda2s))
        return DLambdaT, LambdaT

def testInversion():

     myFile = "./Phenom_Taylor/mine/122_1.77_1.19/bns_example_result.json"
     with open(myFile,"r") as f:
         data = json.load(f)["posterior"]["content"]

     # UNIFORM (L1, L2)
     mc = data["chirp_mass"]
     q = data["mass_ratio"]
     calc_mass1, calc_mass2 = ems.getMasses(np.array(q),np.array(mc))

     file_mass1 = np.array(data["mass_1"])
     file_mass2 = np.array(data["mass_2"])

     color1, color2 = "green", "black"
     legend1, legend2 = "calculated", "file"
    
     title = "mass1"
     plt.clf()
     sns.kdeplot(data=calc_mass1,color=color1)
     sns.kdeplot(data=file_mass1,color=color2)
     plt.legend(labels=[legend1,legend2])
     plt.title(title)
     plt.savefig("plots/compare/{}.png".format(title))

     title = "mass2"
     plt.clf()
     sns.kdeplot(data=calc_mass2,color=color1)
     sns.kdeplot(data=file_mass2,color=color2)
     plt.legend(labels=[legend1,legend2])
     plt.title(title)
     plt.savefig("plots/compare/{}.png".format(title))

     sq = (file_mass1*file_mass2)/((file_mass1+file_mass2)**2)

     file_L1 = np.array(data["lambda_1"])
     file_L2 = np.array(data["lambda_2"])

     calc_dLT, calc_LT = LambdaInvertTilde(sq,file_L1,file_L2)
     file_dLT = data["delta_lambda_tilde"]
     file_LT = data["lambda_tilde"]

     invert = TildeInvertLambda(sq,file_dLT,file_LT)
     calc_L1, calc_L2 = invert.solve_system()

     title = "delta_lambda_tilde"
     plt.clf()
     sns.kdeplot(data=calc_dLT,color=color1)
     sns.kdeplot(data=file_dLT,color=color2)
     plt.legend(labels=[legend1,legend2])
     plt.title(title)
     plt.savefig("plots/compare/{}.png".format(title))

     title = "lambda_tilde"
     plt.clf()
     sns.kdeplot(data=calc_LT,color=color1)
     sns.kdeplot(data=file_LT,color=color2)
     plt.legend(labels=[legend1,legend2])
     plt.title(title)
     plt.savefig("plots/compare/{}.png".format(title))

