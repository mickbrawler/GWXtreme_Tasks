import numpy as np
import json

# Opens files that originate from a single file with GW170817's nested sampling
# evidences for each EoS. We compute the BFs w.r.t. SLY and try out multiple 
# variations on its "error": 1) quadrature sum, 2) "worst possible error",
# 3) fractional error.

with open("files/BNS/TaylorF2_eos_prior_narrow_evidences.json","r") as f:
    Taylor_nest_data = json.load(f)

with open("files/BNS/IMRphenom_eos_prior_narrow_evidences.json","r") as f:
    Phenom_nest_data = json.load(f)

with open("data/BNS/BFs/GW170817_2D_3D_BFs_10000samp.json","r") as f:
    data = json.load(f)

labels = ['TaylorF2 LALInference_Nest','IMRPhenom LALInference_Nest']
datasets = [Taylor_nest_data, Phenom_nest_data]
EoSs = ["SKOP","H4","HQC18","SLY2","SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1","MS1B_PP","MS1_PP"]
for nest_data, label in zip(datasets,labels):
    EoSs_BF_err = {}
    for EoS in EoSs:
        EoS1 = nest_data[EoS][0] # evidence of EoS1
        EoS2 = nest_data['SLY'][0] # evidence of EoS2
        BF = EoS1/EoS2

        EoS1err = nest_data[EoS][-1]
        EoS2err = nest_data['SLY'][-1]

        # 1) quadrature sum
        err1 = ((EoS1err**2)+(EoS2err**2))**0.5

        # 2) "worst possible error"
        EoS1min, EoS1max = EoS1-EoS1err, EoS1+EoS1err
        EoS2min, EoS2max = EoS2-EoS2err, EoS2+EoS2err

        ErrMin = EoS1max/EoS2min
        ErrMax = EoS1min/EoS2max
        err2 = ErrMax - ErrMin

        # 3) fractional error
        err3 = BF * (((EoS1err/EoS1)**2) + ((EoS2err/EoS2)**2)) ** 0.5

        EoSs_BF_err[EoS] = [BF,[err1,err2,err3]]

    data[label] = EoSs_BF_err

with open("data/BNS/BFs/GW170817_2D_3D_BFs_10000samp.json","w") as f:
    json.dump(data, f, indent=2, sort_keys=True)

