from GWXtreme import eos_model_selection as ems
from matplotlib import pyplot as plt
import numpy as np
import json

def eos_bayes_factor():
    
    # Finds bayes factor of each eos in list and puts them in a json file

    choose_eos = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","HQC18","SLY2","SLY230A",
                  "SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6",
                  "SK272","SKI3","SKI5","MPA1","MS1B_PP","MS1_PP"]

    modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")
    
    bayes_factors = []
    for eos in choose_eos:
        bayes_factor = modsel.computeEvidenceRatio(EoS1=eos,EoS2="SLY")
        bayes_factors.append(bayes_factor)

    dictionary = {"bayes factors": bayes_factors}

    with open("results/bayes_factors_SLY.json", "w") as f:
        json.dump(dictionary, f, indent=2, sort_keys=True)
