import numpy as np
import pylab as pl
import glob
import json

Dir_list = ["results/2_MCMC_Runs/","results/3_MCMC_Runs/","results/4_MCMC_Runs/",
            "results/5_MCMC_Runs/","results/6_MCMC_Runs/","results/7_MCMC_Runs/",
            "results/8_MCMC_Runs/","results/9_MCMC_Runs/"]

eos_list = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","HQC18","SLY2",
            "SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP",
            "SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1",
            "MS1B_PP","MS1_PP","BBB2","AP4","MPA1","MS1B","MS1",
            "SLY"]

eos_param_distro = {}
for eos in eos_list:
    
    eos_param_distro.update({eos: {"p1" : [], "g1" : [], "g2" : [], "g3" : [], "r2" : []}})

for Dir in Dir_list:
    
    for eos in eos_list:

        for File in glob.glob("{}{}_*".format(Dir, eos)): # for every file in this directory of this eos

            with open(File,"r") as f:
                data = json.load(f)

            p1_dist = eos_param_distro[eos]["p1"] + data["p1"]
            g1_dist = eos_param_distro[eos]["g1"] + data["g1"]
            g2_dist = eos_param_distro[eos]["g2"] + data["g2"]
            g3_dist = eos_param_distro[eos]["g3"] + data["g3"]
            r2_dist = eos_param_distro[eos]["r2"] + data["r2"]

            eos_param_distro.update({eos: {"p1" : p1_dist, "g1" : g1_dist, "g2" : g2_dist, "g3" : g3_dist, "r2" : r2_dist}})

with open("results/Processed_MCMC_Runs/Refined_9_MCMC_Runs.json","w") as f:
    json.dump(eos_param_distro, f, indent=2, sort_keys=True)
