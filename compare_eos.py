from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import lal
import numpy as np
import pylab as pl
import os
import json

modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")

tot_list = ["PAL6","AP1","AP2","AP3","AP4","FPS","WFF1","WFF2","WFF3"
            ,"BBB2","BPAL12","ENG","MPA1","MS1","MS2","MS1b","PS","GS1a"
            ,"GS2a","BGN1H1","GNH3","H1","H2","H3","H4","H5","H6a","H7"
            ,"PCL2","ALF1","ALF2","ALF3","ALF4"]

GWX_list = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","HQC18","SLY2","SLY230A",
            "SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6",
            "SK272","SKI3","SKI5","MPA1","MS1B_PP","MS1_PP"]

tar_list = ["BBB2","AP4","MPA1","MS1b","MS1"]

ref_eos = "SLY"

# each eos' pressure and gammas
# https://arxiv.org/pdf/0812.2163.pdf
# "Constraints on a phenomenologically parameterized neutron-star equation of state"
t_eos_vals = {"PAL6":[33.380,2.227,2.189,2.159]
             ,"AP1":[32.943,2.442,3.256,2.908]
             ,"AP2":[33.126,2.643,3.014,2.945]
             ,"AP3":[33.392,3.166,3.573,3.281]
             ,"AP4":[33.269,2.830,3.445,3.348]
             ,"FPS":[33.283,2.985,2.863,2.600]
             ,"WFF1":[33.031,2.519,3.791,3.660]
             ,"WFF2":[33.233,2.888,3.475,3.517]
             ,"WFF3":[33.283,3.329,2.952,2.589]
             ,"BBB2":[33.331,3.418,2.835,2.832]             
             ,"BPAL12":[33.358,2.209,2.201,2.176]
             ,"ENG":[33.437,3.514,3.130,3.168]
             ,"MPA1":[33.495,3.446,3.572,2.887]
             ,"MS1":[33.858,3.224,3.033,1.325]
             ,"MS2":[33.605,2.447,2.184,1.855]
             ,"MS1b":[33.855,3.456,3.011,1.425]
             ,"PS":[33.671,2.216,1.640,2.365]
             ,"GS1a":[33.504,2.350,1.267,2.421]
             ,"GS2a":[33.642,2.519,1.571,2.314]
             ,"BGN1H1":[33.623,3.258,1.472,2.464]
             ,"GNH3":[33.648,2.664,2.194,2.304]
             ,"H1":[33.564,2.595,1.845,1.897]
             ,"H2":[33.617,2.775,1.855,1.858]
             ,"H3":[33.646,2.787,1.951,1.901]
             ,"H4":[33.669,2.909,2.246,2.144 ]
             ,"H5":[33.609,2.793,1.974,1.915]
             ,"H6a":[33.593,2.637,2.121,2.064]
             ,"H7":[33.559,2.621,2.048,2.006 ]
             ,"PCL2":[33.507,2.554,1.880,1.977]
             ,"ALF1":[33.055,2.013,3.389,2.033]
             ,"ALF2":[33.616,4.070,2.411,1.890]
             ,"ALF3":[33.283,2.883,2.653,1.952]
             ,"ALF4":[33.314,3.009,3.438,1.803]}

r_eos_vals = {"SLY":[33.384,3.005,2.988,2.851]}

N = 100

# Essentially meant to obtain better p,g1,g2,g3 values

for eos in tot_list:

    log_p0_SI, g1, g2, g3 = t_eos_vals[eos]

    p_range = np.arange(log_p0_SI-1,log_p0_SI+1,.001)
    g1_range = np.arange(g1-1,g1+1,.001)
    g2_range = np.arange(g2-1,g2+1,.001)
    g3_range = np.arange(g3-1,g3+1,.001)
        
    eos_pointer = lalsim.SimNeutronStarEOSByName(eos)
    fam_pointer = lalsim.CreateSimNeutronStarFamily(eos_pointer)
    min_mass = lalsim.SimNeutronStarFamMinimumMass(fam_pointer)/lal.MSUN_SI

    s, _, Lambdas, max_mass = getEoSInterp(eosname=eos, m_min=min_mass)
    target_masses = np.linspace(min_mass,max_mass,N)
    target_Lambdas = s(masses)
    target_lambdas = (Lambdas / lal.G_SI) * ((masses * lal.MRSUN_SI) ** 5)

    CSs = []
    p_combo = []
    matches = []
    for p in p_range:
        for g1 in g1_range:
            for g2 in g2_range:
                for g3 in g3_range:
                    s, min_mass, max_mass = getEoSInterpFrom_piecewise(p,g1,g2,g3)
                    trial_masses = np.linspace(min_mass,max_mass,N)
                    trial_Lambdas = s(masses)
                    trial_lambdas = (Lambdas / lal.G_SI) * ((masses * lal.MRSUN_SI) ** 5)
                    p_combo.append([p,g1,g2,g3])
                    CSs.append(np.sum(target_lambdas-trial_lambdas)**2)
    matches.append((eos,p_combo[np.argmin(CSs)]))

output = np.vstack(matches).T
outputfile = "results/matches.txt"
np.savetxt(outputfile, output, fmt="%f")
