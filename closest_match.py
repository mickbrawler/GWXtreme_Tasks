from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import lal
import numpy as np
import pylab as pl
import os
import json

eos_list = ["PAL6","SLy","AP1","AP2","AP3","AP4","FPS","WFF1","WFF2","WFF3"
            ,"BBB2","BPAL12","ENG","MPA1","MS1","MS2","MS1b","PS","GS1a"
            ,"GS2a","BGN1H1","GNH3","H1","H2","H3","H4","H5","H6a","H7"
            ,"PCL2","ALF1","ALF2","ALF3","ALF4"]

# each eos' pressure and gammas
eos_values = {"PAL6":[34.380,2.227,2.189,2.159]
             ,"SLy":[34.384,3.005,2.988,2.851]
             ,"AP1":[33.943,2.442,3.256,2.908]
             ,"AP2":[34.126,2.643,3.014,2.945]
             ,"AP3":[34.392,3.166,3.573,3.281]
             ,"AP4":[34.269,2.830,3.445,3.348]
             ,"FPS":[34.283,2.985,2.863,2.600]
             ,"WFF1":[34.031,2.519,3.791,3.660]
             ,"WFF2":[34.233,2.888,3.475,3.517]
             ,"WFF3":[34.283,3.329,2.952,2.589]
             ,"BBB2":[34.331,3.418,2.835,2.832]             
             ,"BPAL12":[34.358,2.209,2.201,2.176]
             ,"ENG":[34.437,3.514,3.130,3.168]
             ,"MPA1":[34.495,3.446,3.572,2.887]
             ,"MS1":[34.858,3.224,3.033,1.325]
             ,"MS2":[34.605,2.447,2.184,1.855]
             ,"MS1b":[34.855,3.456,3.011,1.425]
             ,"PS":[34.671,2.216,1.640,2.365]
             ,"GS1a":[34.504,2.350,1.267,2.421]
             ,"GS2a":[34.642,2.519,1.571,2.314]
             ,"BGN1H1":[34.623,3.258,1.472,2.464]
             ,"GNH3":[34.648,2.664,2.194,2.304]
             ,"H1":[34.564,2.595,1.845,1.897]
             ,"H2":[34.617,2.775,1.855,1.858]
             ,"H3":[34.646,2.787,1.951,1.901]
             ,"H4":[34.669,2.909,2.246,2.144 ]
             ,"H5":[34.609,2.793,1.974,1.915]
             ,"H6a":[34.593,2.637,2.121,2.064]
             ,"H7":[34.559,2.621,2.048,2.006 ]
             ,"PCL2":[34.507,2.554,1.880,1.977]
             ,"ALF1":[34.055,2.013,3.389,2.033]
             ,"ALF2":[34.616,4.070,2.411,1.890]
             ,"ALF3":[34.283,2.883,2.653,1.952]}

def closest_match_ML(eos, N):
    """
    METHOD
    ======
    This function should take as input the name of a equation of state and spit 
    out the p,g1,g2,g3 parameters that matches the best for a given tolerance

    PARAMETERS
    ==========
    eos : (String) Name of NS equation of state

    OUTPUT
    ======
    The p,g1,g2,g3 that produce masses/lambdas closest to that produce from the
    target eos by normal means.
    """

    p0,g1,g2,g3 = eos_values[eos] # Get table values for specified eos
    target = [p0,g1,g2,g3]
    
    # Get ranges for each parameter
    p0_0 = p0 - 1
    p0_f = p0 + 1
    p0_space = np.arange(p0_0,p0_f+.001,.001)

    g1_0 = g1 - 1
    g1_f = g1 + 1
    g1_space = np.arange(g1_0,g1_f+.001,.001)

    g2_0 = g2 - 1
    g2_f = g2 + 1
    g2_space = np.arange(g2_0,g2_f+.001,.001)

    g3_0 = g3 - 1
    g3_f = g3 + 1
    g3_space = np.arange(g3_0,g3_f+.001,.001)

    modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")

    # Get target eos lambdas
    min_mass = modsel.getMinMass(eos)
    s, _, _, max_mass = modsel.getEoSInterp(eos, m_min=min_mass, N=1000)
    target_masses = np.linspace(min_mass,max_mass,N)
    Lambdas = s(target_masses)
    target_lambdas = (Lambdas / lal.G_SI) * ((target_masses * lal.MRSUN_SI) ** 5)

    # Delete this when all parameters are unlocked
    g1_space = np.array([g1])
    g2_space = np.array([g2])
    g3_space = np.array([g3])

    # Get trial eos lambdas
    parameter_combos = []
    CSs = np.array([])
    test_counter = 1
    trial_m_list = np.array([])
    trial_l_list = np.array([])
    for p in p0_space:
        for g1 in g1_space:
            for g2 in g2_space:
                for g3 in g3_space:

                    parameter_combos.append([p,g1,g2,g3]) # record of all combinations used

                    try:
                        s, min_mass, max_mass = modsel.getEoSInterpFrom_p_gs(p,g1,g2,g3,N=100)
                    except ValueError:
                        continue
                    except RuntimeError:
                        continue
                    masses = np.linspace(min_mass,max_mass,N)
                    trial_m_list = np.append(trial_m_list, masses)
                    Lambdas = s(masses)
                    trial_lambdas = (Lambdas / lal.G_SI) * ((masses * lal.MRSUN_SI) ** 5)
                    trial_l_list = np.append(trial_l_list,trial_lambdas)

                    print(test_counter)
                    test_counter += 1
                    
                    # chi-square operation
                    CS = np.sum((target_lambdas - trial_lambdas) ** 2)
                    CSs = np.append(CSs,CS)
           
    match = parameter_combos[np.argmin(CSs)] # parameter combo associated with minimum chi-square result

    pl.plot(target_masses, target_lambdas, label="target:"+str(target)) 
    pl.plot(trial_m_list[np.argmin(CSs)], trial_l_list[np.argmin(CSs)], label="match:"+str(match))

    pl.legend()
    pl.xlabel("Masses")
    pl.ylabel("Lambdas")
    pl.title("Target/Match's Masses/Lambdas")
    pl.savefig("Masses_Lambdas_plot.png")
