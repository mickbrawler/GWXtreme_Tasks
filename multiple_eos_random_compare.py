from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import os
import json

# list of eos GWXtreme can work with off the cuff
GWX_list = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","HQC18","SLY2","SLY230A",
            "SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6",
            "SK272","SKI3","SKI5","MPA1","MS1B_PP","MS1_PP","BBB2","AP4",
            "MPA1","MS1b","MS1","SLY"]

# list of eos paper found p0,g1,g2,g3 values for
pap_list = ["PAL6","AP1","AP2","AP3","AP4","FPS","WFF1","WFF2","WFF3"
            ,"BBB2","BPAL12","ENG","MPA1","MS1","MS2","MS1b","PS","GS1a"
            ,"GS2a","BGN1H1","GNH3","H1","H2","H3","H4","H5","H6a","H7"
            ,"PCL2","ALF1","ALF2","ALF3","ALF4","SLY"]

# https://arxiv.org/pdf/0812.2163.pdf
# "Constraints on a phenomenologically parameterized neutron-star equation of state"
p_eos_val = {"PAL6":[33.380,2.227,2.189,2.159]
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
             ,"SLY":[33.384,3.005,2.988,2.851]}

modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")

def likelihood(p0,g1,g2,g3,min_mass,max_mass,N,target_lambdas)
    """
    METHOD: Takes in the value of the random parameters p0,g1,g2,g3 and some 
    other necessities, uses them to produce the trial_lambdas, and compares
    them with the target_lambdas.
    
    PARAMETERS:
    ___________ 
    p0 : pressure.
    g1 : adiabatic energy density indice 1.
    g2 : adiabatic energy density indice 2.
    g3 : adiabatic energy density indice 3.
    min_mass : minimum mass allowed by eos.
    max_mass : maximum mass allowed by eos.
    target_lambdas : lambdas the eos produced via getEosInterp.
    
    OUTPUT: Returns the value of the r-squared.
    """

    s, Lambdas, min_mass, max_mass = modsel.getEoSInterpFrom_piecewise(p0,g1,g2,g3) # had to change "From_piecewise" to now return Lambdas like the normal "getEoSInterp"
    trial_masses = np.linspace(min_mass,max_mass,N)
    trial_Lambdas = s(trial_masses)
    trial_lambdas = (Lambdas / lal.G_SI) * ((trial_masses * lal.MRSUN_SI) ** 5) # lambdas the eos produced via getEosInterpFrom_piecewise.
    r_val = 1 / np.log(np.sum((target_lambdas - trial_lambdas) ** 2)) # r^2 value
           
    return(r_val)

def get_eos_parameter_dist(transitions,N):
    """
    METHOD: Compares the trial_lambdas resulting from different combinations of
    parameters with the target_lambdas for a certain eos. This is done for
    eos.

    PARAMETERS:
    ___________
    transitions : The number of "states" changed. Each state is a parameter combination.
    N : The array size of the mass arrays.

    OUTPUT: Creates a json holding the the parameter distributions for each eos.
    """

    eos_post_p0 = []
    eos_post_g1 = []
    eos_post_g2 = []
    eos_post_g3 = []

    for eos in GWX_list: # we want to find the p0,g1,g2,g3 values to each GWXtreme eos
        
        if eos in pap_list: # if we have the "old" parameters for an eos, use them, otherwise use presets

            log_p0_SI,g1,g2,g3 = p_eos_val[eos]

        else:

            log_p0_SI,g1,g2,g3 = 33.000,2.500,2.500,2.500

        # range and step size of parameter space is hardcoded for now (easier to read)
        p0_range = np.arange(log_p0_SI-.5,log_p0_SI+.5,.001)
        g1_range = np.arange(g1-1.5,g1+1.5,.001)
        g2_range = np.arange(g2-1.5,g2+1.5,.001)
        g3_range = np.arange(g3-1.5,g3+1.5,.001)

        eos_pointer = lalsim.SimNeutronStarEOSByName(eos)
        fam_pointer = lalsim.CreateSimNeutronStarFamily(eos_pointer)
        min_mass = lalsim.SimNeutronStarFamMinimumMass(fam_pointer)/lal.MSUN_SI

        s, _, Lambdas, max_mass = modsel.getEoSInterp(eosname=eos, m_min=min_mass)
        target_masses = np.linspace(min_mass,max_mass,N)
        target_Lambdas = s(target_masses)
        target_lambdas = (Lambdas / lal.G_SI) * ((target_masses * lal.MRSUN_SI) ** 5) # essential to r^2 calculation

        # METROPOLIS-HASTINGS 
        
        no_errors == False

        while no_errors == False:

            p0_choice1, g1_choice1, g2_choice1, g3_choice1 = (np.random.choice(p0_range), np.random.choice(g1_range)
                                                             ,np.random.choice(g2_range), np.random.choice(g3_range))

            try: 

                L1 = likelihood(p0_choice1,g1_choice1,g2_choice1,g3_choice1,min_mass,max_mass,N,target_lambdas)
                no_errors = True # if L1 doesn't give an error, the while loop will end

            # Can run into RuntimeError from (don't remember)
            # Can run into ValueError from the use of lal's piecewise function (I think)
            except RuntimeError: continue
            except ValueError: continue

        post_p0 = []
        post_g1 = []
        post_g2 = []
        post_g3 = []
        
        while len(post_p0) <= transitions:

            p0_choice2, g1_choice2, g2_choice2, g3_choice2 = (np.random.choice(p0_range), np.random.choice(g1_range)
                                                             ,np.random.choice(g2_range), np.random.choice(g3_range))

            try: L2 = likelihood(p0_choice2,g1_choice2,g2_choice2,g3_choice2,min_mass,max_mass,N,target_lambdas) # if L2 gives an error it'll keep trying

            except RuntimeError: continue
            except ValueError: continue

            if L2/L1 >= np.random.random():

                p0_choice1 = p0_choice2
                g1_choice1 = g1_choice2
                g2_choice1 = g2_choice2
                g3_choice1 = g3_choice2
            
            # current eos' p0,g1,g2,g3 combination is stored (can then see what parameter combinations lasts the "longest")
            post_p0.append(p0_choice1)
            post_g1.append(g1_choice1)
            post_g2.append(g2_choice1)
            post_g3.append(g3_choice1)
       
        # each eos parameter distribution is stored and can be found using the eos' index
        eos_post_p0.append(post_p0)
        eos_post_g1.append(post_g1)
        eos_post_g2.append(post_g2)
        eos_post_g3.append(post_g3)

data = {"p0" : eos_post_p0, "g1" : eos_post_g1, "g2" : eos_post_g2, "g3" : eos_post_g3}
outputfile = "results/{}.json".format(filename)
with open(outputfile, "w") as f:
    json.dump(data, f, indent=2, sort_keys=True)
