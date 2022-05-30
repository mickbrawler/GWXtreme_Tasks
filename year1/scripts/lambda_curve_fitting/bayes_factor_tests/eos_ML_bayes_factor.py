from GWXtreme import eos_model_selection as ems
import lal
import numpy as np
import os
import json

modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")

def mass_lambda_bfactor(N):
    
    # Obtains bayes factor of each target eos using mass/lambda files

    target_eos = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","HQC18","SLY2","SLY230A",
                  "SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6",
                  "SK272","SKI3","SKI5","MPA1","MS1B_PP","MS1_PP"]
   
    refer_eos = "SLY"
   
    t_dir = "mass_lambda_files/target_eos/"

    r_dir = "mass_lambda_files/refer_eos/"

    for eos in target_eos:
        
        min_mass = modsel.getMinMass(eos) # Small function I added to the GWXtreme script that returns an eos's min_mass
        s, _, _, max_mass = modsel.getEoSInterp(eos, m_min=min_mass, N=1000)
        masses = np.linspace(min_mass,max_mass,N)
        Lambdas = s(masses)
        lambdas = (Lambdas / lal.G_SI) * ((masses * lal.MRSUN_SI) ** 5)

        output = np.vstack((masses, lambdas)).T
        outputfile = t_dir+eos+'.txt'
        np.savetxt(outputfile, output, fmt="%f\t%f")
    
    min_mass = modsel.getMinMass(refer_eos)
    s, _, _, max_mass = modsel.getEoSInterp(refer_eos, m_min=min_mass, N=1000)
    masses = np.linspace(min_mass,max_mass,N)
    Lambdas = s(masses)
    lambdas = (Lambdas / lal.G_SI) * ((masses * lal.MRSUN_SI) ** 5)

    output = np.vstack((masses, lambdas)).T
    outputfile = r_dir+refer_eos+'.txt'
    np.savetxt(outputfile, output, fmt="%f\t%f")
    
    bayes_factors = [] 
    for tfile in target_eos:
        bayes_factor = modsel.computeEvidenceRatio(EoS1=t_dir+tfile+'.txt', EoS2=r_dir+refer_eos+'.txt')
        bayes_factors.append(bayes_factor)

    dictionary = {"bayes factors": bayes_factors}

    with open("results/bayes_factors_ML_SLY.json", "w") as f:
        json.dump(dictionary, f, indent=2, sort_keys=True)

