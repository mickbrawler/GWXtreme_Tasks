from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import json

GWX_list = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","HQC18","SLY2",
            "SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP",
             "SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1",
             "MS1B_PP","MS1_PP","BBB2","AP4","MPA1","MS1B","MS1",
             "SLY"]

p_eos_val = {"AP4":[33.269,2.830,3.445,3.348]
            ,"BBB2":[33.331,3.418,2.835,2.832]             
            ,"MPA1":[33.495,3.446,3.572,2.887]
            ,"MS1":[33.858,3.224,3.033,1.325]
            ,"SLY":[33.384,3.005,2.988,2.851]}

modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")

# Produces liklihood from paper values and MCMC results

def get_likelihoods(N, filename, outputfile):
    # Gets likelihood values of lal and piecewise polytrope eos

    # get MCMC results dictionary
    with open(filename, "r") as f:
        data = json.load(f)

    eos_likelihoods = {}
    for eos in data:

        # get MCMC parameters per eos
        m_p1,m_g1,m_g2,m_g3,_ = data[eos]

        if eos in p_eos_val:

            eos_pointer = lalsim.SimNeutronStarEOSByName(eos)
            fam_pointer = lalsim.CreateSimNeutronStarFamily(eos_pointer)
            min_mass = lalsim.SimNeutronStarFamMinimumMass(fam_pointer)/lal.MSUN_SI
            s, _, _, max_mass = modsel.getEoSInterp(eosname=eos, m_min=min_mass)
            target_masses = np.linspace(min_mass,max_mass,N)
            target_Lambdas = s(target_masses)
            target_lambdas = (target_Lambdas / lal.G_SI) * ((target_masses * lal.MRSUN_SI) ** 5) 
                
            # Produces r2 value between lal and piecewise lambdas
            s, min_mass, max_mass = modsel.getEoSInterpFrom_piecewise(m_p1,m_g1,m_g2,m_g3)
            trial_masses = np.linspace(min_mass,max_mass,N)
            trial_Lambdas = s(trial_masses)
            trial_lambdas = (trial_Lambdas / lal.G_SI) * ((trial_masses * lal.MRSUN_SI) ** 5) # lambdas the eos produced via getEosInterpFrom_piecewise.
            m_r_val = 1 / np.log(np.sum((target_lambdas - trial_lambdas) ** 2)) # r^2 value
                   
            p_p1,p_g1,p_g2,p_g3 = p_eos_val[eos]
            s, min_mass, max_mass = modsel.getEoSInterpFrom_piecewise(p_p1,p_g1,p_g2,p_g3)
            trial_masses = np.linspace(min_mass,max_mass,N)
            trial_Lambdas = s(trial_masses)
            trial_lambdas = (trial_Lambdas / lal.G_SI) * ((trial_masses * lal.MRSUN_SI) ** 5) # lambdas the eos produced via getEosInterpFrom_piecewise.
            p_r_val = 1 / np.log(np.sum((target_lambdas - trial_lambdas) ** 2)) # r^2 value
                   
            eos_likelihoods.update({eos:[m_r_val,p_r_val]})

        else:

            eos_pointer = lalsim.SimNeutronStarEOSByName(eos)
            fam_pointer = lalsim.CreateSimNeutronStarFamily(eos_pointer)
            min_mass = lalsim.SimNeutronStarFamMinimumMass(fam_pointer)/lal.MSUN_SI
            s, _, _, max_mass = modsel.getEoSInterp(eosname=eos, m_min=min_mass)
            target_masses = np.linspace(min_mass,max_mass,N)
            target_Lambdas = s(target_masses)
            target_lambdas = (target_Lambdas / lal.G_SI) * ((target_masses * lal.MRSUN_SI) ** 5) 
                
            # Produces r2 value between lal and piecewise lambdas
            s, min_mass, max_mass = modsel.getEoSInterpFrom_piecewise(m_p1,m_g1,m_g2,m_g3)
            trial_masses = np.linspace(min_mass,max_mass,N)
            trial_Lambdas = s(trial_masses)
            trial_lambdas = (trial_Lambdas / lal.G_SI) * ((trial_masses * lal.MRSUN_SI) ** 5) # lambdas the eos produced via getEosInterpFrom_piecewise.
            m_r_val = 1 / np.log(np.sum((target_lambdas - trial_lambdas) ** 2)) # r^2 value
                   
            eos_likelihoods.update({eos:[m_r_val]})

    with open(outputfile, "w") as f:
        json.dump(eos_likelihoods, f, indent=2, sort_keys=True)
