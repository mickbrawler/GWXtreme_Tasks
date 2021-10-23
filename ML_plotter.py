from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import json

# list of eos GWXtreme can work with off the cuff
GWX_list = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","HQC18","SLY2","SLY230A",
            "SKMP","RS","SK255","SLY9","APR4_EPP","SKI2","SKI4","SKI6",
            "SK272","SKI3","SKI5","MPA1","MS1B_PP","MS1_PP","BBB2","AP4",
            "MPA1","MS1B","MS1","SLY"]

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
             ,"MS1B":[33.855,3.456,3.011,1.425]
             ,"PS":[33.671,2.216,1.640,2.365]
             ,"GS1A":[33.504,2.350,1.267,2.421]
             ,"GS2A":[33.642,2.519,1.571,2.314]
             ,"BGN1H1":[33.623,3.258,1.472,2.464]
             ,"GNH3":[33.648,2.664,2.194,2.304]
             ,"H1":[33.564,2.595,1.845,1.897]
             ,"H2":[33.617,2.775,1.855,1.858]
             ,"H3":[33.646,2.787,1.951,1.901]
             ,"H4":[33.669,2.909,2.246,2.144 ]
             ,"H5":[33.609,2.793,1.974,1.915]
             ,"H6A":[33.593,2.637,2.121,2.064]
             ,"H7":[33.559,2.621,2.048,2.006 ]
             ,"PCL2":[33.507,2.554,1.880,1.977]
             ,"ALF1":[33.055,2.013,3.389,2.033]
             ,"ALF2":[33.616,4.070,2.411,1.890]
             ,"ALF3":[33.283,2.883,2.653,1.952]
             ,"ALF4":[33.314,3.009,3.438,1.803]
             ,"SLY":[33.384,3.005,2.988,2.851]}

def plot_from_piecewise(p1,g1,g2,g3,N):

    eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(p1,g1,g2,g3) 
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    m_min = lalsim.SimNeutronStarFamMinimumMass(fam)/lal.MSUN_SI
    max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI

    # This is necessary so that interpolant is computed over the full range
    # Keeping number upto 3 decimal places
    # Not rounding up, since that will lead to RuntimeError
    max_mass = int(max_mass*1000)/1000
    masses = np.linspace(m_min, max_mass, N)
    masses = masses[masses <= max_mass]
    Lambdas = []
    gravMass = []
    for m in masses:
        try:
            rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
            kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
            cc = m*lal.MRSUN_SI/rr
            Lambdas = np.append(Lambdas, (2/3)*kk/(cc**5))
            gravMass = np.append(gravMass, m)
        except RuntimeError:
            break
    Lambdas = np.array(Lambdas)
    gravMass = np.array(gravMass)
#    s = interp1d(gravMass, Lambdas)
    return(masses,Lambdas)

def plot_from_lal(eosname,N):
    
    eos = lalsim.SimNeutronStarEOSByName(eosname)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    m_min = lalsim.SimNeutronStarFamMinimumMass(fam)/lal.MSUN_SI
    max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI

    # This is necessary so that interpolant is computed over the full range
    # Keeping number upto 3 decimal places
    # Not rounding up, since that will lead to RuntimeError
    max_mass = int(max_mass*1000)/1000
    masses = np.linspace(m_min, max_mass, N)
    masses = masses[masses <= max_mass]
    Lambdas = []
    gravMass = []
    for m in masses:
        try:
            rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
            kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
            cc = m*lal.MRSUN_SI/rr
            Lambdas = np.append(Lambdas, (2/3)*kk/(cc**5))
            gravMass = np.append(gravMass, m)
        except RuntimeError:
            break
    Lambdas = np.array(Lambdas)
    gravMass = np.array(gravMass)
#    s = interp1d(gravMass, Lambdas)
    return(masses,Lambdas)

def plotter(eosname,N):

    print(eosname)

    if eosname in pap_list:

        p1,g1,g2,g3 = p_eos_val[eosname]
        pap_masses, pap_Lambdas = plot_from_piecewise(p1,g1,g2,g3,N)
        pl.plot(pap_masses,pap_Lambdas,label="Paper_piecewise")

    lal_masses, lal_Lambdas = plot_from_lal(eosname,N)
    pl.plot(lal_masses,lal_Lambdas,label="lal")

    p1,g1,g2,g3 = m_eos_val[eosname]
    pw_masses, pw_Lambdas = plot_from_piecewise(p1,g1,g2,g3,N)
    pl.plot(pw_masses,pw_Lambdas,label="MCMC_piecewise")

    pl.legend()
    pl.xlabel("Masses")
    pl.ylabel("$\\Lambda$")
    pl.savefig("plots/mass_lambda_plots/{}_comparison.png".format(eosname))

def plotter_runner(N):
    # Meant to get ML plots for every GWXtreme eos

    for eos in GWX_list:

        plotter(eos,N)
        pl.clf()

def eos_GWXtreme_kde_plot(filename):
    # Plots each eos' input options' kde plots using GWXtreme's plot_func

    modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")
        
    # MCMC results
    with open(filename,"r") as f:
        m_eos_val = json.load(f)

    for eos in m_eos_val:

        outputfile = "Plots/kde_plots/kde_{}.png".format(eos)

        if eos in pap_list:
            
            p_p1,p_g1,p_g2,p_g3 = p_eos_val[eos]
            m_p1,m_g1,m_g2,m_g3,_ = m_eos_val[eos]
            modsel.plot_func([eos,[m_p1,m_g1,m_g2,m_g3],[p_p1,p_g1,p_g2,p_g3]],filename=outputfile)

        else:

            m_p1,m_g1,m_g2,m_g3,_ = m_eos_val[eos]
            modsel.plot_func([eos,[m_p1,m_g1,m_g2,m_g3]],filename=outputfile)
