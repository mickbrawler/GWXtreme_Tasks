from GWXtreme import eos_model_selection as ems
from GWXtreme.parametrized_eos_sampler import mcmc_sampler
from GWXtreme.eos_prior import is_valid_eos,eos_p_of_rho, spectral_eos,polytrope_eos
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path
import h5py
import emcee as mc
from multiprocessing import cpu_count, Pool
import lalsimulation
import lal

def calcConstraint1():
    # Adopted from the driver provided in GWXtreme's git. That said it may be restrictive.
    # I don't think I can alter the "look" of the plot, and overlaying different constraints
    # may be harder than the simple use of fig.show(). Old script used to this.

    #Array Containing list of paths to the .dat files  containing the posterior samples for the events:
    #uLTs_File = "/home/michael/projects/eos/GWXtreme_Tasks/year2/bilby_runs/simulations/outdir/real/uniformP_LTs/GW170817/simplified_result.json" 
    #uLs_File = "/home/michael/projects/eos/GWXtreme_Tasks/year3/GW170817_prior_L1L2/CIT_attempt_successful/outdir/simplified_result.json"
    uLs_phenom_File = "/home/michael/projects/eos/GWXtreme_Tasks/year3/lastStretch/files/BNS/GW170817phenom.json"
    #filesToCompare = [uLTs_File,uLs_File,uLs_phenom_File]
    filesToCompare = [uLs_phenom_File] # Trying just Phenom again

    #labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT"]
    labels = ["3D KDE PhenomNRT"]
    #Labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT"]
    Labels = ["3D-KDE-PhenomNRT"]
    #dims = [2,3,3]
    dims = [3]

    for ii in range(len(filesToCompare)):

        fnames=[filesToCompare[ii]]
        #Name of/ Path to file in which EoS parameter posterior samples will be saved:
        outname='data/BNS/constraints/{}_GW170817inference_10000samp'.format(Labels[ii])

        #Initialize Sampler Object:
        """For SPectral"""
        # LOGQ = False
        sampler=mcmc_sampler(fnames, {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}, outname, nwalkers=100, Nsamples=10000, ndim=4, spectral=True,npool=100,kdedim=dims[ii],logq=False)
        # LOGQ = True
        #sampler=mcmc_sampler(fnames, {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}, outname, nwalkers=100, Nsamples=10000, ndim=4, spectral=True,npool=100,kdedim=dims[ii],logq=True)

        #Run, Save , Plot
        sampler.initialize_walkers()
        sampler.run_sampler()
        sampler.save_data()

        fig=sampler.plot(cornerplot={'plot':True,'true vals':None},p_vs_rho={'plot':True,'true_eos':'AP4'})
        # We follow the driver's logic that saves a constraint and corner plot cause... why not
        fig['corner'].savefig('plots/BNS/corners/{}_GW170817_corner_10000samp.png'.format(Labels[ii]))
        fig['p_vs_rho'][0].savefig('plots/BNS/constraints/{}_GW170817_constraint_10000samp.png'.format(Labels[ii]))


def calcConstraint2(burn_in_frac=0.5,thinning=None):
    # Adopted from Anarya's GWXtreme 3d kde prod branch's plotting logic.

    #Labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT"]
    Labels = ["3D-KDE-PhenomNRT"]
    for label in Labels:
        # Load the samples
        filename='data/BNS/constraints/{}_GW170817inference_10000samp.h5'.format(label)
        with h5py.File(filename,'r') as f:
            Samples = np.array(f['chains'])
            logp = np.array(f['logp'])

        # "Clean" the samples
        Ns=Samples.shape
        burn_in=int(Ns[0]*burn_in_frac)
        samples=[]

        if thinning is None:
            thinning=int(Ns[0]/50.)

            try:
                thinning=int(max(mc.autocorr.integrated_time(Samples))/2.)
            except mc.autocorr.AutocorrError as e:
                print(e)

        for i in range(burn_in, Ns[0], thinning):
            for j in range(Ns[1]):
                samples.append(Samples[i,j,:])

        samples = np.array(samples)
        # Save gamma sample data
        np.savetxt("data/BNS/constraints/{}_GW170817inference_gammas_10000samp.txt".format(label),samples)

        # Turn into confidence interval data
        logp=[]
        rho=np.logspace(17.1,18.25,1000)

        for s in samples:
            params=(s[0], s[1], s[2], s[3])

            p=eos_p_of_rho(rho,spectral_eos(params))

            logp.append(p)

        logp=np.array(logp)
        logp_CIup=np.array([np.quantile(logp[:,i],0.95) for i in range(len(rho))])
        logp_CIlow=np.array([np.quantile(logp[:,i],0.05) for i in range(len(rho))])
        logp_med=np.array([np.quantile(logp[:,i],0.5) for i in range(len(rho))])

        # Save confidence interval data
        np.savetxt("data/BNS/constraints/{}_GW170817inference_10000samp.txt".format(label),np.array([rho,logp_CIlow,logp_med,logp_CIup]).T)


def CalcConstraint2():
    # Adopted from Anarya's GWXtreme 3d kde prod branch's plotting logic.
    # This function is specifically for the lalsim nest parametric EoS

    Label = "lalsim_nest-PhenomNRT"

    # Turn into confidence interval data
    logp=[]
    rho=np.logspace(17.1,18.25,1000)

    samples = np.loadtxt("files/BNS/{}_GW170817inference_gammas.txt".format(Label))
    print(len(samples))
    for s in samples:
        params=(s[0], s[1], s[2], s[3])

        p=eos_p_of_rho(rho,spectral_eos(params))

        logp.append(p)

    logp=np.array(logp)
    logp_CIup=np.array([np.quantile(logp[:,i],0.95) for i in range(len(rho))])
    logp_CIlow=np.array([np.quantile(logp[:,i],0.05) for i in range(len(rho))])
    logp_med=np.array([np.quantile(logp[:,i],0.5) for i in range(len(rho))])

    # Save confidence interval data
    np.savetxt("data/BNS/constraints/{}_GW170817inference.txt".format(Label),np.array([rho,logp_CIlow,logp_med,logp_CIup]).T)


def plotConstraint():
    # Adopted from Anarya's GWXtreme 3d kde prod branch's plotting logic.

    labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT"] # for three methods being compared
    #labels = ["3D-KDE-PhenomNRT", "lalsim_nest-PhenomNRT"]
    #labels = ["2D-KDE-TaylorF2", "3D-KDE-PhenomNRT", "lalsim_nest-PhenomNRT"]

    Labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE IMRPhenomPv2_NRTidal"] # for three methods being compared
    #Labels = ["3D KDE IMRPhenomPv2_NRTidal", "full-parameter-space Nest runs"]
    #Labels = ["2D KDE TaylorF2", "3D KDE IMRPhenomPv2_NRTidal", "full-parameter-space Nest runs"]

    Colors = ['#ffffb3','#bebada','#fb8072'] # for three methods being compared
    #Colors = ['#beaed4','#fdc086']
    #Colors = ['#ffffb3','#beaed4','#fdc086']

    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    #plt.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    plt.rc('xtick', direction='out', color='black')
    plt.rc('ytick', direction='out', color='black')
    plt.rc('lines', linewidth=2)

    Hatches = ["|","-",""] # for three methods being compared
    #Hatches = ["","x"]
    #Hatches = ["|","","x"]

    for label, Label, Color, Hatch in zip(labels,Labels,Colors,Hatches): # increment over each plot file

        # Load the samples
        filename='data/BNS/constraints/{}_GW170817inference_10000samp.txt'.format(label) # my files are called this
        if os.path.isfile(filename) != True: filename='data/BNS/constraints/{}_GW170817inference.txt'.format(label) # nest result is named differently
        rho, lower_bound, median, upper_bound = np.loadtxt(filename).T

        #plt.plot(lower_bound, rho, label=Label, color=Color)
        #plt.plot(upper_bound, rho, color=Color)
        plt.fill_between(np.log10(rho), lower_bound, upper_bound, color=Color, alpha=0.45, label=Label, zorder=1., hatch=Hatch)

    #EoSs = ["APR4_EPP","H4","SLY","MS1_PP"]
    #for EoS in EoSs:
    #    logp=eos_p_of_rho(rho,lalsimulation.SimNeutronStarEOSByName(EoS))
    #    plt.plot(np.log10(rho),logp, linewidth=2.0, label=EoS, alpha=0.35)

    plt.xlim([min(np.log10(rho)), 18.25])
    plt.xlabel(r'$\log10{\frac{\rho}{g cm^-3}}$',fontsize=20)
    plt.ylabel(r'$log10(\frac{p}{dyne cm^{-2}})$',fontsize=20)
    plt.legend()
    plt.savefig("plots/BNS/constraints/GW170817_constraint1_10000samp.png", bbox_inches='tight')
    #plt.savefig("plots/BNS/constraints/GW170817_constraint2_10000samp.png", bbox_inches='tight')
    #plt.savefig("plots/BNS/constraints/GW170817_constraint3_10000samp.png", bbox_inches='tight')

