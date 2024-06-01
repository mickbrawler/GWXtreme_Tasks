from GWXtreme import eos_model_selection as ems
from GWXtreme.parametrized_eos_sampler import mcmc_sampler
from GWXtreme.eos_prior import is_valid_eos,eos_p_of_rho, spectral_eos,polytrope_eos
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path
import h5py
import emcee as mc
import glob
from multiprocessing import cpu_count, Pool

def calcConstraint1():
    # Adopted from the driver provided in GWXtreme's git. That said it may be restrictive.
    # I don't think I can alter the "look" of the plot, and overlaying different constraints
    # may be harder than the simple use of fig.show(). Old script used to this.

    uLTs_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Taylor/uniformP_LTs/phenom-injections/TaylorF2"
    uLs_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Taylor/uniformP_Ls/IMRPhenomPv2_NRTidal/APR4_EPP"
    phenomPhenom_Dir = "../../year2/bilby_runs/simulations/outdir/1st_Phenom_Phenom/IMRPhenomPv2_NRTidal/APR4_EPP"

    uLTs_Files = glob.glob("{}/*/*simplified.json".format(uLTs_Dir)) + glob.glob("{}/troublesome/*/*simplified.json".format(uLTs_Dir))
    uLs_Files = glob.glob("{}/*/*simplified.json".format(uLs_Dir)) + glob.glob("{}/troublesome/*/*simplified.json".format(uLs_Dir))
    phenomPhenom_Files = glob.glob("{}/*/*simplified.json".format(phenomPhenom_Dir)) + glob.glob("{}/troublesome/*/*simplified.json".format(phenomPhenom_Dir))
    #filesToCompare = [uLTs_Files,uLs_Files,phenomPhenom_Files]
    filesToCompare = [uLTs_Files,phenomPhenom_Files] # Tester !!!

    #Labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT"]
    Labels = ["2D-KDE-TaylorF2", "3D-KDE-PhenomNRT"] # Tester !!!
    dims = [2,3,3]

    for ii in range(len(Labels)):

        #fnames=filesToCompare[ii]
        fnames=filesToCompare[ii][:2] # Tester!!!
        #Name of/ Path to file in which EoS parameter posterior samples will be saved:
        outname='data/constraints/{}_16simulationsInference'.format(Labels[ii])

        #Initialize Sampler Object:
        """For SPectral"""
        #sampler=mcmc_sampler(fnames, {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}, outname, nwalkers=100, Nsamples=10000, ndim=4, spectral=True,npool=100,kdedim=dims[ii])
        # Tester !!!
        sampler=mcmc_sampler(fnames, {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}, outname, nwalkers=10, Nsamples=1000, ndim=4, spectral=True,npool=100,kdedim=dims[ii])

        #Run, Save , Plot
        sampler.initialize_walkers()
        sampler.run_sampler()
        sampler.save_data()

        fig=sampler.plot(cornerplot={'plot':True,'true vals':None},p_vs_rho={'plot':True,'true_eos':'AP4'})
        # We follow the driver's logic that saves a constraint and corner plot cause... why not
        fig['corner'].savefig('plots/corners/{}_16simulations_corner.png'.format(Labels[ii]))
        fig['p_vs_rho'][0].savefig('plots/constraints/{}_16simulations_constraint.png'.format(Labels[ii]))


def calcConstraint2(burn_in_frac=0.5,thinning=None):
    # Adopted from Anarya's GWXtreme 3d kde prod branch's plotting logic.

    #Labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT"]
    Labels = ["2D-KDE-TaylorF2", "3D-KDE-PhenomNRT"] # Tester !!!
    for label in Labels:
        # Load the samples
        filename='data/constraints/{}_16simulationsInference.h5'.format(label)
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
        np.savetxt("data/constraints/{}_16simulationsInference.txt".format(label),np.array([rho,logp_CIlow,logp_med,logp_CIup]).T)


def plotConstraint():
    # Adopted from Anarya's GWXtreme 3d kde prod branch's plotting logic.

    #labels = ["2D-KDE-TaylorF2", "3D-KDE-TaylorF2", "3D-KDE-PhenomNRT"]
    labels = ["2D-KDE-TaylorF2", "3D-KDE-PhenomNRT"]
    #Labels = ["2D KDE TaylorF2", "3D KDE TaylorF2", "3D KDE PhenomNRT"]
    Labels = ["2D KDE TaylorF2", "3D KDE PhenomNRT"]
    #Colors = ["#d7191c","#fdae61","#abdda4"]
    Colors = ["#d7191c","#abdda4"]

    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    plt.rc('xtick', direction='out', color='black')
    plt.rc('ytick', direction='out', color='black')
    plt.rc('lines', linewidth=2)

    for label, Label, Color in zip(labels,Labels,Colors): # increment over each plot file

        # Load the samples
        filename='data/constraints/{}_16simulationsInference.txt'.format(label)
        rho, lower_bound, median, upper_bound = np.loadtxt(filename).T

        #plt.plot(lower_bound, rho, label=Label, color=Color)
        #plt.plot(upper_bound, rho, color=Color)
        plt.fill_between(np.log10(rho), lower_bound, upper_bound, color=Color, alpha=0.45, label=Label, zorder=1.)

    plt.xlim([16.99, 18.25])
    plt.xlabel(r'$\log10{\frac{\rho}{g cm^-3}}$',fontsize=20)
    plt.ylabel(r'$log10(\frac{p}{dyne cm^{-2}})$',fontsize=20)
    plt.legend()
    plt.savefig("plots/constraints/16simulations_constraint.png", bbox_inches='tight')

