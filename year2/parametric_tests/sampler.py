from GWXtreme import eos_model_selection as ems
from GWXtreme import eos_prior as ep
from multiprocessing import cpu_count, Pool
import emcee as mc
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import json
import math

# Class designed to find bestfit parameters for multiple EoS

class mcmc_sampler():
    def __init__(self, N=1000, nwalkers=10, nsamples=5000, spectral=True):
        '''
        Samples in the mass-lambda parameter space
        using the specified parametrized EoS in search
        of the best fit to a proposed EoS.

        N   ::  Number of points making up each sample's mass-lambda curve

        nwalkers    ::  Number of walkers to use for mcmc

        Nsamples    ::  Number of samples to use

        spectral    ::  Choice of spectral or piecewise parametrization
        '''
        
        self.N = N
        self.nwalkers = nwalkers
        self.nsamples = nsamples
        self.spectral = spectral
        self.ndim = 4
        self.pool = 64
#        self.EoS_names = ['APR4_EPP', 'BHF_BBB2', 'H4', 'HQC18',
#                          'KDE0V', 'KDE0V1', 'MPA1', 'MS1B_PP',
#                          'MS1_PP', 'RS', 'SK255', 'SK272',
#                          'SKI2', 'SKI3', 'SKI4', 'SKI5', 'SKI6',
#                          'SKMP', 'SKOP', 'SLY9', 'WFF1']

        self.EoS_names = ['APR4_EPP']
        if spectral:
            self.priorbounds = {'gamma1':{'params':{"min":0.2,"max":2.00}},
                                 'gamma2':{'params':{"min":-1.6,"max":1.7}},
                                 'gamma3':{'params':{"min":-0.6,"max":0.6}},
                                'gamma4':{'params':{"min":-0.02,"max":0.02}}}
            self.keys = ["gamma1", "gamma2", "gamma3", "gamma4"]
        else:
            self.priorbounds = {'logP':{'params':{"min":33.6,"max":34.5}},
                                'gamma1':{'params':{"min":2.0,"max":4.5}},
                                'gamma2':{'params':{"min":1.1,"max":4.5}},
                                'gamma3':{'params':{"min":1.1,"max":4.5}}}
            self.keys = ["logP", "gamma1", "gamma2", "gamma3"]
        self.modsel = ems.Model_selection("posterior_samples/posterior_samples_narrow_spin_prior.dat", spectral=self.spectral)
    
    def target_eos_values(self, eos):
        '''
        Gets target EoS' lambda values for a range of masses

        eos ::  name of tabulated EoS
        '''

        eos_pointer = lalsim.SimNeutronStarEOSByName(eos)
        fam_pointer = lalsim.CreateSimNeutronStarFamily(eos_pointer)
        min_mass = lalsim.SimNeutronStarFamMinimumMass(fam_pointer)/lal.MSUN_SI
        s, _, _, max_mass = self.modsel.getEoSInterp(eosname=eos, m_min=min_mass)
        target_masses = np.linspace(min_mass,max_mass,self.N)
        target_Lambdas = s(target_masses)
        self.target_lambdas = (target_Lambdas / lal.G_SI) * ((target_masses * lal.MRSUN_SI) ** 5)

    def log_likelihood(self, parameters):
        '''
        Obtains inverse of r-squared value of target and sample lambda values

        parameters  ::  4 parameters of EoS
        '''

        g1_p1, g2, g3, g4 = parameters
        s, min_mass, max_mass = self.modsel.getEoSInterp_parametrized([g1_p1,g2,g3,g4])
        trial_masses = np.linspace(min_mass,max_mass,self.N)
        trial_Lambdas = s(trial_masses)
        trial_lambdas = (trial_Lambdas / lal.G_SI) * ((trial_masses * lal.MRSUN_SI) ** 5)
        r_val = - math.log(np.sum((self.target_lambdas - trial_lambdas) ** 2))
        return r_val
            
    def log_posterior(self, parameters):
        '''
        If EoS is valid (doesn't produce an error) its likelihood is calcuated.

        parameters  ::  4 parameters of EoS
        '''

        params = {k:np.array([par]) for k,par in zip(self.keys,parameters)}

        if not ep.is_valid_eos(params, self.priorbounds, spectral=self.spectral):
            return -np.inf

        return self.log_likelihood(parameters)

    def initialize_walkers(self):
        '''
        Produce random starting points for each walker within bounds of 
        parameter space and that don't produce other errors.
        '''

        n=0
        p0=[]
        while True:
            g=np.array([np.random.uniform(self.priorbounds[k]["params"]["min"],self.priorbounds[k]["params"]["max"]) for k in self.keys])
            params={k:np.array([g[i]]) for i,k in enumerate(self.keys)}

            if(ep.is_valid_eos(params,self.priorbounds,spectral=self.spectral)):
                p0.append(g)
                n+=1
            if(n>=self.nwalkers):
                break

        self.p0=p0

    def run_sampler(self):

        self.target_eos_values(self.EoS)
        self.initialize_walkers()

        with Pool(self.pool) as pool:
            sampler=mc.EnsembleSampler(self.nwalkers,self.ndim,self.log_posterior,pool=pool)
            sampler.run_mcmc(self.p0,self.nsamples,progress=True)

        self.flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    def over_all_EoS(self, Dir):
        '''
        Get chain for each EoS.

        Dir ::  Directory name for each EoS samples file (include /)
        '''

        self.Dir = Dir
        
        for EoS_name in self.EoS_names:
            self.EoS = EoS_name
            self.run_sampler()
            outfile = Dir + EoS_name + ".txt"
            np.savetxt(outfile,self.flat_samples)

    def max_likelihood(self, outfile, EoS_chains_Dir=None):
        '''
        Get max likelihood sample for each EoS

        outfile ::  .json file name for bestfit EoS results

        EoS_chains_Dir ::  EoS chains directory in case sampler hasn't been run
        '''
        
        if EoS_chains_Dir != None:
            self.Dir = EoS_chains_Dir

        self.bestfit_EoS = {}
        for EoS_name in self.EoS_names:
            self.target_eos_values(EoS_name)
            samples = np.loadtxt(self.Dir + EoS_name + ".txt")
            likelihoods = []
            for sample in samples:
                likelihood = self.log_likelihood(sample)
                print(likelihood)
                likelihoods.append(likelihood)
            self.bestfit_EoS.update({EoS_name:list(samples[np.argmax(likelihoods)])})

        with open(outfile, "w") as f:
            json.dump(self.bestfit_EoS, f, indent=2, sort_keys=True)

    def plot_kde_EoS(self, Dir, bestfit_EoS_file=None):
        '''
        Plot the target EoS and its best fit parameters.
        
        Dir ::  Directory name for plot files (include /)

        bestfit_EoS_file    ::  bestfit filename in case sampler hasn't been run
        '''
        
        if bestfit_EoS_file != None:
            with open(bestfit_EoS_file, "r") as f:
                self.bestfit_EoS = json.load(f)

        for EoS_name in self.EoS_names:
            outfile = Dir + EoS_name + ".png"
            self.modsel.plot_func([EoS_name, self.bestfit_EoS[EoS_name]],filename=outfile)
            
