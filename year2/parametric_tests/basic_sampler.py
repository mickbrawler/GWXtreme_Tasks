from GWXtreme import eos_model_selection as ems
from GWXtreme import eos_prior as ep
import lalsimulation as lalsim
import lal
import numpy as np
import json

class param_distro:

    def __init__(self, N, samples, min_mass=1.0, spectral=True):

        self.N = N
        self.samples = samples
        self.min_mass = min_mass
        self.spectral = spectral
        self.modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat", spectral=self.spectral)
        self.priorbounds = {'gamma1':{'params':{"min":0.2,"max":2.00}},
                            'gamma2':{'params':{"min":-1.6,"max":1.7}},
                            'gamma3':{'params':{"min":-0.6,"max":0.6}},
                            'gamma4':{'params':{"min":-0.02,"max":0.02}}}
        self.keys = ['gamma1','gamma2','gamma3','gamma4']

    def likelihood(self, g1_p1, g2_g1, g3_g2, g4_g3):
        # Produces r2 value between lal and parametrized lambdas
        
        parameters = [g1_p1, g2_g1, g3_g2, g4_g3]
        params = {k:np.array([par]) for k,par in zip(self.keys,parameters)}

        try:
            if not ep.is_valid_eos(params, self.priorbounds, spectral=self.spectral):
                return -np.inf
            s, _, max_mass = self.modsel.getEoSInterp_parametrized(parameters)
            trial_masses = np.linspace(self.min_mass,max_mass,self.N)
            trial_Lambdas = s(trial_masses)
            trial_lambdas = (trial_Lambdas / lal.G_SI) * ((trial_masses * lal.MRSUN_SI) ** 5) 
            r_val = 1 / np.log(np.sum((self.target_lambdas - trial_lambdas) ** 2))
            return r_val
        except:
            return -np.inf

    def run_MCMC(self, eos_name):

        s, _, _, max_mass = self.modsel.getEoSInterp(eosname=eos_name, m_min=self.min_mass)
        target_masses = np.linspace(self.min_mass,max_mass,self.N)
        target_Lambdas = s(target_masses)
        self.target_lambdas = (target_Lambdas / lal.G_SI) * ((target_masses * lal.MRSUN_SI) ** 5)

        # METROPOLIS-HASTINGS  
        no_errors = False
        while no_errors == False:

            p1_choice1 = np.random.uniform(low=0.2,high=2.0)
            p2_choice1 = np.random.uniform(low=-1.6,high=1.6)
            p3_choice1 = np.random.uniform(low=-0.6,high=0.6)
            p4_choice1 = np.random.uniform(low=-0.02,high=0.02)

            L1 = self.likelihood(p1_choice1,p2_choice1,p3_choice1,p4_choice1)
            if L1 != -np.inf: no_errors = True # if L1 doesn't give an error, the while loop will end

        post_1 = []
        post_2 = []
        post_3 = []
        post_4 = []
        post_r2 = []
        while len(post_1) < (self.samples):

            p1_choice2 = np.random.uniform(low=0.2,high=2.0)
            p2_choice2 = np.random.uniform(low=-1.6,high=1.6)
            p3_choice2 = np.random.uniform(low=-0.6,high=0.6)
            p4_choice2 = np.random.uniform(low=-0.02,high=0.02)

            L2 = self.likelihood(p1_choice2,p2_choice2,p3_choice2,p4_choice2)

            if L2/L1 >= np.random.random():
                p1_choice1 = p1_choice2
                p2_choice1 = p2_choice2
                p3_choice1 = p3_choice2
                p4_choice1 = p4_choice2
                post_r2.append(L2) # if choice2s are better, append their likelihood
                
            else:
                post_r2.append(L1) # otherwise choice1s are better, so their likelihood is appended instead
            
            # current eos' p1,g1,g2,g3 combination is stored (can then see what parameter combinations lasts the "longest")
            post_1.append(p1_choice1)
            post_2.append(p2_choice1)
            post_3.append(p3_choice1)
            post_4.append(p4_choice1)

        max_index = np.argmax(post_r2)
        self.bestfit_samples = [post_1[max_index],post_2[max_index],post_3[max_index],post_4[max_index]]
        bestfit = {eos_name:self.bestfit_samples}
        with open("testing_bestfit.json", "w") as f: json.dump(bestfit, f, indent=2, sort_keys=True)
        self.modsel.plot_func([eos_name, self.bestfit_samples],filename="testing_bestfit.png")

