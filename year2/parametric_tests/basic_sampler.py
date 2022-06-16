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

        if self.spectral:
            self.priorbounds = {'gamma1':{'params':{"min":0.2,"max":2.00}},
                                'gamma2':{'params':{"min":-1.6,"max":1.7}},
                                'gamma3':{'params':{"min":-0.6,"max":0.6}},
                                'gamma4':{'params':{"min":-0.02,"max":0.02}}}
            #self.priorbounds = {'gamma1':{'params':{"min":0.0,"max":2.2}},
            #                    'gamma2':{'params':{"min":-1.8,"max":1.8}},
            #                    'gamma3':{'params':{"min":-0.8,"max":0.8}},
            #                    'gamma4':{'params':{"min":-0.04,"max":0.04}}}
            self.keys = ['gamma1','gamma2','gamma3','gamma4']
            with open("files/basic_runs/1_spectral_EoS_bestfits.json", "r") as f:
                self.saves = json.load(f)
        else:
            #self.priorbounds = {'logP':{'params':{"min":33.6,"max":34.5}},
            #                    'gamma1':{'params':{"min":2.0,"max":4.5}},
            #                    'gamma2':{'params':{"min":1.1,"max":4.5}},
            #                    'gamma3':{'params':{"min":1.1,"max":4.5}}}
            self.priorbounds = {'logP':{'params':{"min":32.5,"max":34.5}},
                                'gamma1':{'params':{"min":1.5,"max":4.75}},
                                'gamma2':{'params':{"min":1.0,"max":4.75}},
                                'gamma3':{'params':{"min":1.0,"max":4.75}}}
            self.keys = ["logP", "gamma1", "gamma2", "gamma3"]
            with open("files/basic_runs/1_piecewise_EoS_bestfits.json", "r") as f:
                self.saves = json.load(f)

        self.low_g1_p1 = self.priorbounds[self.keys[0]]['params']['min']
        self.high_g1_p1 = self.priorbounds[self.keys[0]]['params']['max']
        self.low_g2_g1 = self.priorbounds[self.keys[1]]['params']['min']
        self.high_g2_g1 = self.priorbounds[self.keys[1]]['params']['max']
        self.low_g3_g2 = self.priorbounds[self.keys[2]]['params']['min']
        self.high_g3_g2 = self.priorbounds[self.keys[2]]['params']['max']
        self.low_g4_g3 = self.priorbounds[self.keys[3]]['params']['min']
        self.high_g4_g3 = self.priorbounds[self.keys[3]]['params']['max']

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

    def run_MCMC(self, eos_name, checkpoint=True):

        s, _, _, max_mass = self.modsel.getEoSInterp(eosname=eos_name, m_min=self.min_mass)
        target_masses = np.linspace(self.min_mass,max_mass,self.N)
        target_Lambdas = s(target_masses)
        self.target_lambdas = (target_Lambdas / lal.G_SI) * ((target_masses * lal.MRSUN_SI) ** 5)

        # METROPOLIS-HASTINGS  
        no_errors = False
        while no_errors == False:

            if checkpoint:
                p1_choice1, p2_choice1, p3_choice1, p4_choice1 = self.saves[eos_name]
            else:
                p1_choice1 = np.random.uniform(low=self.low_g1_p1,high=self.high_g1_p1)
                p2_choice1 = np.random.uniform(low=self.low_g2_g1,high=self.high_g2_g1)
                p3_choice1 = np.random.uniform(low=self.low_g3_g2,high=self.high_g3_g2)
                p4_choice1 = np.random.uniform(low=self.low_g4_g3,high=self.high_g4_g3)

            L1 = self.likelihood(p1_choice1,p2_choice1,p3_choice1,p4_choice1)
            if L1 != -np.inf: no_errors = True # if L1 doesn't give an error, the while loop will end

        post_1 = []
        post_2 = []
        post_3 = []
        post_4 = []
        post_r2 = []
        sample_count = 0
        while sample_count < self.samples:

            print(sample_count)
            p1_choice2 = np.random.uniform(low=self.low_g1_p1,high=self.high_g1_p1)
            p2_choice2 = np.random.uniform(low=self.low_g2_g1,high=self.high_g2_g1)
            p3_choice2 = np.random.uniform(low=self.low_g3_g2,high=self.high_g3_g2)
            p4_choice2 = np.random.uniform(low=self.low_g4_g3,high=self.high_g4_g3)

            L2 = self.likelihood(p1_choice2,p2_choice2,p3_choice2,p4_choice2)
            if L2 != -np.inf: sample_count += 1 # This prevents error prone samples from speeding our sampler

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
        self.bestfit = {eos_name:self.bestfit_samples}
    
    def run_over_EoS(self, plot_dir, outfile, checkpoint=True):

        bestfits = {}
        EoS_names = ["APR4_EPP","H4","MS1"] 
        for EoS_name in EoS_names:
            self.run_MCMC(EoS_name,checkpoint=checkpoint)
            bestfits.update(self.bestfit)
            self.modsel.plot_func([EoS_name,self.bestfit_samples],filename="files/basic_plots/{}{}.png".format(plot_dir,EoS_name))
        with open(outfile, "w") as f: json.dump(bestfits, f, indent=2, sort_keys=True)

