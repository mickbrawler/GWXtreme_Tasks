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
            self.keys = ['gamma1','gamma2','gamma3','gamma4']
            self.saves = {"APR4_EPP":[0.3575207975626009,0.5236716767115328,-0.08707568613208228,0.004157158155809634],
                          "H4":[1.256759939199018,0.05631944082170803,-0.037309233973168654,0.0024141067570869618],
                          "MS1":[1.4430958964746519,-0.0741556716354066,0.014663981143735905,-0.0015881224822786415]}
        else:
            self.priorbounds = {'logP':{'params':{"min":33.6,"max":34.5}},
                                'gamma1':{'params':{"min":2.0,"max":4.5}},
                                'gamma2':{'params':{"min":1.1,"max":4.5}},
                                'gamma3':{'params':{"min":1.1,"max":4.5}}}
            self.keys = ["logP", "gamma1", "gamma2", "gamma3"]
            self.saves = {"APR4_EPP":[33.60346924973564,4.072869518144783,2.0191831885299654,2.9584522656818013],
                          "H4":[33.736513994975844,3.264482560445284,2.4717407116976267,2.536349314864326],
                          "MS1":[33.90773273959166,3.1657048629954563,3.2515816053960584,2.4586302197718153]}

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
        while len(post_1) < (self.samples):

            if checkpoint:
                p1_choice2, p2_choice2, p3_choice2, p4_choice2 = self.saves[eos_name]
            else:
                p1_choice2 = np.random.uniform(low=self.low_g1_p1,high=self.high_g1_p1)
                p2_choice2 = np.random.uniform(low=self.low_g2_g1,high=self.high_g2_g1)
                p3_choice2 = np.random.uniform(low=self.low_g3_g2,high=self.high_g3_g2)
                p4_choice2 = np.random.uniform(low=self.low_g4_g3,high=self.high_g4_g3)

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
        self.bestfit = {eos_name:self.bestfit_samples}
    
    def run_over_EoS(self, plot_dir, outfile):

        bestfits = {}
        EoS_names = ["APR4_EPP","H4","MS1"] 
        for EoS_name in EoS_names:
            self.run_MCMC(EoS_name)
            bestfits.update(self.bestfit)
            self.modsel.plot_func([EoS_name,self.bestfit_samples],filename="files/basic_plots/{}{}.png".format(plot_dir,EoS_name))
        with open(outfile, "w") as f: json.dump(bestfits, f, indent=2, sort_keys=True)

