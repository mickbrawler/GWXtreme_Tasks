from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import json

# Sampler meant to save problematic samples.
class sample_quest:

    def __init__(self, N, samples, spectral=True):

        self.N = N
        self.samples = samples
        self.spectral = spectral
        self.eos = "APR4_EPP"
        self.modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat",spectral=spectral)

        if spectral == True:
            # NEED TO CALCULATE WHAT THESE WOULD BE 
            #{'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}
            #self.g1_p1_incr, self.g2_1_incr ,self.g3_2_incr, self.g4_3_incr = .9, 1.65, .6, .02
            self.g1_p1_incr, self.g2_1_incr ,self.g3_2_incr, self.g4_3_incr = 2.0, 2.5, 1.5, .1
            self.g1_p1, self.g2_1, self.g3_2, self.g4_3 = 1.1, .05, 0.0, 0.0
            Dir = "./contain_error_samples/spectral/"
        else:
            #self.g1_p1_incr, self.g2_1_incr ,self.g3_2_incr, self.g4_3_incr = .4575, .927, 1.1595, .9285 # defaults
            #self.g1_p1, self.g2_1, self.g3_2, self.g4_3 = 33.4305, 3.143, 2.6315, 2.7315 # defaults
            #{'logP':{'params':{"min":33.6,"max":34.5}},'gamma1':{'params':{"min":2.0,"max":4.5}},'gamma2':{'params':{"min":1.1,"max":4.5}},'gamma3':{'params':{"min":1.1,"max":4.5}}}
            #self.g1_p1_incr, self.g2_1_incr ,self.g3_2_incr, self.g4_3_incr = .45, 1.25, 1.7, 1.7
            self.g1_p1, self.g2_1, self.g3_2, self.g4_3 = 34.05, 3.125, 2.8, 2.8 
            self.g1_p1_incr, self.g2_1_incr ,self.g3_2_incr, self.g4_3_incr = 1.5, 2.5, 2.5, 2.5 # designed to fail
            Dir = "./contain_error_samples/piecewise/"

        # First two files have to be made using np.savetxt(filename,[emptylist])
        # Last file is made with a dictionary in form {"seg_fault":[emptylist]} saved to a json
        self.value_error_samples_file = Dir + "value_error_samples.txt"
        self.runtime_error_samples_file = Dir + "runtime_error_samples.txt"
        self.type_error_samples_file = Dir + "type_error_samples.txt"
        self.seg_fault_samples_file = Dir + "seg_fault_samples.json"

    def likelihood(self, log_p1_SI, g1, g2, g3):
        # Produces r2 value between lal and piecewise lambdas

        s, min_mass, max_mass = self.modsel.getEoSInterp_parametrized([log_p1_SI,g1,g2,g3])
        trial_masses = np.linspace(min_mass,max_mass,self.N)
        trial_Lambdas = s(trial_masses)
        trial_lambdas = (trial_Lambdas / lal.G_SI) * ((trial_masses * lal.MRSUN_SI) ** 5) # lambdas the eos produced via getEosInterpFrom_piecewise.
        r_val = 1 / np.log(np.sum((self.target_lambdas - trial_lambdas) ** 2)) # r^2 value
               
        return r_val

    def run_MCMC(self):
        # For an eos, gets distribution of "best fit" parameters
        
        # Randomly selected "start" parameters (fraction of increment is used to assure first sample is well within bounds)
        ratio = 1 # Used to be at .25
        g1_p1 = ((self.g1_p1 - (ratio * self.g1_p1_incr)) + ((2 * (ratio * self.g1_p1_incr)) * np.random.random()))
        g2_1 = ((self.g2_1 - (ratio * self.g2_1_incr)) + ((2 * (ratio * self.g2_1_incr)) * np.random.random()))
        g3_2 = ((self.g3_2 - (ratio * self.g3_2_incr)) + ((2 * (ratio * self.g3_2_incr)) * np.random.random()))
        g4_3 = ((self.g4_3 - (ratio * self.g4_3_incr)) + ((2 * (ratio * self.g4_3_incr)) * np.random.random()))

        eos_pointer = lalsim.SimNeutronStarEOSByName(self.eos)
        fam_pointer = lalsim.CreateSimNeutronStarFamily(eos_pointer)
        min_mass = lalsim.SimNeutronStarFamMinimumMass(fam_pointer)/lal.MSUN_SI

        s, _, _, max_mass = self.modsel.getEoSInterp(eosname=self.eos, m_min=min_mass)
        target_masses = np.linspace(min_mass,max_mass,self.N)
        target_Lambdas = s(target_masses)
        self.target_lambdas = (target_Lambdas / lal.G_SI) * ((target_masses * lal.MRSUN_SI) ** 5)
        
        value_error_samples = list(np.loadtxt(self.value_error_samples_file))
        runtime_error_samples = list(np.loadtxt(self.runtime_error_samples_file))
        with open(self.seg_fault_samples_file, "r") as f: seg_fault_samples = json.load(f)
        seg_fault_samples.append([0.0,0.0,0.0,0.0]) # placeholder for seg fault sample (needs to be in try: or this indice's value gets weird)

        # METROPOLIS-HASTINGS  
        no_errors = False
        while no_errors == False:
            
            # 1st sample
            g1_p1_choice1 = ((self.g1_p1 - self.g1_p1_incr) + ((2 * self.g1_p1_incr) * np.random.random()))
            g2_1_choice1 = ((self.g2_1 - self.g2_1_incr) + ((2 * self.g2_1_incr) * np.random.random()))
            g3_2_choice1 = ((self.g3_2 - self.g3_2_incr) + ((2 * self.g3_2_incr) * np.random.random()))
            g4_3_choice1 = ((self.g4_3 - self.g4_3_incr) + ((2 * self.g4_3_incr) * np.random.random()))
            
            try: 

                print(seg_fault_samples)
                seg_fault_samples[-1] = [g1_p1_choice1,g2_1_choice1,g3_2_choice1,g4_3_choice1]
                with open(self.seg_fault_samples_file, "w") as f: json.dump(seg_fault_samples, f, indent=2)
                L1 = self.likelihood(g1_p1_choice1,g2_1_choice1,g3_2_choice1,g4_3_choice1)
                no_errors = True # if L1 doesn't give an error, the while loop will end

            # Can run into ValueError from the use of lal's piecewise function (I think)
            # APPENDS ERROR-PRONE SAMPLES AND SAVES THEM
            except ValueError: 
                value_error_samples.append([g1_p1_choice1,g2_1_choice1,g3_2_choice1,g4_3_choice1])
                np.savetxt(self.value_error_samples_file, value_error_samples)
                continue 
            except RuntimeError: 
                runtime_error_samples.append([g1_p1_choice1,g2_1_choice1,g3_2_choice1,g4_3_choice1])
                np.savetxt(self.runtime_error_samples_file, runtime_error_samples)
                continue
            except TypeError: # Error specific to spectral method
                type_error_samples.append([g1_p1_choice1,g2_1_choice1,g3_2_choice1,g4_3_choice1])
                np.savetxt(self.type_error_samples_file, type_error_samples)
                continue

        post_p1 = []
        post_g1 = []
        post_g2 = []
        post_g3 = []
        post_r2 = []
        
        increment = 1
        while increment <= (self.samples):

            print("Attempt" + str(increment))
            # New sample
            g1_p1_choice2 = ((self.g1_p1 - self.g1_p1_incr) + ((2 * self.g1_p1_incr) * np.random.random()))
            g2_1_choice2 = ((self.g2_1 - self.g2_1_incr) + ((2 * self.g2_1_incr) * np.random.random()))
            g3_2_choice2 = ((self.g3_2 - self.g3_2_incr) + ((2 * self.g3_2_incr) * np.random.random()))
            g4_3_choice2 = ((self.g4_3 - self.g4_3_incr) + ((2 * self.g4_3_incr) * np.random.random()))

            try: 

                print(seg_fault_samples)
                seg_fault_samples[-1] = [g1_p1_choice2,g2_1_choice2,g3_2_choice2,g4_3_choice2]
                with open(self.seg_fault_samples_file, "w") as f: json.dump(seg_fault_samples, f, indent=2)
                L2 = self.likelihood(g1_p1_choice2,g2_1_choice2,g3_2_choice2,g4_3_choice2) # if L2 gives an error it'll keep trying

            # APPENDS ERROR-PRONE SAMPLES AND SAVES THEM
            except ValueError: 
                value_error_samples.append([g1_p1_choice2,g2_1_choice2,g3_2_choice2,g4_3_choice2])
                np.savetxt(self.value_error_samples_file, value_error_samples)
                continue
            except RuntimeError: 
                runtime_error_samples.append([g1_p1_choice2,g2_1_choice2,g3_2_choice2,g4_3_choice2])
                np.savetxt(self.runtime_error_samples_file, runtime_error_samples)
                continue

            if L2/L1 >= np.random.random():

                g1_p1_choice1 = g1_p1_choice2
                g2_1_choice1 = g2_1_choice2
                g3_2_choice1 = g3_2_choice2
                g4_3_choice1 = g4_3_choice2


                #post_r2.append(L2) # if choice2s are better, append their likelihood
                
            #else:

                #post_r2.append(L1) # otherwise choice1s are better, so their likelihood is appended instead
            increment += 1
            # current eos' p1,g1,g2,g3 combination is stored (can then see what parameter combinations lasts the "longest")
            #post_g1_p1.append(g1_p1_choice1)
            #post_g2_1.append(g2_1_choice1)
            #post_g3_2.append(g3_2_choice1)
            #post_g4_3.append(g4_3_choice1)

            # Saving script for samples and their likelihoods was deleted. Not needed

        #If it makes it past this while loop, it means no seg_faults happened
        seg_fault_samples.pop() # Need to clip seg_fault_samples list in case none were found.

