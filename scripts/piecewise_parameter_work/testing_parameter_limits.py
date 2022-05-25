from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import json

# This is reusing MCMC_eos_comparison.py with some alterations. We're going to 
# be using the same presets [p1,g1,g2,g3] for all eos, and going to vary the 
# increments when running things on ipython. It is important not just to vary 
# one parameter's increment in runs, but to try out new increments together.

# Default parameter starting points: [33.4305,3.143,2.6315,2.7315]
# Combination 0: increments = [.4575,.927,1.1595,.9285]
# Combination 1: increments = [.5,1.3,2.15,2.0]
# Combination 2: increments = [.70,1.350,2.75,2.65]

# New increment tried in the past: (Parenthesis show what increments were used together for a run)
# 1) p[(.5),.75,.80] g1[1.0,1.25,(1.3),1.4] g2[1.25,1.5,1.75,2.0,(2.15)] g3[1.0,1.25,1.5.1.75,(2.0)]

# (In the second increment test we went to far and the increments used together resulted in errors)
# 2) p[.55,.60,.65,(.70)] g1[1.325,(1.350),1.360] g2[2.25,2.35,2.45,2.55,2.65,(2.75)] g3[2.15,2.25,2.35,2.45,2.55,(2.65)]

class param_distro:

    def __init__(self, N, transitions):
        # Hold attributes that used to be global variables.
        # N             : Parameter size.
        # transitions   : Number of parameter combinations tested.

        # List of some of the eos GWXtreme can work with off the cuff
        self.GWX_list = ["BHF_BBB2","KDE0V","KDE0V1","SKOP","HQC18","SLY2",
                         "SLY230A","SKMP","RS","SK255","SLY9","APR4_EPP",
                         "SKI2","SKI4","SKI6","SK272","SKI3","SKI5","MPA1",
                         "MS1B_PP","MS1_PP","BBB2","AP4","MPA1","MS1B","MS1",
                         "SLY"]

        # List of compatible eos paper found p1,g1,g2,g3 values for
        self.pap_list = ["AP4","BBB2","MPA1","MS1","SLY"]

        self.modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")

        self.N = N
        self.transitions = transitions

    def eos_to_run(self, eos_list, runs, directory, run0, p1_incr=.4575, 
                   g1_incr=.927, g2_incr=1.1595, g3_incr=.9285):
        # eos_list  : List of named equations of state.
        # runs      : Number of different starting chains.
        # directory : Directory for distributions.
        # runs0     : Label for repeat runs to prevent overwriting.

        for eos in eos_list:
            
            for run in range(run0,runs+run0):

                print(eos)
                outputfile = "{}{}_{}.json".format(directory,eos,run)
                self.run_MCMC(eos,outputfile,p1_incr,g1_incr,g2_incr,g3_incr)

    def run_MCMC(self, eos, outputfile, p1_incr, g1_incr, g2_incr, g3_incr):
        # For an eos, gets distribution of "best fit" parameters
        
        log_p1_SI,g1,g2,g3 = 33.4305,3.143,2.6315,2.7315 # defaults

        # Randomly selected "start" parameters
        log_p1_SI = ((log_p1_SI - (.25 * p1_incr)) + ((2 * (.25 * p1_incr)) * np.random.random()))
        g1 = ((g1 - (.25 * g1_incr)) + ((2 * (.25 * g1_incr)) * np.random.random()))
        g2 = ((g2 - (.25 * g2_incr)) + ((2 * (.25 * g2_incr)) * np.random.random()))
        g3 = ((g3 - (.25 * g3_incr)) + ((2 * (.25 * g3_incr)) * np.random.random()))

        eos_pointer = lalsim.SimNeutronStarEOSByName(eos)
        fam_pointer = lalsim.CreateSimNeutronStarFamily(eos_pointer)
        min_mass = lalsim.SimNeutronStarFamMinimumMass(fam_pointer)/lal.MSUN_SI

        s, _, _, max_mass = self.modsel.getEoSInterp(eosname=eos, m_min=min_mass)
        target_masses = np.linspace(min_mass,max_mass,self.N)
        target_Lambdas = s(target_masses)
        self.target_lambdas = (target_Lambdas / lal.G_SI) * ((target_masses * lal.MRSUN_SI) ** 5)

        # METROPOLIS-HASTINGS  
        no_errors = False
        while no_errors == False:

            p1_choice1 = ((log_p1_SI - p1_incr) + ((2 * p1_incr) * np.random.random()))
            g1_choice1 = ((g1 - g1_incr) + ((2 * g1_incr) * np.random.random()))
            g2_choice1 = ((g2 - g2_incr) + ((2 * g2_incr) * np.random.random()))
            g3_choice1 = ((g3 - g3_incr) + ((2 * g3_incr) * np.random.random()))

            try: 

                print([p1_choice1,g1_choice1,g2_choice1,g3_choice1])
                L1 = self.likelihood(p1_choice1,g1_choice1,g2_choice1,g3_choice1)
                no_errors = True # if L1 doesn't give an error, the while loop will end

            # Can run into ValueError from the use of lal's piecewise function (I think)
            except ValueError: continue
            except RuntimeError: continue

        post_p1 = []
        post_g1 = []
        post_g2 = []
        post_g3 = []
        post_r2 = []
        
        while len(post_p1) <= (self.transitions-1):

            p1_choice2 = ((log_p1_SI - p1_incr) + ((2 * p1_incr) * np.random.random()))
            g1_choice2 = ((g1 - g1_incr) + ((2 * g1_incr) * np.random.random()))
            g2_choice2 = ((g2 - g2_incr) + ((2 * g2_incr) * np.random.random()))
            g3_choice2 = ((g3 - g3_incr) + ((2 * g3_incr) * np.random.random()))

            try: 

                print([p1_choice2,g1_choice2,g2_choice2,g3_choice2])
                L2 = self.likelihood(p1_choice2,g1_choice2,g2_choice2,g3_choice2) # if L2 gives an error it'll keep trying

            except ValueError: continue
            except RuntimeError: continue

            if L2/L1 >= np.random.random():

                p1_choice1 = p1_choice2
                g1_choice1 = g1_choice2
                g2_choice1 = g2_choice2
                g3_choice1 = g3_choice2

                post_r2.append(L2) # if choice2s are better, append their likelihood
                
            else:

                post_r2.append(L1) # otherwise choice1s are better, so their likelihood is appended instead
            
            # current eos' p1,g1,g2,g3 combination is stored (can then see what parameter combinations lasts the "longest")
            post_p1.append(p1_choice1)
            post_g1.append(g1_choice1)
            post_g2.append(g2_choice1)
            post_g3.append(g3_choice1)
       
        data = {"p1" : post_p1, "g1" : post_g1, "g2" : post_g2, "g3" : post_g3, "r2" : post_r2}
        with open(outputfile, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        
    def likelihood(self, log_p1_SI, g1, g2, g3):
        # Produces r2 value between lal and piecewise lambdas
        # log_p1_SI : Pressure.
        # g1        : Adiabatic Index 1.
        # g2        : Adiabatic Index 2.
        # g3        : Adiabatic Index 3.

        s, min_mass, max_mass = self.modsel.getEoSInterp_parametrized([log_p1_SI,g1,g2,g3])
        trial_masses = np.linspace(min_mass,max_mass,self.N)
        trial_Lambdas = s(trial_masses)
        trial_lambdas = (trial_Lambdas / lal.G_SI) * ((trial_masses * lal.MRSUN_SI) ** 5) # lambdas the eos produced via getEosInterpFrom_piecewise.
        r_val = 1 / np.log(np.sum((self.target_lambdas - trial_lambdas) ** 2)) # r^2 value
               
        return(r_val)
