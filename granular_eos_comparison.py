from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import os
import json

# First there is a parameter space (log_p0_SI,g1,g2,g3)

# In our case the R-Squared (RS) value of LAL and PP lambdas will 
# indicate if a new state is better

# If new state has a better RS value, adopt those parameters; else, keep 
# current parameters and try again

# After multiple transitions, we should localize to a certain area of 
# the parameter space, and start using more parameters in that area

# Eventually, after a designated number of attempted transitions prove 
# fruitless, the current parameters get deemed the best or closest

###############################################################################

# Lets attempt obtaining the best parameters for a single EoS with a 
# small resolution

def get_eos_parameters(transitions, N):

    modsel = ems.Model_selection(posteriorFile="posterior_samples/posterior_samples_narrow_spin_prior.dat")

    eos_list = ["AP4"]
    log_p0_SI = 33.269
    g1 = 2.830
    g2 = 3.445
    g3 = 3.348

    diff = .5
    increment = .005
    p_range = np.arange(log_p0_SI-diff,log_p0_SI+diff,increment)
    g1_range = np.arange(g1-diff,g1+diff,increment)
    g2_range = np.arange(g2-diff,g2+diff,increment)
    g3_range = np.arange(g3-diff,g3+diff,increment)

    loop_tracker = 0

    for eos in eos_list:

        eos_pointer = lalsim.SimNeutronStarEOSByName(eos)
        fam_pointer = lalsim.CreateSimNeutronStarFamily(eos_pointer)
        min_mass = lalsim.SimNeutronStarFamMinimumMass(fam_pointer)/lal.MSUN_SI

        s, _, Lambdas, max_mass = modsel.getEoSInterp(eosname=eos, m_min=min_mass)
        target_masses = np.linspace(min_mass,max_mass,N)
        target_Lambdas = s(target_masses)
        target_lambdas = (Lambdas / lal.G_SI) * ((target_masses * lal.MRSUN_SI) ** 5)

        R_val = 1000 # at the start we'll have to pick up the parameter we have

        for transition in range(transitions):

            p_choice, g1_choice, g2_choice, g3_choice = (np.random.choice(p_range), np.random.choice(g1_range)
                                                         ,np.random.choice(g2_range), np.random.choice(g3_range))

            try:

                s, min_mass, max_mass = modsel.getEoSInterpFrom_piecewise(p_choice,g1_choice,g2_choice,g3_choice)
                trial_masses = np.linspace(min_mass,max_mass,N)
                trial_Lambdas = s(trial_masses)
                trial_lambdas = (Lambdas / lal.G_SI) * ((trial_masses * lal.MRSUN_SI) ** 5)

                r_val = np.sum((target_lambdas - trial_lambdas) ** 2) # Check if this is the right way. Might be square then sum

            except RuntimeError:

                print("Runtime!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                continue

            except ValueError:

                print("ValueError!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                continue

            loop_tracker += 1

            print([p_choice,g1_choice,g2_choice,g3_choice],r_val)

            if r_val < R_val: # Since the smaller the R-Squared Value, the better the parameters

                R_val = r_val
                p_combo = [p_choice,g1_choice,g2_choice,g3_choice]
                lowest_r_val = r_val

    return(p_combo, lowest_r_val, loop_tracker)
