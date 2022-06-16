from GWXtreme import eos_model_selection as ems
from GWXtreme import eos_prior as ep
import lal
import numpy as np

def likelihood(g1_p1, g2_g1, g3_g2, g4_g3, priorbounds, keys, modsel, spectral):
    # Produces r2 value between lal and parametrized lambdas
    
    parameters = [g1_p1, g2_g1, g3_g2, g4_g3]
    params = {k:np.array([par]) for k,par in zip(keys,parameters)}

    try:
        if not ep.is_valid_eos(params, priorbounds, spectral=spectral):
            return -np.inf
        s, _, max_mass = modsel.getEoSInterp_parametrized(parameters)
        trial_masses = np.linspace(1.0,max_mass,1000)
        trial_Lambdas = s(trial_masses)
        trial_lambdas = (trial_Lambdas / lal.G_SI) * ((trial_masses * lal.MRSUN_SI) ** 5) 
        #r_val = 1 / np.log(np.sum((self.target_lambdas - trial_lambdas) ** 2)) unecessary for this test
        return r_val
    except:
        return -np.inf
