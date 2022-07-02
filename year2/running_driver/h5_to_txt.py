import numpy as np
import h5py

# script meant to grab last n samples from h5 file holding sample 
# distribution from emcee run and convert to formatted txt file

hf = h5py.File('runs/post_runtime_fix/piecewise_galpop_1000_sample_run3/Ap4_O3_injections.h5', 'r')
chains = np.array(hf.get('chains'))
end_samples = chains[-1]
np.savetxt("test.txt", end_samples)
