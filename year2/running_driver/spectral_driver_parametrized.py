from GWXtreme.parametrized_eos_sampler import mcmc_sampler
import numpy as np
import glob
import time

#Array Containing list of paths to the .dat files  containing the posterior samples for the events:

paths = np.array(glob.glob("files/galpop/APR4_EPP/*"))
files = np.repeat("/bns_example_samples.dat", len(paths))
fnames = list(np.char.add(paths,files))

outputDir = "runs/post_runtime_fix/spectral_galpop_1000_sample_run2/"

#Name of/ Path to file in which EoS parameter posterior samples will be saved:
outname='{}Ap4_O3_injections'.format(outputDir)

start = time.time()
#Initialize Sampler Object:

"""For SPectral"""

sampler=mcmc_sampler(fnames, {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}, outname, nwalkers=100, Nsamples=1000, ndim=4, spectral=True, npool=50)


"""OR"""

"""For Piece wise polytrope"""

#sampler=mcmc_sampler(fnames, {'logP':{'params':{"min":33.6,"max":34.5}},'gamma1':{'params':{"min":2.0,"max":4.5}},'gamma2':{'params':{"min":1.1,"max":4.5}},'gamma3':{'params':{"min":1.1,"max":4.5}}}, outname, nwalkers=100, Nsamples=1000, ndim=4, spectral=False, npool=50)

#Run, Save , Plot

sampler.initialize_walkers()
sampler.run_sampler()
sampler.save_data()

end = time.time()
hours = (end - start) / 3600
print("Hours: {}".format(hours))

fig=sampler.plot(cornerplot={'plot':True,'true vals':None},p_vs_rho={'plot':True,'true_eos':'AP4'})
fig['corner'].savefig('{}corner4_O3.png'.format(outputDir))
fig['p_vs_rho'][0].savefig('{}eos4_O3.png'.format(outputDir))

