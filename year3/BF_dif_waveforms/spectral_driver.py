from GWXtreme.parametrized_eos_sampler import mcmc_sampler
import numpy as np
import time

#Array Containing list of paths to the .dat files  containing the posterior samples for the events:

fnames = np.array(["/home/michael/projects/eos/GWXtreme_Tasks/year2/BF_dif_waveforms/Files/high_spin_PhenomPNRT_posterior_samples.json"])

outputDir = "outdir/sampling/"

#Name of/ Path to file in which EoS parameter posterior samples will be saved:
outname='{}GW170817'.format(outputDir)

start = time.time()
#Initialize Sampler Object:

"""For SPectral"""

sampler=mcmc_sampler(fnames, {'gamma1':{'params':{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{'params':{"min":-0.02,"max":0.02}}}, outname, nwalkers=100, Nsamples=1000, ndim=4, spectral=True, npool=100)

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

