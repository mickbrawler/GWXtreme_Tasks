from GWXtreme.parametrized_eos_sampler import mcmc_sampler
import numpy as np
import glob
import time

#Array Containing list of paths to the .dat files  containing the posterior samples for the events:

paths = np.array(glob.glob("files/snrbin/APR4_EPP/33_to_35/5_low_m/*"))
files = np.repeat("/bns_example_samples.dat", len(paths))
fnames = list(np.char.add(paths,files))

outputDir = "runs/official/piecewise_snr_33_35_5low_10000/"

#Name of/ Path to file in which EoS parameter posterior samples will be saved:
outname='{}Ap4_O3_injections'.format(outputDir)

start = time.time()
#Initialize Sampler Object:

"""For Piece wise polytrope"""

sampler=mcmc_sampler(fnames, {'logP':{'params':{"min":33.6-1,"max":34.5-1}},'gamma1':{'params':{"min":2.0,"max":4.5}},'gamma2':{'params':{"min":1.1,"max":4.5}},'gamma3':{'params':{"min":1.1,"max":4.5}}}, outname, nwalkers=100, Nsamples=10000, ndim=4, spectral=False, npool=20)

#Run, Save , Plot

#sampler.p0=np.loadtxt('files/initialize_walkers/valid_art_pop_piecewise2_2.txt')
sampler.initialize_walkers()
sampler.run_sampler()
sampler.save_data()

end = time.time()
hours = (end - start) / 3600
print("Hours: {}".format(hours))

fig=sampler.plot(cornerplot={'plot':True,'true vals':None},p_vs_rho={'plot':True,'true_eos':'AP4'})
fig['corner'].savefig('{}corner4_O3.png'.format(outputDir))
fig['p_vs_rho'][0].savefig('{}eos4_O3.png'.format(outputDir))

