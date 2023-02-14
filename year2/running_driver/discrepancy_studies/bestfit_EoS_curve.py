import pylab as pl
import numpy as np
import lalsimulation as lalsim
import lal

# Script may not be necessary

# APR4_EPP bestfits (is_valid tested; though had to adjust priorbounds from before with -1 to logP
#[33.285954584684475, 3.0633821935068726, 3.285839760407425, 3.031384683724908]
#[0.6483014736029169, 0.22549530718867078, -0.020071115984931484, -0.0003498568113544248]

EoSs = {'name': "APR4_EPP",
        'spectral': [0.6483014736029169, 0.22549530718867078, -0.020071115984931484, -0.0003498568113544248],
        'piecewise': [33.285954584684475, 3.0633821935068726, 3.285839760407425, 3.031384683724908]}

for EoS in EoSs:
    if type(EoS) =:
        eos = lalsim.SimNeutronStarEOSByName(EoS)

fam = lalsim.CreateSimNeutronStarFamily(eos)
max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
max_mass = int(max_mass*1000)/1000
min_mass = lalsim.SimNeutronStarFamMinimumMass(fam)/lal.MSUN_SI
masses = np.linspace(min_mass, max_mass, N)
masses = masses[masses <= max_mass]

Lambdas = []
gravMass = []
radii = []
kappas = []
for m in masses:

    try:
        rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
        kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
        cc = m*lal.MRSUN_SI/rr
        Lambdas = np.append(Lambdas, (2/3)*kk/(cc**5))
        gravMass = np.append(gravMass, m)
        radii.append(rr)
        kappas.append(kk)
    except RuntimeError:
        break

Lambdas = np.array(Lambdas)
gravMass = np.array(gravMass)
radii = np.array(radii)
kappas = np.array(kappas)
