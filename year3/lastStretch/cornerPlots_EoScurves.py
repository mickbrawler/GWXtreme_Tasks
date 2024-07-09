import json
import corner
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from GWXtreme.eos_model_selection import get_Lambda_for_eos
from GWXtreme.bounded_3d_kde import Bounded_3d_kde
import lalsimulation as lalsim
import lal
from scipy.interpolate import interp1d

# Load event posterior
#event = "files/gw230529_phenom_lowSpin.json"
event = "files/GW170817Phenom.json"
with open(event,"r") as f:
    data = json.load(f)['posterior']['content']
(m1,m2,q,mc,Lambda1,Lambda2)=(np.array(data['m1_source']),
                            np.array(data['m2_source']),
                            np.array(data['q']),
                            np.array(data['mc_source']),
                            np.array(data['lambda_1']),
                            np.array(data['lambda_2']))

# Obtain EoS curve
EoS_Name = "APR4_EPP"
eos = lalsim.SimNeutronStarEOSByName(EoS_Name)
fam = lalsim.CreateSimNeutronStarFamily(eos)
min_mass = 0.8
m_min = 0.8
max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI

# This is necessary so that interpolant is computed over the full range
# Keeping number upto 3 decimal places
# Not rounding up, since that will lead to RuntimeError
max_mass = int(max_mass*1000)/1000
masses = np.linspace(m_min, max_mass, 1000)
masses = masses[masses <= max_mass]
Lambdas = []
gravMass = []
for m in masses:
    try:
        rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
        kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
        cc = m*lal.MRSUN_SI/rr
        Lambdas = np.append(Lambdas, (2/3)*kk/(cc**5))
        gravMass = np.append(gravMass, m)
    except RuntimeError:
        break
Lambdas = np.array(Lambdas)
gravMass = np.array(gravMass)
ss = interp1d(gravMass, Lambdas)

eosfunc = ss
M1, M2 = np.linspace(min(m1),max(m1),1000), np.linspace(min(m2),max(m2),1000)
Q = M2/M1
MC = ((M1*M2)**(3/5)) / ((M1+M2)**(1./5.))
Lambda1,Lambda2 = get_Lambda_for_eos(M1, max_mass, eosfunc),get_Lambda_for_eos(M2, max_mass, eosfunc)


# Format curve values
EoS_values = np.array([Lambda1,Lambda2,M1,M2,MC,Q])

# Construct corner & and overlay appropriate curves
Data = az.from_dict(data)
figure = corner.corner(Data)
ndim = 6
axes = np.array(figure.axes).reshape((ndim, ndim))
for yi in range(ndim):
    for xi in range(yi):
        print([yi,xi])
        ax = axes[yi, xi]
        ax.plot(EoS_values[xi], EoS_values[yi], color="red")

#plt.savefig("cornerCurvesGW230529.png")
plt.savefig("plots/cornerCurvesGW170817_2.png")

