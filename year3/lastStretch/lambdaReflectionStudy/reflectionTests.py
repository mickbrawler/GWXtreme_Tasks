import json
import matplotlib.pyplot as plt
import numpy as np
import scipy
from GWXtreme import eos_model_selection as ems

# There's three ways to test if lambda_1's (really any lambda) kde can and should be reflected

# One is simply using the resample logic on GWXtreme
# Second is doing similar but including the logic in the bounded_3d_kde code that accounts for boundaries
# Third is to compare the hist of lambda_1 to the pdf of the kde of the lambda_1 distribution

# LOAD Data!!!
filename = "../files/NSBH/gw230529_phenom_lowSpin.json"
with open(filename,"r") as f: data = json.load(f)['posterior']['content']
lambda_1 = data['lambda_1']
lambda_2 = data['lambda_2']
m1_source = data['m1_source']
m2_source = data['m2_source']
mc_source = data['mc_source']
q = data['q']

# Method 1/2:
modsel = ems.Model_selection(filename,kdedim=3,logq=True)
margPostData = modsel.margPostData
new_margPostData = modsel.kde.resample(size=len(margPostData)).T
resampledLambda1 = new_margPostData[:,0]

plt.clf()
#plt.hist(lambda_1,density=True,color='red',alpha=0.25,label="lambda_1")
#plt.hist(resampledLambda1,density=True,color='blue',alpha=0.25,label="resampled")
plt.hist(lambda_1,density=True,color='red',label="lambda_1")
plt.hist(resampledLambda1,density=True,color='blue',label="resampled")
plt.legend()
plt.xlabel("Lambda 1")
plt.savefig("method12.png")
# I think method 1 is already method 2 if kdedim=3. The only way to do method 1
# as I thought would be to remove the boundary logic... I see why Anarya was surprised.
# I had misunderstood that resample was part of an external functionality.
