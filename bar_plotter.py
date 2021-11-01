from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import pylab as pl
import glob
import json

# Need list of eos names used
eos_list = []

# Need bayes factors from lal source in list
lal_bf = []

# Need bayes factors from piecewise polytrope values in list
pp_bf = []

# Need standard deviation for each bayes factor in a list
error = [[],[],[]]

# Build Plot
