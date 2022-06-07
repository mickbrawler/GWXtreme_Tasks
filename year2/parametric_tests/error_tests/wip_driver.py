import lalsimulation as lalsim
import numpy as np
import testing_samples
import os

def runner():
    logP1, gamma1, gamma2, gamma3 = np.loadtxt("files/samples.txt").T
    no_errors = []
    errors = []
    seg_faults = []
    for g1_p1, g2_g1, g3_g2, g4_g3 in zip(logP1, gamma1, gamma2, gamma3):
        os.system("python testing_samples.py {} {} {} {}".format(g1_p1, g2_g1, g3_g2, g4_g3))
        x = int(np.loadtxt("files/placeholder.txt"))

        if x==0:
            seg_faults.append([g1_p1, g2_g1, g3_g2, g4_g3])
        if x==1:
            no_errors.append([g1_p1, g2_g1, g3_g2, g4_g3])
        if x==2:
            errors.append([g1_p1, g2_g1, g3_g2, g4_g3])

        np.savetxt("files/seg_faults.txt", np.array(seg_faults)) 
        np.savetxt("files/no_errors.txt", np.array(no_errors)) 
        np.savetxt("files/errors.txt", np.array(errors))

