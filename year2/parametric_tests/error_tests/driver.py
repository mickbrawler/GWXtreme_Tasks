import lalsimulation as lalsim
import numpy as np
import os
import argparse

# Driver designed to test SPECTRAL samples for errors

class error_search:

    def __init__(self, samples, core):
        self.samples = samples
        self.core = core
        self.Dir = "core{}/".format(core)

    def get_random_samples(self):
        # Creates samples with predetermined bounds

        self.gamma1 = np.random.uniform(low=0.0,high=2.5,size=self.samples)
        self.gamma2 = np.random.uniform(low=-2.0,high=2.0,size=self.samples)
        self.gamma3 = np.random.uniform(low=-1.0,high=1.0,size=self.samples)
        self.gamma4 = np.random.uniform(low=-0.1,high=0.1,size=self.samples)

    def runner(self):
        # Runs the error_test on each sample
        seg_faults = []
        no_errors = []
        runtime_errors = []
        value_errors = []

        for g1_p1, g2_g1, g3_g2, g4_g3 in zip(self.gamma1, self.gamma2, self.gamma3, self.gamma4):
            os.system("python3 error_test.py {} {} {} {} {}".format(g1_p1, g2_g1, g3_g2, g4_g3, self.core))
            x = int(np.loadtxt("files/runs/{}placeholder.txt".format(self.Dir)))

            sample = [g1_p1, g2_g1, g3_g2, g4_g3]
            if x==0: seg_faults.append(sample)
            if x==1: no_errors.append(sample)
            if x==2: runtime_errors.append(sample)
            if x==3: value_errors.append(sample)

        np.savetxt("files/runs/{}seg_faults.txt".format(self.Dir), seg_faults) 
        np.savetxt("files/runs/{}no_errors.txt".format(self.Dir), no_errors) 
        np.savetxt("files/runs/{}runtime_errors.txt".format(self.Dir), runtime_errors)
        np.savetxt("files/runs/{}value_errors.txt".format(self.Dir), value_errors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", help="Number of samples", type=int)
    parser.add_argument("core", help="Core number", type=int)
    args = parser.parse_args()

    tester = error_search(args.samples,args.core)
    tester.get_random_samples()
    tester.runner()
