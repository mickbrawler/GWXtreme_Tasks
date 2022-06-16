import lalsimulation as lalsim
import numpy as np
import os
import argparse

# Driver designed to test SPECTRAL or PIECEWISE samples for errors

class error_search:

    def __init__(self, samples, core, spectral):
        self.samples = samples
        self.core = core
        self.Dir = "core{}/".format(core)
        self.spectral = spectral

        if self.spectral == 1:
            #self.priorbounds = {'gamma1':{'params':{"min":0.2,"max":2.00}},
            #                    'gamma2':{'params':{"min":-1.6,"max":1.7}},
            #                    'gamma3':{'params':{"min":-0.6,"max":0.6}},
            #                    'gamma4':{'params':{"min":-0.02,"max":0.02}}}
            self.priorbounds = {'gamma1':{'params':{"min":0.0,"max":2.5}},
                                'gamma2':{'params':{"min":-2.0,"max":2.0}},
                                'gamma3':{'params':{"min":-1.0,"max":1.0}},
                                'gamma4':{'params':{"min":-0.1,"max":0.1}}}
            self.keys = ["gamma1","gamma2","gamma3","gamma4"]
        else:
            #self.priorbounds = {'logP':{'params':{"min":33.6,"max":34.5}},
            #                    'gamma1':{'params':{"min":2.0,"max":4.5}},
            #                    'gamma2':{'params':{"min":1.1,"max":4.5}},
            #                    'gamma3':{'params':{"min":1.1,"max":4.5}}}
            self.priorbounds = {'logP':{'params':{"min":32.5,"max":34.5}},
                                'gamma1':{'params':{"min":1.0,"max":5.0}},
                                'gamma2':{'params':{"min":1.0,"max":5.0}},
                                'gamma3':{'params':{"min":1.0,"max":5.0}}}
            self.keys = ["logP","gamma1","gamma2","gamma3"]

        self.low_g1_p1 = self.priorbounds[self.keys[0]]['params']['min']
        self.high_g1_p1 = self.priorbounds[self.keys[0]]['params']['max']
        self.low_g2_g1 = self.priorbounds[self.keys[1]]['params']['min']
        self.high_g2_g1 = self.priorbounds[self.keys[1]]['params']['max']
        self.low_g3_g2 = self.priorbounds[self.keys[2]]['params']['min']
        self.high_g3_g2 = self.priorbounds[self.keys[2]]['params']['max']
        self.low_g4_g3 = self.priorbounds[self.keys[3]]['params']['min']
        self.high_g4_g3 = self.priorbounds[self.keys[3]]['params']['max']

    def get_random_samples(self):
        # Creates samples with predetermined bounds

        self.g1_p1_array = np.random.uniform(low=self.low_g1_p1,high=self.high_g1_p1,size=self.samples)
        self.g2_g1_array = np.random.uniform(low=self.low_g2_g1,high=self.high_g2_g1,size=self.samples)
        self.g3_g2_array = np.random.uniform(low=self.low_g3_g2,high=self.high_g3_g2,size=self.samples)
        self.g4_g3_array = np.random.uniform(low=self.low_g4_g3,high=self.high_g4_g3,size=self.samples)

    def runner(self):
        # Runs the error_test on each sample
        seg_faults = []
        no_errors = []
        runtime_errors = []
        value_errors = []

        for g1_p1, g2_g1, g3_g2, g4_g3 in zip(self.g1_p1_array, self.g2_g1_array, self.g3_g2_array, self.g4_g3_array):
            os.system("python3 error_test.py {} {} {} {} {} {}".format(g1_p1, g2_g1, g3_g2, g4_g3, self.core, self.spectral))
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
    parser.add_argument("spectral", help="Use of spectral(1) or piecewise(0)", type=int)
    args = parser.parse_args()

    tester = error_search(args.samples,args.core,args.spectral)
    tester.get_random_samples()
    tester.runner()
