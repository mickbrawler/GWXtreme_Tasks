import numpy as np

def get_random_samples(N, outfile):
    # Creates txt file containing N number of samples with predetermined bounds
    # Just spectral for now

    g1 = np.random.uniform(low=0.0,high=2.5,size=N)
    g2 = np.random.uniform(low=-2.0,high=2.0,size=N)
    g3 = np.random.uniform(low=-0.1,high=0.1,size=N)
    g4 = np.random.uniform(low=-0.1,high=0.1,size=N)
    
    np.savetxt(outfile, np.array([g1,g2,g3,g4]).T)

