import lalsimulation as lalsim
import numpy as np
import argparse

# Script that will test spectral or piecewise paremetric parameters for errors
# Just spectral for now

def tester(g1_p1, g2_g1, g3_g2, g4_g3, core):

    Dir = "core{}/".format(core)
    np.savetxt("files/runs/{}placeholder.txt".format(Dir),[0])
    try:
        eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g1_p1, g2_g1, g3_g2, g4_g3)
        fam = lalsim.CreateSimNeutronStarFamily(eos) 
        np.savetxt("files/runs/{}placeholder.txt".format(Dir),[1])
    except:
        np.savetxt("files/runs/{}placeholder.txt".format(Dir),[2])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("g1_p1", help="First Parameter", type=float)
    parser.add_argument("g2_g1", help="Second Parameter", type=float)
    parser.add_argument("g3_g2", help="Third Parameter", type=float)
    parser.add_argument("g4_g3", help="Fourth Parameter", type=float)
    parser.add_argument("core", help="Core Number", type=int)
    args = parser.parse_args()

    tester(args.g1_p1,args.g2_g1,args.g3_g2,args.g4_g3,args.core)
