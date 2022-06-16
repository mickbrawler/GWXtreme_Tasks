import lalsimulation as lalsim
import lal
import numpy as np
import argparse
from scipy.interpolate import interp1d

# Script that will test spectral or piecewise paremetric parameters for errors

def tester(g1_p1, g2_g1, g3_g2, g4_g3, core, spectral):

    Dir = "core{}/".format(core)
    np.savetxt("files/runs/{}placeholder.txt".format(Dir),[0]) # seg_fault
    try:
        if spectral == 1: eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g1_p1, g2_g1, g3_g2, g4_g3) # runtime_error can arise
        else: eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(g1_p1, g2_g1, g3_g2, g4_g3)
        fam = lalsim.CreateSimNeutronStarFamily(eos) # seg_fault can arise
        max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
        max_mass = int(max_mass*1000)/1000
        m_min = 1.0
        masses = np.linspace(m_min,max_mass,1000)
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
        s = interp1d(gravMass, Lambdas) # value_error can arise from not having suitable number of gravMass and Lambdas value, which is cause by line 31
        #min_mass = lalsim.SimNeutronStarFamMinimumMass(fam)/lal.MSUN_SI
        #trial_masses = np.linspace(min_mass,max_mass,1000)
        #trial_Lambdas = s(trial_masses) # value_error can arise from going beyond the interpolant's bounds if you using min_mass. Using m_min solves it
        np.savetxt("files/runs/{}placeholder.txt".format(Dir),[1]) # no_error
    except RuntimeError:
        np.savetxt("files/runs/{}placeholder.txt".format(Dir),[2]) # runtime_error
    except ValueError:
        np.savetxt("files/runs/{}placeholder.txt".format(Dir),[3]) # value_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("g1_p1", help="First Parameter", type=float)
    parser.add_argument("g2_g1", help="Second Parameter", type=float)
    parser.add_argument("g3_g2", help="Third Parameter", type=float)
    parser.add_argument("g4_g3", help="Fourth Parameter", type=float)
    parser.add_argument("core", help="Core Number", type=int)
    parser.add_argument("spectral", help="Use of spectral or piecewise", type=int)
    args = parser.parse_args()

    tester(args.g1_p1,args.g2_g1,args.g3_g2,args.g4_g3,args.core,args.spectral)
