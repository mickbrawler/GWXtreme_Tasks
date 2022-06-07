import lalsimulation as lalsim

# Script that will test spectral or piecewise paremetric parameters for errors
# Just spectral for now

def tester(g1_p1, g2_g1, g3_g2, g4_g3):
    try:
        eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g1_p1, g2_g1, g3_g2, g4_g3)
        fam = lalsim.CreateSimNeutronStarFamily(eos) 
        return 0
    except:
        return 1

def runner():
    for g1_p1, g2_g1, g3_g2, g4_g3 in zip(logP1, gamma1, gamma2, gamma3):
        x = driver(g1_p1, g2_g1, g3_g2, g4_g3)
            if x==0:
                save(g1_p1, g2_g1, g3_g2, g4_g3) # no_errors.txt
            if x==1:
                save(g1_p1, g2_g1, g3_g2, g4_g3) # value_runtime_errors.txt
            # if samples seg_faults this script will crash

