# This file contains all functions/methods used in the GWXtreme code to
# analyze the accuracy of each method at obtaining bayes factor

def getMinMass(self, eosname):
    '''
    This method obtains the minimum mass tolerated by an
    equation of state.
    '''

    eos = lalsim.SimNeutronStarEOSByName(eosname)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    min_mass = lalsim.SimNeutronStarFamMinimumMass(fam)/lal.MSUN_SI

    return min_mass

def getMR(self, eosname=None, m_min=1.0, N=100):
    '''
    Obtains the mass, radius, and kappa tolerated by
    an equation of state
    '''
    if eosname is None:
        print('Allowed equation of state models are:')
        print(lalsim.SimNeutronStarEOSNames)
        print('Pass the model name as a string')
        return None
    try:
      assert eosname in list(lalsim.SimNeutronStarEOSNames)
    except AssertionError:
      print('EoS family is not available in lalsimulation')
      print('Allowed EoS are :\n' + str(lalsim.SimNeutronStarEOSNames))
      print('Make sure that if you are passing a custom file, it exists')
      print('in the path that you have provided...')
      sys.exit(0)
       
    eos = lalsim.SimNeutronStarEOSByName(eosname)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
      
    # This is necessary so that interpolant is computed over the full range
    # Keeping number upto 3 decimal places
    # Not rounding up, since that will lead to RuntimeError
    max_mass = int(max_mass*1000)/1000
    masses = np.linspace(m_min, max_mass, N)
    masses = masses[masses <= max_mass]
    ms = []
    rs = []
    ks = []
    for m in masses:
      try:
        rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
        kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
        rs.append(rr)
        ks.append(kk)
        ms.append(m)
      except RuntimeError:
        break
  return [ms, rs, ks]
