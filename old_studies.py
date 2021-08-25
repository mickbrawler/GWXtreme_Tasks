# This file contains all functions/methods used in the GWXtreme code to
# analyze the accuracy of each method at obtaining bayes factor

  11     def getMinMass(self, eosname):
  10         '''
   9         This method obtains the minimum mass tolerated by an
   8         equation of state.
   7         '''
   6
   5         eos = lalsim.SimNeutronStarEOSByName(eosname)
   4         fam = lalsim.CreateSimNeutronStarFamily(eos)
   3         min_mass = lalsim.SimNeutronStarFamMinimumMass(fam)/lal.MSUN_SI
   2
   1         return min_mass

def getMR(self, eosname=None, m_min=1.0, N=100):
   1         '''
   2         Obtains the mass, radius, and kappa tolerated by
   3         an equation of state
   4         '''
   5
   6         if eosname is None:
   7             print('Allowed equation of state models are:')
   8             print(lalsim.SimNeutronStarEOSNames)
   9             print('Pass the model name as a string')
  10             return None
  11         try:
  12             assert eosname in list(lalsim.SimNeutronStarEOSNames)
  13         except AssertionError:
  14             print('EoS family is not available in lalsimulation')
  15             print('Allowed EoS are :\n' + str(lalsim.SimNeutronStarEOSNames))
  16             print('Make sure that if you are passing a custom file, it exists')
  17             print('in the path that you have provided...')
  18             sys.exit(0)
  19
  20         eos = lalsim.SimNeutronStarEOSByName(eosname)
  21         fam = lalsim.CreateSimNeutronStarFamily(eos)
  22         max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
  23
  24         # This is necessary so that interpolant is computed over the full range
  25         # Keeping number upto 3 decimal places
  26         # Not rounding up, since that will lead to RuntimeError
  27         max_mass = int(max_mass*1000)/1000
  28         masses = np.linspace(m_min, max_mass, N)
  29         masses = masses[masses <= max_mass]
  30         ms = []
  31         rs = []
  32         ks = []
  33         for m in masses:
  34             try:
  35                 rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
  36                 kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
  37                 rs.append(rr)
  38                 ks.append(kk)
  39                 ms.append(m)
  40             except RuntimeError:
  41                 break
  42
  43         return [ms, rs, ks]
