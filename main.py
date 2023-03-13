from parameters   import parameters
from hamiltonian  import hamiltonian
from numpy        import *

import time, warnings, multiprocessing

warnings.simplefilter('ignore')

if __name__=='__main__':
  time_b = time.time()
  p = parameters()
  h = hamiltonian(p)
  h.calc_band(p,"band")
  if p.calc_pdos:
    h.r_to_k(p)
    h.diagonalization(p)
    h.calc_pdos(p,"pdos")

  if p.band_dehybridize :
    h.dehybridization(p)
    h.calc_band(p,"band_dehybridized")
    if p.calc_pdos:
      h.r_to_k(p)
      h.diagonalization(p)
      h.calc_pdos(p,"pdos_dehybridized")

  with open(p.std_file, "a") as f:
    f.write("TIME(ALL) : {0}\n".format(time.time()-time_b))
