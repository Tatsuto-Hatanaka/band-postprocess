import numpy as np
import os

class parameters:
  def __init__(self):

    self.filename = "CrNb4Se8_iter100"
    self.prefix   = "CrNb4Se8_iter100"
    self.codes    = "VASP"

    # flags for calculations
    self.band_dehybridize = True
    self.band_decompose   = True
    self.calc_pdos       = True

    # tb model information
    self.ispin    = 1   # 1: spinless, 2: collinear, 3: non-colinear
    self.so_order = 2   # 1: a=(up1,...,upN,dn1,...,dnN), 2:a=(up1,dn2,...,upN,dnN), used only for ispin==3
    self.fermi = 18.2568   # fe_tb.dat (so:2)
    # self.fermi =

    self.orbitals          = [5,5, 6,6,6,6, 6,6,6,6, 3,3,3,3, 3,3,3,3, 3,3,3,3, 3,3,3,3]
    self.orbitals_dehybrid = [[[0,1],[2,3,4,5,6,7,8,9]]]
    self.n_orbital_type    = 4
    self.orbital_decompose = [0]*5*2 + ([1]+[2]*5)*8 + [3]*3*16
    self.orbital_each_pdos = [0]

    assert(sum(self.orbitals)==len(self.orbital_decompose))
    assert(self.n_orbital_type==len(np.unique(self.orbital_decompose)))

    # set parameters
    self.pdos_e_range  = np.linspace(-8,4,601)
    self.beta     = 500
    self.smearing = 0.05
    self.nk1, self.nk2, self.nk3  = 8, 8, 8

    # other setups
    self.core = int(os.environ.get('MKL_NUM_THREADS'))
    self.nk   = self.nk1 * self.nk2 * self.nk3
    self.dk1, self.dk2, self.dk3 = 1./self.nk1, 1./self.nk2, 1./self.nk3
    k1, k2, k3 = np.meshgrid(np.arange(self.nk1)*self.dk1, np.arange(self.nk2)*self.dk2, np.arange(self.nk3)*self.dk3, indexing="ij")
    self.k1, self.k2, self.k3 = k1.flatten(), k2.flatten(), k3.flatten()
    self.kpoints = np.array([self.k1,self.k2,self.k3])
    self.natom    = len(self.orbitals)
    self.orbitals = np.array(self.orbitals)
    self.norb = self.orbitals.sum()

    # from parameters file
    self.out_dir  = "out_"+self.filename
    if not os.path.isdir(self.out_dir): os.mkdir(self.out_dir)
    os.system("cp *py "+self.out_dir)
    self.std_file = self.out_dir+"/std.out"

    with open(self.std_file, "w") as f:
      f.write("Number of Parallelization: {0}\n".format(self.core))
