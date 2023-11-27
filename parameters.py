import os
from itertools import chain
import numpy as np

class parameters:
    def __init__(self):
        # self.example = "chain1d_spinless"
        # self.example = "chain1d"
        self.example = "fe_bcc"
        # self.example = "CrNb4Se8"
        # self.example = None
        self.set_example(self.example)

        if not self.example:
            self.dir = "hogehoge"
            self.filename = "hogehoge"
            self.prefix   = "hogehoge"
            self.nk1, self.nk2, self.nk3  = 1, 1, 1
            self.ispin = 1   # 1: spinless, 2: collinear, 3: non-colinear
            self.fermi = 0.0
            self.orbitals = [1,1]
            self.e_min = -10
            self.e_max = 10
            self.e_num = 11
            # some flags for the calculation
            self.orbital_types = [0,1]
            self.decompose = True
            self.dehybridize = True
            # [[[a1,a2...], [b1,b2...]],...]
            self.atoms_dehybridize = [[[0],[1]]]

        self.out_dir     = self.dir+"/out_"+self.filename

        # set parameters
        self.e_range = np.linspace(self.e_min,self.e_max,self.e_num)
        self.beta     = 500
        self.smearing = 0.03

        # other setups
        # self.core = int(os.environ.get('MKL_NUM_THREADS'))
        self.so_order = 2   # 1: a=(up1,...,upN,dn1,...,dnN), 2:a=(up1,dn2,...,upN,dnN), used only for ispin==3
        self.nkp   = self.nk1 * self.nk2 * self.nk3
        self.dk1, self.dk2, self.dk3 = 1./self.nk1, 1./self.nk2, 1./self.nk3
        k1, k2, k3 = np.meshgrid(np.arange(self.nk1)*self.dk1, np.arange(self.nk2)*self.dk2\
            , np.arange(self.nk3)*self.dk3, indexing="ij")
        self.k1, self.k2, self.k3 = k1.flatten(), k2.flatten(), k3.flatten()
        self.kpoints  = np.array([self.k1, self.k2, self.k3])
        self.natom    = len(self.orbitals)
        self.orbitals = np.array(self.orbitals)
        self.norb = self.orbitals.sum()
        self.n_orbital_type = len(np.unique(self.orbital_types))
        self.orbital_types = np.array(self.orbital_types)

        # from parameters file
        self.log_dir     = self.out_dir + "/log"
        self.fig_dir     = self.out_dir + "/figures"
        self.program_dir = self.out_dir + "/programs"
        if not os.path.isdir(self.out_dir): os.makedirs(self.out_dir)
        if not os.path.isdir(self.log_dir): os.makedirs(self.log_dir,exist_ok=True)
        if not os.path.isdir(self.fig_dir): os.makedirs(self.fig_dir,exist_ok=True)
        if not os.path.isdir(self.program_dir): os.makedirs(self.program_dir,exist_ok=True)
        os.system("cp *.py "+self.program_dir)

        assert sum(self.orbitals)==len(self.orbital_types), "types of every orbital should be specified"
        if self.atoms_dehybridize:
            assert (max(list(chain(*list(chain(*self.atoms_dehybridize))))) < self.natom) \
                , "index {} is out of range of {} atoms".format(max(list(chain(*list(chain(*self.atoms_dehybridize))))), self.natom)


    def set_example(self, example=None):
        if example=="chain1d_spinless":
            self.dir  = "examples/chain1d_spinless"
            self.filename = "chain1d_spinless"
            self.prefix   = "chain1d_spinless"
            self.nk1, self.nk2, self.nk3  = 256, 1, 1
            self.ispin = 1
            self.fermi = 0.0
            self.orbitals = [1 ,1]
            self.e_min = -3
            self.e_max = 3
            self.e_num = 501
            # some flags for the calculation
            self.orbital_types = [0, 1]
            self.decompose = True
            self.dehybridize = True
            # [[[a1,a2...], [b1,b2...]],...]
            self.atoms_dehybridize = [[[0],[1]]]
        if example=="chain1d":
            self.dir  = "examples/chain1d"
            self.filename = "chain1d"
            self.prefix   = "chain1d"
            self.nk1, self.nk2, self.nk3  = 256, 1, 1
            self.ispin = 2
            self.fermi = 0.0
            self.orbitals = [1, 1]
            self.e_min = -5
            self.e_max = 5
            self.e_num = 501
            # some flags for the calculation
            self.orbital_types = [0, 1]
            self.decompose = True
            self.dehybridize = True
            # [[[a1,a2...], [b1,b2...]],...]
            self.atoms_dehybridize = [[[0],[1]]]
        elif example=="fe_bcc":
            self.dir  = "examples/fe_bcc"
            self.filename = "fe_bcc"
            self.prefix   = "fe_bcc"
            self.nk1, self.nk2, self.nk3  = 16, 16, 16
            self.ispin = 2
            self.fermi = 18.2568
            self.orbitals = [9]
            self.e_min = -10
            self.e_max = 10
            self.e_num = 501
            self.orbital_types = [0]*6 + [1,2,3] # sp3d2,dxz,dyz,dxy
            self.decompose = True
            self.dehybridize = False
            # [[[a1,a2...], [b1,b2...]],...]
            self.atoms_dehybridize = None
        elif example=="CrNb4Se8":
            self.dir  = "examples/CrNb4Se8"
            self.filename = "CrNb4Se8"
            self.prefix   = "CrNb4Se8"
            self.nk1, self.nk2, self.nk3  = 8, 8, 8
            self.ispin = 2
            self.fermi = 0.
            # Cr(3d)*2, Nb(4d+5s)*8, Se(4p)*16
            self.orbitals = [5,5, 6,6,6,6, 6,6,6,6, 3,3,3,3, 3,3,3,3, 3,3,3,3, 3,3,3,3]
            self.e_min = -10
            self.e_max = 5
            self.e_num = 451
            self.orbital_types = [0]*5*2 + ([1]+[2]*5)*8 + [3]*3*16 # index of orbital type
            self.n_orbital_type = 4
            self.decompose = True
            self.dehybridize = True
            # [[[a1,a2...], [b1,b2...]],...]
            self.atoms_dehybridize = [[[0],[1]]]
        elif not example:
            pass

if __name__=="__main__":
    parameters()
