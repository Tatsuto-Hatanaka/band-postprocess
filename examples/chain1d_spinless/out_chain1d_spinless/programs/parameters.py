import os
import numpy as np

class parameters:
    def __init__(self):
        self.example = "chain1d_spinless"
        # self.example = "chain1d"
        # self.example = "fe_prim"
        # self.example = "CrNb4Se8"
        # self.example = None
        self.set_example(self.example)

        if not self.example:
            self.filename = "hogehoge"
            self.prefix   = "hogehoge"
            self.out_dir  = "hogehoge"
            self.nk1, self.nk2, self.nk3  = 1, 1, 1
            self.ispin = 2   # 1: spinless, 2: collinear, 3: non-colinear
            self.fermi = 0.0
            self.orbitals = [1]
            self.e_min = -10
            self.e_max = 10
            self.e_num = 101
            self.e_range = np.linspace(self.e_min, self.e_max, self.e_num)
            self.band_dehybridize = False
            self.band_decompose = False
            self.calc_pdos = False
            self.orbitals_dehybrid = None
            self.n_orbital_type    = 1
            self.orbital_decompose = [0] # index of orbital type

        # tb model information
        self.so_order = 2   # 1: a=(up1,...,upN,dn1,...,dnN), 2:a=(up1,dn2,...,upN,dnN), used only for ispin==3

        # assert(sum(self.orbitals)==len(self.orbital_decompose))
        # assert(self.n_orbital_type==len(np.unique(self.orbital_decompose)))

        # set parameters
        self.e_range = np.linspace(self.e_min,self.e_max,self.e_num)
        self.beta     = 500
        self.smearing = 0.05

        # other setups
        # self.core = int(os.environ.get('MKL_NUM_THREADS'))
        self.nkp   = self.nk1 * self.nk2 * self.nk3
        self.dk1, self.dk2, self.dk3 = 1./self.nk1, 1./self.nk2, 1./self.nk3
        k1, k2, k3 = np.meshgrid(np.arange(self.nk1)*self.dk1, np.arange(self.nk2)*self.dk2, np.arange(self.nk3)*self.dk3, indexing="ij")
        self.k1, self.k2, self.k3 = k1.flatten(), k2.flatten(), k3.flatten()
        self.kpoints = np.array([self.k1, self.k2, self.k3])
        self.natom    = len(self.orbitals)
        self.orbitals = np.array(self.orbitals)
        self.norb = self.orbitals.sum()

        # from parameters file
        self.log_dir     = self.out_dir + "/log"
        self.fig_dir     = self.out_dir + "/figures"
        self.program_dir = self.out_dir + "/programs"
        if not os.path.isdir(self.out_dir): os.makedirs(self.out_dir)
        if not os.path.isdir(self.log_dir): os.makedirs(self.log_dir,exist_ok=True)
        if not os.path.isdir(self.fig_dir): os.makedirs(self.fig_dir,exist_ok=True)
        if not os.path.isdir(self.program_dir): os.makedirs(self.program_dir,exist_ok=True)
        os.system("cp *.py "+self.program_dir)
        if os.path.isfile(self.log_dir+"/log_hamiltonian.dat"): os.system("rm "+self.log_dir+"/*")


    def set_example(self, example=None):
        if example=="chain1d_spinless":
            self.filename = "chain1d_spinless"
            self.prefix   = "examples/chain1d_spinless/chain1d_spinless"
            self.out_dir  = "examples/chain1d_spinless/out_"+self.filename
            self.nk1, self.nk2, self.nk3  = 256, 1, 1
            self.ispin = 1
            self.fermi = 0.0
            self.orbitals = [1,1]
            self.e_min = -3
            self.e_max = 3
            self.e_num = 501
            # some flags for the calculation
            self.decompose = True
            self.orbital_types = [0,1]
            self.n_orbital_type = 2
            self.dehybridize = True
            # [[[a1,a2...], [b1,b2...]],...]
            self.atoms_dehybridize = [[[0],[1]]]
        if example=="chain1d":
            self.filename = "chain1d"
            self.prefix   = "examples/chain1d/chain1d"
            self.out_dir  = "examples/chain1d/out_"+self.filename
            self.nk1, self.nk2, self.nk3  = 256, 1, 1
            self.ispin = 2
            self.fermi = 0.0
            self.orbitals = [1,1]
            self.e_min = -3
            self.e_max = 3
            self.e_num = 501
            # some flags for the calculation
            self.decompose = True
            self.orbital_types = [0,1]
            self.n_orbital_type = 2
            self.dehybridize = True
            # [[[a1,a2...], [b1,b2...]],...]
            self.atoms_dehybridize = [[[0],[1]]]
        elif example=="fe_prim":
            self.filename = "fe_prim"
            self.prefix   = "examples/fe_bcc/fe_prim"
            self.out_dir  = "examples/fe_bcc/out_"+self.filename
            self.nk1, self.nk2, self.nk3  = 16, 16, 16
            self.ispin = 2
            self.fermi = 18.2568
            self.orbitals = [9]
            self.e_min = -10
            self.e_max = 10
            self.e_num = 501
            self.decompose = True
            self.dehybridize = False
        elif example=="CrNb4Se8":
            self.filename = "CrNb4Se8"
            self.prefix   = "examples/CrNb4Se8/CrNb4Se8"
            self.out_dir  = "examples/CrNb4Se8/out_"+self.filename
            self.nk1, self.nk2, self.nk3  = 8, 8, 8
            self.ispin = 2
            self.fermi = 0.
            self.orbitals = [5,5, 6,6,6,6, 6,6,6,6, 3,3,3,3, 3,3,3,3, 3,3,3,3, 3,3,3,3]
            self.e_min = -8
            self.e_max = 4
            self.e_num = 501
            self.decompose = True
            self.dehybridize = True
            self.orbitals_dehybrid = [[[0,1],[2,3,4,5,6,7,8,9]]] # index of atom
            self.n_orbital_type    = 4
            self.orbital_decompose = [0]*5*2 + ([1]+[2]*5)*8 + [3]*3*16 # index of orbital type
        elif not example:
            pass

if __name__=="__main__":
    parameters()
