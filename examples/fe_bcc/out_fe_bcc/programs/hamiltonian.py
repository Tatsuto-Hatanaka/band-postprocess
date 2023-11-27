import os
from time import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import norm
from make_logger import make_logger
from kpath import kpath

class hamiltonian:
    """_summary_

    Args:
        p (class): Instance of 'parameters' class.

    Attributes:
        lattice_r (np.array, shape:(3,3)): Primitive lattice vectors in real-space (cartesian [Ang] unit).
        lattice_k (np.array, shape:(3,3)): Primitive lattice vectors in reciprocal-space (cartesian [2pi/Ang] unit).
        nwan (int): Number of wannier functions including the DOF of spins.
        nrp (int): Number of lattice points where hopping terms in the TB-model are calculated using Wannier90.
        nkp (int): Number of k points specified by 'parameters.py'.
        cdeg (float): Inverse of the degree of degeneracy in the BZ.
        rvec (np.array, shape:(nrp,3)): rvec[i][i1,i2,i3], R_i = ni1 a1 + ni2 a2 + ni3 a3.
        hr (np.array, shape:(nrp,nwan,nwan)): Hamiltonian of the system in the r-space.
            H_lm(R_i) =  <w_l(0)|H|w_m(R_i)> (eV).
        hk (np.array, shape:(nkp,nwan,nwan)): Hamiltonian of the system in the k-space.
            hk[i,l,m] = H_lm(k_i) = sum_j exp(1j*k_i.R_j) H_lm(R_j).
        ek (np.array, shape:(nkp,nwan)): Eigenvalues of TB-hamiltonian.
        uk (np.array, shape:(nkp,nwan,nwan)): Eigenvectors of TB-hamiltonian.
        uk_dagger (np.array, shape:(nkp,nwan,nwan)): Hermite conjugate of uk.
    """
    def __init__(self, p):
        self.logger = make_logger(p, self.__class__.__name__)
        self.nkp = p.nkp
        if "chain1d" in p.example:
            nk1, nk2, nk3 = 101, 11, 11
            dk1, dk2, dk3 = 1./(nk1-1.), 1./(nk2-1.), 1./(nk3-1.)
            kpath_a = np.vstack([np.arange(nk1)*dk1, np.zeros(nk1), np.zeros(nk1)])
            kpath_b = np.vstack([np.ones(nk2), np.arange(nk2)*dk2, np.zeros(nk2)])
            kpath_c = np.vstack([np.ones(nk3), np.ones(nk3), np.arange(nk3)/dk3])
            self.kpath = np.hstack([kpath_a, kpath_b, kpath_c])
            self.kpline = np.concatenate([np.arange(nk1)*dk1, 1+0.5*np.arange(nk2)*dk2, 1.5+0.5*np.arange(nk3)*dk3])
        else:
            self.kpath, self.kpline, self.line_dens= kpath(p)

        if p.ispin==1 or p.ispin==3:
            self.hr = self.read_tb(p, p.dir+"/"+p.prefix+'_tb.dat')
            self.hk = self.r_to_k(p, self.hr)
            self.ek, self.uk, self.uk_dagger = self.diagonalization(p, self.hk)
            self.pdos = self.calc_pdos(p, self.ek, self.uk)
            self.pdos_orbtype = []
            self.integrated_pdos_orbtype = []
            for orb_type in range(p.n_orbital_type):
                pdos_ = self.pdos[:,np.array(p.orbital_types)==orb_type].sum(axis=1)
                self.pdos_orbtype.append(pdos_)
                self.integrated_pdos_orbtype.append(self.calc_cumulative_simpson(p.e_range, pdos_))
            self.dos = self.pdos.sum(axis=1)
            self.integrated_dos = self.calc_cumulative_simpson(p.e_range, self.dos)
            self.ek_band, self.uk_band = self.calc_band(p, self.hr, self.kpath, self.kpline)
            if p.dehybridize:
                hr = self.dehybridization(p, self.hr)
                hk = self.r_to_k(p, hr)
                self.ek_dehyb, self.uk_dehyb, self.uk_dagger_dehyb = self.diagonalization(p, hk)
                self.pdos_dehyb = self.calc_pdos(p, self.ek_dehyb, self.uk_dehyb)
                self.pdos_orbtype_dehyb = []
                self.integrated_pdos_orbtype_dehyb = []
                for orb_type in range(p.n_orbital_type):
                    pdos_dehyb = self.pdos_dehyb[:,np.array(p.orbital_types)==orb_type].sum(axis=1)
                    self.pdos_orbtype_dehyb.append(pdos_dehyb)
                    self.integrated_pdos_orbtype_dehyb.append(self.calc_cumulative_simpson(p.e_range, pdos_dehyb))
                self.dos_dehyb = self.pdos_dehyb.sum(axis=1)
                self.integrated_dos_dehyb = self.calc_cumulative_simpson(p.e_range, self.dos_dehyb)
                self.ek_band_dehyb, self.uk_band_dehyb = self.calc_band(p, hr, self.kpath, self.kpline)
        elif p.ispin==2:
            self.hr_up = self.read_tb(p, p.dir+"/"+p.prefix+'_tb.up.dat')
            self.hr_dn = self.read_tb(p, p.dir+"/"+p.prefix+'_tb.dn.dat')
            self.nwan *= 2
            self.hk_up = self.r_to_k(p, self.hr_up)
            self.hk_dn = self.r_to_k(p, self.hr_dn)
            self.ek_up, self.uk_up, self.uk_dagger_up = self.diagonalization(p, self.hk_up)
            self.ek_dn, self.uk_dn, self.uk_dagger_dn = self.diagonalization(p, self.hk_dn)
            self.pdos_up = self.calc_pdos(p, self.ek_up, self.uk_up)
            self.pdos_dn = self.calc_pdos(p, self.ek_dn, self.uk_dn)
            self.dos_up = self.pdos_up.sum(axis=1)
            self.dos_dn = self.pdos_dn.sum(axis=1)
            self.pdos_orbtype_up = []
            self.integrated_pdos_orbtype_up = []
            self.pdos_orbtype_dn = []
            self.integrated_pdos_orbtype_dn = []
            for orb_type in range(p.n_orbital_type):
                pdos_up = self.pdos_up[:,np.array(p.orbital_types)==orb_type].sum(axis=1)
                pdos_dn = self.pdos_dn[:,np.array(p.orbital_types)==orb_type].sum(axis=1)
                self.pdos_orbtype_up.append(pdos_up)
                self.integrated_pdos_orbtype_up.append(self.calc_cumulative_simpson(p.e_range, pdos_up))
                self.pdos_orbtype_dn.append(pdos_dn)
                self.integrated_pdos_orbtype_dn.append(self.calc_cumulative_simpson(p.e_range, pdos_dn))
            self.integrated_dos_up = self.calc_cumulative_simpson(p.e_range, self.dos_up)
            self.integrated_dos_dn = self.calc_cumulative_simpson(p.e_range, self.dos_dn)
            self.ek_band_up, self.uk_band_up = self.calc_band(p, self.hr_up, self.kpath, self.kpline)
            self.ek_band_dn, self.uk_band_dn = self.calc_band(p, self.hr_dn, self.kpath, self.kpline)
            if p.dehybridize:
                hr_up = self.dehybridization(p, self.hr_up)
                hr_dn = self.dehybridization(p, self.hr_dn)
                hk_up = self.r_to_k(p, hr_up)
                hk_dn = self.r_to_k(p, hr_dn)
                self.ek_dehyb_up, self.uk_dehyb_up, self.uk_dagger_dehyb_up = self.diagonalization(p, hk_up)
                self.ek_dehyb_dn, self.uk_dehyb_dn, self.uk_dagger_dehyb_dn = self.diagonalization(p, hk_dn)
                self.pdos_dehyb_up = self.calc_pdos(p, self.ek_dehyb_up, self.uk_dehyb_up)
                self.pdos_dehyb_dn = self.calc_pdos(p, self.ek_dehyb_dn, self.uk_dehyb_dn)
                self.pdos_orbtype_dehyb_up = []
                self.integrated_pdos_orbtype_dehyb_up = []
                self.pdos_orbtype_dehyb_dn = []
                self.integrated_pdos_orbtype_dehyb_dn = []
                for orb_type in range(p.n_orbital_type):
                    pdos_dehyb_up = self.pdos_dehyb_up[:,np.array(p.orbital_types)==orb_type].sum(axis=1)
                    pdos_dehyb_dn = self.pdos_dehyb_dn[:,np.array(p.orbital_types)==orb_type].sum(axis=1)
                    self.pdos_orbtype_dehyb_up.append(pdos_dehyb_up)
                    self.integrated_pdos_orbtype_dehyb_up.append(self.calc_cumulative_simpson(p.e_range, pdos_dehyb_up))
                    self.pdos_orbtype_dehyb_dn.append(pdos_dehyb_dn)
                    self.integrated_pdos_orbtype_dehyb_dn.append(self.calc_cumulative_simpson(p.e_range, pdos_dehyb_dn))
                self.dos_dehyb_up = self.pdos_dehyb_up.sum(axis=1)
                self.dos_dehyb_dn = self.pdos_dehyb_dn.sum(axis=1)
                self.integrated_dos_dehyb_up = self.calc_cumulative_simpson(p.e_range, self.dos_dehyb_up)
                self.integrated_dos_dehyb_dn = self.calc_cumulative_simpson(p.e_range, self.dos_dehyb_dn)
                self.ek_band_dehyb_up, self.uk_band_dehyb_up = self.calc_band(p, hr_up, self.kpath, self.kpline)
                self.ek_band_dehyb_dn, self.uk_band_dehyb_dn = self.calc_band(p, hr_dn, self.kpath, self.kpline)
        self.out_data_pdos(p)
        self.out_data_band(p)
        self.save_figure_pdos(p)
        self.save_figure_band(p)


    def read_tb(self, p, filename):
        """ Read TB-model data from SEEDNAME_tb.dat file (generated by Wannier90 code)

        Args:
            p (class): Instance of 'parameters' class.
            filename (str): Seedname of tb-file (formatted with SEEDNAME_tb.dat).
        Returns:
            hr (np.array, shape:(nrp,norb,norb)): Hamiltonian in the r-space.
        """
        with open(filename, 'r') as f:
            contents = f.readlines()
            f.close()
        self.lattice_r = np.fromstring(''.join(contents[1:4]), sep='\n').reshape((3,3))
        self.lattice_k = np.transpose(np.linalg.inv(self.lattice_r))
        self.nwan      = int(contents[4].strip())
        self.nrp       = int(contents[5].strip())
        ndeg_line = self.nrp // 15
        if self.nrp % 15 != 0: ndeg_line += 1
        self.cdeg  = np.fromstring(''.join(contents[6: 6+ndeg_line]), sep='\n')
        self.cdeg  = 1.0/self.cdeg
        self.rvec  = np.fromstring(''.join(contents[7+ndeg_line:7+ndeg_line+(2+self.nwan**2)*self.nrp:(2+self.nwan**2)]), sep='\n').reshape((self.nrp,3))
        assert(len(self.rvec)==self.nrp)

        contents_hr = contents[6+ndeg_line:7+ndeg_line+(2+self.nwan**2)*self.nrp]
        del contents_hr[:(2+self.nwan**2)*self.nrp:2+self.nwan**2]
        del contents_hr[:(1+self.nwan**2)*self.nrp:1+self.nwan**2]
        hr = np.fromstring(''.join(contents_hr), sep='\n').reshape(self.nrp,self.nwan,self.nwan,4)
        hr = np.transpose(hr[:,:,:,2]+1j*hr[:,:,:,3], axes=(0,2,1))
        return hr


    def r_to_k(self, p, hr):
        """ Fourier transformation of Hamiltonian from r-space to k-space
        Args:
            p (class): Instance of 'parameters' class.
            hr (np.array, shape:(nrp,nwan,nwan)): Hamiltonian in the r-space.
        Returns:
            hk (np.array, shape:(nkp,nwan,nwan)): Hamiltonian in the k-space.
        """
        s = time()
        phase = np.exp(2j*np.pi*(self.rvec@p.kpoints).T) * self.cdeg
        hr = hr.reshape(self.nrp, p.norb**2)
        hk = phase@hr
        hk = hk.reshape(self.nkp, p.norb, p.norb)
        self.logger.info("time of r_to_k = {}".format(time()-s))
        return hk


    def diagonalization(self, p, hk):
        """ Diagonalization of Hamiltonian in k-space
        Note that uk is defined such that U^\dagger H_k U is a diagonal matrix.

        Args:
            hk (np.array, shape:(nkp,nwan,nwan)): TB-hamiltonian in the k-space
        Returns:
            ek (np.array, shape:(nkp,norb)): Eigenvalues of TB-hamiltonian.
            uk (np.array, shape:(nkp,norb,norb)): Eigenvectors of TB-hamiltonian.
            uk_dagger (np.array, shape:(nkp,norb,norb)): Hermite conjugate of uk.
        """
        s = time()
        ek, uk = np.linalg.eigh(hk)
        uk_dagger = uk.transpose((0,2,1)).conjugate()
        ek -= p.fermi*np.ones((self.nkp, p.norb))
        # self.fermi_dist = 0.5*(1.0-np.tanh(0.5*p.beta*ek))
        self.logger.info("time of diagonalization = {}".format(time()-s))
        return ek, uk, uk_dagger


    def calc_band(self, p, hr, kpath, kpline):
        """ calculate the band structure on the k-path specified with the input

        Args:
            p (class): Instance of 'parameters' class
            kpath (np.array, shape:(3, nkpline)): _description_
            kpline (np.array, shape:(nkpline)): _description_
        """
        s = time()
        phase  = np.exp(2j * np.pi * np.matmul(self.rvec, kpath).transpose()) # (nkpline, nrp)
        factor = phase * self.cdeg # (nkpline, nrp)
        hk     = np.einsum('ij,jkl->ikl', factor, hr) # (nkpline, norb, norb)
        ek, uk = np.linalg.eigh(hk)
        ek -= p.fermi*np.ones((len(kpline), p.norb))
        self.logger.info("time of calc_band = {}".format(time()-s))
        return ek, uk


    def calc_pdos(self, p, ek, uk):
        s = time()
        pdos = np.zeros((p.e_num, p.norb))
        for ie, e in enumerate(p.e_range):
            delta_func = norm.pdf(e, ek, p.smearing)
            for iwan in range(p.norb):
                u2 = abs(uk[:,iwan,:])**2
                pdos[ie,iwan] = (u2*delta_func).sum()/self.nkp
        self.logger.info("time of calc_pdos = {}".format(time()-s))
        return pdos


    def dehybridization(self, p, hr):
        s = time()
        hr_ = hr.copy()
        for combs in p.atoms_dehybridize:
            for a1 in combs[0]:
                for a2 in combs[1]:
                    s1 = p.orbitals[:a1].sum()
                    s2 = p.orbitals[:a2].sum()
                    e1 = s1 + p.orbitals[a1].sum()
                    e2 = s2 + p.orbitals[a2].sum()
                    hr_[:,s1:e1, s2:e2] = 0.0
                    hr_[:,s2:e2, s1:e1] = 0.0
        self.logger.info("time of dehybridization = {}".format(time()-s))
        return hr_


    @staticmethod
    def calc_cumulative_simpson(func_range, func_vals):
        assert len(func_range)==func_vals.shape[0]
        n     = len(func_range)
        start = 0
        stop  = n-2 if n%2==1 else n-3
        de    = (func_range[-1]-func_range[0])/(n-1)
        outs  = np.zeros(n)
        outs[start:stop:2]     += func_vals[start:stop:2]
        outs[start+1:stop+1:2] += func_vals[start+1:stop+1:2]
        outs[start+2:stop+2:2] += func_vals[start+2:stop+2:2]
        outs *= de/3.
        # https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
        if n%2==0:
            alpha = 5/12
            beta  = 2/3
            eta   = 1/12
            outs[-1] += alpha * func_vals[-1] * de
            outs[-2] += beta * func_vals[-2] * de
            outs[-3] -= eta * func_vals[-3] * de
        return np.cumsum(outs)


    def out_data_pdos_template(self, p, filename, dos, integrated_dos, pdos_orbtype, integrated_pdos_orbtype):
        with open(filename,"w") as f:
                f.write("# mu, dos, integrated_dos, [pdos, integrated_pdos]*n_orb_type\n")
                for ie, e in enumerate(p.e_range):
                    f.write("{0:10.6f}  {1:10.6f}  {2:10.6f}".format(e, dos[ie], integrated_dos[ie]))
                    if p.decompose:
                        for orb_type in range(p.n_orbital_type):
                            f.write("  {0:10.6f}  {1:10.6f}".format(\
                                pdos_orbtype[orb_type][ie], integrated_pdos_orbtype[orb_type][ie]))
                    f.write("\n")
                f.write("\n")
                f.close()



    def out_data_pdos(self, p):
        s = time()
        filename = p.filename+"_pdos"
        if p.ispin==1 or p.ispin==3:
            self.out_data_pdos_template(p, p.out_dir+"/"+filename+".dat", self.dos, self.integrated_dos, self.pdos_orbtype, self.integrated_pdos_orbtype)
            if p.dehybridize:
                self.out_data_pdos_template(p, p.out_dir+"/"+filename+"_dehyb.dat"\
                    , self.dos_dehyb, self.integrated_dos_dehyb, self.pdos_orbtype_dehyb, self.integrated_pdos_orbtype_dehyb)
        elif p.ispin==2:
            self.out_data_pdos_template(p, p.out_dir+"/"+filename+"_up.dat"\
                , self.dos_up, self.integrated_dos_up, self.pdos_orbtype_up, self.integrated_pdos_orbtype_up)
            self.out_data_pdos_template(p, p.out_dir+"/"+filename+"_dn.dat"\
                , self.dos_dn, self.integrated_dos_dn, self.pdos_orbtype_dn, self.integrated_pdos_orbtype_dn)
            if p.dehybridize:
                self.out_data_pdos_template(p, p.out_dir+"/"+filename+"_dehyb_up.dat", self.dos_dehyb_up\
                    , self.integrated_dos_dehyb_up, self.pdos_orbtype_dehyb_up, self.integrated_pdos_orbtype_dehyb_up)
                self.out_data_pdos_template(p, p.out_dir+"/"+filename+"_dehyb_dn.dat", self.dos_dehyb_dn\
                    , self.integrated_dos_dehyb_dn, self.pdos_orbtype_dehyb_dn, self.integrated_pdos_orbtype_dehyb_dn)
        self.logger.info("time of out_data_pdos = {}".format(time()-s))


    def out_data_band_template(self, p, filename, kpline, ek_band, uk_band):
        with open(filename,"w") as f:
            f.write("# kpline, ek, [weight] * n_orb_type\n")
            for iwan in range(ek_band.shape[1]):
                for ikp in range(ek_band.shape[0]):
                    f.write("{0:10.6f}  {1:10.6f}".format(kpline[ikp], ek_band[ikp][iwan].real))
                    if p.decompose:
                        weight = abs(uk_band[ikp,:,iwan])**2
                        for orb_type in range(p.n_orbital_type):
                            f.write("  {0:10.6f}".format(sum(weight[np.array(p.orbital_types)==orb_type])))
                    f.write("\n")
                f.write("\n")
            f.close()


    def out_data_band(self, p):
        s = time()
        filename = p.filename+"_bands"
        if p.ispin==1 or p.ispin==3:
            self.out_data_band_template(p, p.out_dir+"/"+filename+".dat", self.kpline, self.ek_band, self.uk_band)
            if p.dehybridize:
                self.out_data_band_template(p, p.out_dir+"/"+filename+"_dehyb.dat"\
                , self.kpline, self.ek_band_dehyb, self.uk_band_dehyb)
        elif p.ispin==2:
            self.out_data_band_template(p, p.out_dir+"/"+filename+"_up.dat", self.kpline, self.ek_band_up, self.uk_band_up)
            self.out_data_band_template(p, p.out_dir+"/"+filename+"_dn.dat", self.kpline, self.ek_band_dn, self.uk_band_dn)
            if p.dehybridize:
                self.out_data_band_template(p, p.out_dir+"/"+filename+"_dehyb_up.dat"\
                    , self.kpline, self.ek_band_dehyb_up, self.uk_band_dehyb_up)
                self.out_data_band_template(p, p.out_dir+"/"+filename+"_dehyb_dn.dat"\
                , self.kpline, self.ek_band_dehyb_dn, self.uk_band_dehyb_dn)
        self.logger.info("time of out_data_band = {}".format(time()-s))


    def save_figure_pdos(self, p):
        s = time()
        filename = p.filename+"_pdos"
        fig, ax = plt.subplots(1, 1, figsize=(8,5), tight_layout=True)
        ax.set_title("PDOS")
        ax.set_xlim(p.e_min,p.e_max)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("PDOS")
        ax.axvline(0, color="black", linestyle='dashed', linewidth=0.5)
        ax.axhline(0, color="black", linestyle='dashed', linewidth=0.5)
        if p.ispin==1 or p.ispin==3:
            for iorbtype in range(p.n_orbital_type):
                ax.plot(p.e_range, self.pdos_orbtype[iorbtype], label="type{}".format(iorbtype+1), linewidth=0.5)
            if p.dehybridize:
                for iorbtype in range(p.n_orbital_type):
                    ax.plot(p.e_range, self.pdos_orbtype_dehyb[iorbtype], label="type{}-dehyb".format(iorbtype+1), linewidth=0.5)
        elif p.ispin==2:
            for iorbtype in range(p.n_orbital_type):
                ax.plot(p.e_range,  self.pdos_orbtype_up[iorbtype], label="type{}-up".format(iorbtype+1), linewidth=0.5)
                ax.plot(p.e_range, -self.pdos_orbtype_dn[iorbtype], label="type{}-dn".format(iorbtype+1), linewidth=0.5)
            if p.dehybridize:
                for iorbtype in range(p.n_orbital_type):
                    ax.plot(p.e_range,  self.pdos_orbtype_dehyb_up[iorbtype], label="type{}-up-dehyb".format(iorbtype+1), linewidth=0.5)
                    ax.plot(p.e_range, -self.pdos_orbtype_dehyb_dn[iorbtype], label="type{}-dn-dehyb".format(iorbtype+1), linewidth=0.5)
        ax.legend()
        fig.savefig(p.fig_dir+"/"+filename+".pdf")
        self.logger.info("time of save_figure_pdos = {}".format(time()-s))


    def save_figure_band(self, p):
        s = time()
        filename = p.filename+"_bands"
        fig, ax = plt.subplots(1, 1, figsize=(8,5), tight_layout=True)
        ax.set_title("Band structure")
        ax.set_xlim(self.kpline[0],self.kpline[-1])
        if p.example=="fe_bcc": ax.set_ylim(-9, 4)
        ax.set_ylabel("Energy (eV)")
        ax.set_xlabel(r"$\boldsymbol{k}$")
        ax.axhline(0, color="black", linestyle='dashed', linewidth=0.5)
        if p.ispin==1 or p.ispin==3:
            for iband in range(self.ek_band.shape[1]):
                ax.plot(self.kpline, self.ek_band[:,iband], c="red", linewidth=0.5)
            if p.dehybridize:
                for iband in range(self.ek_band_dehyb.shape[1]):
                    if iband==0:
                        ax.plot(self.kpline, self.ek_band_dehyb[:,iband], c="green", label="dehybridized", linewidth=0.5)
                    else:
                        ax.plot(self.kpline, self.ek_band_dehyb[:,iband], c="green", linewidth=0.5)
        elif p.ispin==2:
            for iband in range(self.ek_band_up.shape[1]):
                if iband==0:
                    ax.plot(self.kpline, self.ek_band_up[:,iband], c="blue", label="up",linewidth=0.5)
                else:
                    ax.plot(self.kpline, self.ek_band_up[:,iband], c="blue",linewidth=0.5)
            for iband in range(self.ek_band_dn.shape[1]):
                if iband==0:
                    ax.plot(self.kpline, self.ek_band_dn[:,iband], c="red", label="dn",linewidth=0.5)
                else:
                    ax.plot(self.kpline, self.ek_band_dn[:,iband], c="red",linewidth=0.5)
            if p.dehybridize:
                for iband in range(self.ek_band_dehyb_up.shape[1]):
                    if iband==0:
                        ax.plot(self.kpline, self.ek_band_dehyb_up[:,iband], c="green", label="dehybridized", linewidth=0.5)
                        ax.plot(self.kpline, self.ek_band_dehyb_dn[:,iband], c="green", linewidth=0.5)
                    else:
                        ax.plot(self.kpline, self.ek_band_dehyb_up[:,iband], c="green", linewidth=0.5)
                        ax.plot(self.kpline, self.ek_band_dehyb_dn[:,iband], c="green", linewidth=0.5)
        if "chain1d" in p.example:
            ax.set_xticks([0, 1.0, 1.5, 2.0])
            ax.set_xticklabels(["(0,0,0)", "(1,0,0)", "(1,1,0)", "(1,1,1)"])
            ax.axvline(1.0, color="black", linestyle='dashed', linewidth=0.5)
            ax.axvline(1.5, color="black", linestyle='dashed', linewidth=0.5)
        else:
            for isp in range(1, self.kpath.shape[1]//self.line_dens):
                ax.axvline(self.kpline[self.line_dens*isp], color="black", linestyle='dashed', linewidth=0.5)
        ax.legend()
        fig.savefig(p.fig_dir+"/"+filename+".pdf")
        self.logger.info("time of save_figure_band = {}".format(time()-s))

