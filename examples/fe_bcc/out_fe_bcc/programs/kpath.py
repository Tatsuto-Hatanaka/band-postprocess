import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Kpoints
from parameters import parameters


def kpath(p):
    def get_kpath(high_sym_kpts_lines, line_dens, lat_mat):
        kpts = np.zeros((0,3))
        lengths = np.zeros(0)
        for kpts_line in high_sym_kpts_lines:
            kpts_path = []
            kpts_path.append(np.array(kpts_line[0]))
            for isp, sym_points in enumerate(kpts_line):
                if isp == len(kpts_line) - 1:
                    break
                kpt_start = np.array(sym_points)
                kpt_end = np.array(kpts_line[isp + 1])
                for ikp_inline in range(line_dens):
                    r = (ikp_inline+1) / float(line_dens)
                    kpt = kpt_end * r + kpt_start * (1. - r)
                    kpts_path.append(kpt)
            steps = np.linalg.norm(np.diff(kpts_path, axis=0)@np.linalg.inv(lat_mat).T, axis=-1)
            lengths = np.concatenate([lengths, np.concatenate([[0], steps])])
            kpts = np.concatenate([kpts, kpts_path])
        return kpts.transpose(), np.cumsum(lengths), line_dens

    def get_kpath_vasp(kpoints_file, lat_mat):
        kpoints = Kpoints.from_file(kpoints_file)
        high_sym_kpts, line_dens = kpoints.kpts, kpoints.num_kpts
        high_sym_kpts_pair = np.array(high_sym_kpts).reshape(len(high_sym_kpts)//2,2,3)

        high_sym_kpts_lines = [[]]
        high_sym_kpts_lines[-1].append(high_sym_kpts[0])
        for kpt1, kpt2 in high_sym_kpts_pair.tolist():
            if np.linalg.norm(np.array(high_sym_kpts_lines[-1][-1])-np.array(kpt1)) < 1e-9:
                high_sym_kpts_lines[-1].append(kpt2)
            else:
                high_sym_kpts_lines.append([])
                high_sym_kpts_lines[-1].append(kpt1)
                high_sym_kpts_lines[-1].append(kpt2)
        return get_kpath(high_sym_kpts_lines, line_dens, lat_mat)

    structure = Structure.from_file(p.dir+'/POSCAR')
    return get_kpath_vasp(p.dir+"/KPOINTS", structure.lattice.matrix)

if __name__=="__main__":
    p = parameters()
    assert p.example=="fe_bcc", "set self.example = 'fe_bcc'"
    kpath_, kpline, line_dens = kpath(p)
    print(kpath_[:3,30:35].transpose())
