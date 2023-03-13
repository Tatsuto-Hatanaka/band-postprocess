from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Kpoints
from numpy import *

def kpath():
  def get_kpath(sp_kpts, kpt_den, lat_mat):
    kpts = zeros((0,3))
    lens = zeros(0)
    for sp_kpt_path in sp_kpts:
      kpts_path = []
      kpts_path.append(array(sp_kpt_path[0]))
      for kpt_ind, kpt in enumerate(sp_kpt_path):
        if kpt_ind == len(sp_kpt_path) - 1:
          break
        kpt_i = array(kpt)
        kpt_f = array(sp_kpt_path[kpt_ind + 1])
        for seg_i in range(kpt_den):
          frac = (seg_i + 1.) / float(kpt_den)
          kpt_seg = kpt_f * frac + kpt_i * (1. - frac)
          kpts_path.append(kpt_seg)
      diffs = linalg.norm(diff(kpts_path, axis=0)@linalg.inv(lat_mat).T, axis=-1)
      lens = concatenate([lens, concatenate([[0],diffs])])
      kpts = concatenate([kpts,kpts_path])
    return kpts, cumsum(lens)

  def get_kpath_vasp(kpoints_file, lat_mat):
    kpoints = Kpoints.from_file(kpoints_file)
    kpts, dens = kpoints.kpts, kpoints.num_kpts
    kpts_pair = array(kpts).reshape(len(kpts)//2,2,3)

    kpts_ = [[]]
    kpts_[-1].append(kpts[0])
    for kpt1, kpt2 in kpts_pair.tolist():
      if linalg.norm(array(kpts_[-1][-1])-array(kpt1))<1e-9:
        kpts_[-1].append(kpt2)
      else:
        kpts_.append([])
        kpts_[-1].append(kpt1)
        kpts_[-1].append(kpt2)

    return get_kpath(kpts_, dens, lat_mat)

  structure = Structure.from_file('POSCAR')
  return get_kpath_vasp("KPOINTS", structure.lattice.matrix)
