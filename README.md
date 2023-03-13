# band-postprocess
post-processing of the band structure based on the tight-binding model from Wannier90


### What
- Decomposing <br>
Decompose the band structure into contributions from each orbital (orbital-wise).

- De-hybridization <br>
Calculate the band structure when ignoring hopping terms between some atoms (atom-wise). Ignoring means setting the hopping terms in the hamiltonian to 0; we set $\langle w_{l}(0)|H|w_{m}(R)\rangle=0$.

- Projected density of state (PDOS) <br>
Calculate PDOS of each orbital from the band structure.


### Usage
1.  Prepare `SEEDNAME_tb.dat` from Wannier90, `POSCAR`, and `KPOINTS` from VASP in the same directory
2.  Edit some parameters in `parameters.py`
3.  Run the code below
```
python main.py
```

### Note


### Author
Tatsuto Hatanaka
