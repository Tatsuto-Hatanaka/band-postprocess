# band-postprocess

## Overview

Post-processing of the band structure of the tight-binding model obtained from Wannier90 and VASP.

## What

- Decomposing
    Decompose the band structure into contributions from each orbital (orbital-wise).

- De-hybridization
    Calculate the band structure when ignoring hopping terms between some atoms (atom-wise). Ignoring means setting the hopping terms in the hamiltonian to 0; we set $\langle w_{l}(0)|H|w_{m}(R)\rangle=0$.

- Projected density of state (PDOS)
    Calculate PDOS of each orbital from the band structure.

## Usage

1. Prepare `SEEDNAME_tb.dat` from Wannier90, `POSCAR`, and `KPOINTS` from VASP in the same directory
2. Edit some parameters in `parameters.py`
3. Run the code below

```python
python main.py
```

## Note

## Author

Tatsuto Hatanaka

## License

MIT license
