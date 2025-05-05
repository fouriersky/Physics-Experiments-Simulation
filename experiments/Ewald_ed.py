import numpy as np
from ase.io import read
from ewald import ewaldsum

# calculate madelung constant using ewald method

if __name__ == '__main__':

    crystals = [
        'NaCl.vasp',
    ]

    ZZ = {
        'Na':  1, 
        'Cl': -1 
    }

    print('-' * 41)
    print(f'{"Crystal":>9s} | Ref Atom | {"Madelung Constant":>18s}')
    print('-' * 41)

    for crys in crystals:
        atoms = read(crys)
        esum = ewaldsum(atoms, ZZ) 

        # print(esum.get_ewaldsum())
        M = esum.get_madelung()
        C = crys.replace('.vasp', '')
        R = atoms.get_chemical_symbols()[0]
        print(f'{C:>9s} | {R:^8s} | {M:18.12f}')

    print('-' * 41)
