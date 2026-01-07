# Physics-Experiments-Simulation
including some codes for simulating some physical experiments,which you can use to generate data and analyze
i will keep updating some of my codes used in experiments onto this repository

this repository(developing version) now includes 2 parts : crystal structure simulation and other toy model codes.

## crystal structure simulation 
This is mainly a toy model based on LAMMPS API, more like a course homework (it actually is). I mainly realize crystal vibration simulation of diamond and metal Al, compute their strain-stress relation as well as stiffness tensor at 0K and finite temperature, I also compute their phonon spectrum by computing force constant matrix and dynamic matrix.
I use a win64-LAMMPS software installed on my own PC to compute atoms-inter-forces and energy, so the codes my not apply to other system. All in all, it's aim is for learning about related knowledges on crystal and potential surface, not to actually realize a DFT-accuracy calculation. I will update my learning notes later

## optical property simulation
use Slater-Koster method to give the band of diamond Si (0K configuration), then compute its optical properties such as $\epsilon(\omega)$ and absorption $\alpha(\omega)$, but NO phonon interaction is included, so the electron only hop directly.

## photonic crystal 
Use MPB to calculate the H field of 221 space group , do symmetry analysis to its band structure at high-symmetry k-points in 1BZ and you can get the character of thier irreducible representations, using compatibility relation and you can judge the splitting behavior of the bands , but there is some problem when $\omega$ reaches 0, which has a quite interesting reason.

