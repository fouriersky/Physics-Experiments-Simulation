# Physics-Experiments-Simulation
including some codes for simulating some physical experiments,which you can use to generate data and analyze
i will keep updating some of my codes used in experiments onto this repository

this repository(developing version) now includes 2 parts : crystal structure simulation and other toy model codes.

## crystal structure simulation 
This is mainly a toy model based on LAMMPS API, more like a course homework (it actually is). I mainly realize crystal vibration simulation of diamond and metal Al, compute their strain-stress relation as well as stiffness tensor at 0K and finite temperature, I also compute their phonon spectrum by computing force constant matrix and dynamic matrix.
I use a win64-LAMMPS software installed on my own PC to compute atoms-inter-forces and energy, so the codes my not apply to other system. All in all, it's aim is for learning about related knowledges on crystal and potential surface, not to actually realize a DFT-accuracy calculation. I will update my learning notes later
