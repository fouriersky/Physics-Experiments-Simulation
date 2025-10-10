## Project Introduction
this code aims at simulating elastic properties of crystals, we investigate 2 kinds of crystal: Al and Si diamond, which represent two types of bonding, covalent and metallic bonds.

## Basic theory

- **Hooke theorem**
$$ \sigma = \varepsilon E $$
$\sigma$ means stress , $\varepsilon$ means strain , E means modulus of elasticity.

- **Poisson effect**
Atensile stress along the z axis causes the material to stretch along the z axis and to contract along the x and y axes.Thus define the  Poisson's Ratio $\nu$ :
$$\nu = -\frac{\text{lateral strain}}{\text{longitudinal strain}} $$

for isotropic crystal $\nu$ remain constant in 3 directions

stress tensor + strain tensor  to give strain-stress relation
$$ \sigma_{ij} = -\frac{\partial E}{\partial \eta_{ji}}$$
$\eta_{ji}$ from strain tensor $3\times 3$ on cell matrix

## Basic workflow
1. problem at 0K , we want to get different cells' strain-stress relation
   how to achieve? first do stru-optim at 0K, get ground configuration $E_0$. second give strain perturbation on cell , then do stru-optim again, now get E', finite difference to get stress. third complete the $6\times 6$ matrix. 

- **Linnard Jones Potential**
$$ V(r) = 4\varepsilon[(\frac{\sigma}{r})^{12} − (\frac{\sigma}{r})^6] $$