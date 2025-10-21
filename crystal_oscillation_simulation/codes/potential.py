import numpy as np

# ------------------------
# Potential interface & LJ implement
# ------------------------
def min_image(dr, cell, invcell):
    """
    dr: cartesian difference r_i - r_j
    cell: 3x3
    invcell: inverse of cell
    returns: wrapped dr (cartesian) using minimum image via fractional coords
    """
    frac = invcell @ dr
    # wrap to (-0.5, 0.5]
    frac -= np.floor(frac + 0.5)
    return cell @ frac

class PotentialBase:
    def energy_and_forces(self, positions, cell, neighbor_list=None):
        """
        返回 (total_energy, forces_array)
        forces shape (N,3)
        """
        raise NotImplementedError

class LJPotential(PotentialBase):
    def __init__(self, eps, sigma, rc=None):
        self.eps = eps
        self.sigma = sigma
        self.rc = rc if rc is not None else 2.5*sigma  # radius cutoff
        self.rc2 = self.rc*self.rc
        # shift for continuity at rc
        self.shift = 4*eps*((sigma/self.rc)**12 - (sigma/self.rc)**6)

    def energy_and_forces(self, positions, cell, neighbor_list=None):
        N = len(positions)
        forces = np.zeros_like(positions)
        energy = 0.0
        vol = np.linalg.det(cell)
        invcell = np.linalg.inv(cell)
        sigma6 = self.sigma**6
        rc2 = self.rc2

        if neighbor_list is None:
            # naive O(N^2)
            for i in range(N-1):
                ri = positions[i]
                for j in range(i+1, N):
                    dr = ri - positions[j]
                    dr = min_image(dr, cell, invcell)
                    r2 = dr.dot(dr)
                    if r2 < rc2:
                        invr2 = 1.0 / r2
                        sr6 = sigma6 * (invr2**3)
                        sr12 = sr6*sr6
                        e = 4*self.eps*(sr12 - sr6) - self.shift
                        energy += e
                        fmag = 24*self.eps*invr2*(2*sr12 - sr6)
                        fvec = fmag * dr
                        forces[i] += fvec
                        forces[j] -= fvec
        else:
            # use neighbor list to iterate pairs
            for i in range(N):
                ri = positions[i]
                for j in neighbor_list.neighbors_of(i):
                    if j <= i:  # ensure each pair handled once
                        continue
                    dr = ri - positions[j]
                    dr = min_image(dr, cell, invcell)
                    r2 = dr.dot(dr)
                    if r2 < rc2:
                        invr2 = 1.0 / r2
                        sr6 = sigma6 * (invr2**3)
                        sr12 = sr6*sr6
                        e = 4*self.eps*(sr12 - sr6) - self.shift
                        energy += e
                        fmag = 24*self.eps*invr2*(2*sr12 - sr6)
                        fvec = fmag * dr
                        forces[i] += fvec
                        forces[j] -= fvec
        return energy, forces

# add more potential EAM for Al, Tersoff for C-diamond
# to be continued...


