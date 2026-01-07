import os
import meep as mp
from meep import mpb

# ============ geometry: simple cubic lattice with one dielectric sphere ============
# real-space lattice vectors (conventional cubic cell with a=1)
geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, 1, 1),      # conventional cell
    basis1=mp.Vector3(1, 0, 0),
    basis2=mp.Vector3(0, 1, 0),
    basis3=mp.Vector3(0, 0, 1)
)

# dielectric parameters (you can change to your benchmark values)
eps_sphere = 13.0   # sphere dielectric (benchmark figure uses ~13)
radius = 0.30       # sphere radius in units of a (benchmark ~0.3)
diel = mp.Medium(epsilon=eps_sphere)

# simple cubic: one sphere at the origin per cell
geometry = [
    mp.Sphere(radius, center=mp.Vector3(0, 0, 0), material=diel)
]

resolution = 32     # increase for cleaner fields/irrep analysis
num_bands  = 12     # compute more bands to match the benchmark figure better

# ============ high-symmetry points for simple cubic BZ ============
# coordinates are in fractional reciprocal-lattice units (relative to b1,b2,b3)
# Path to mimic the benchmark figure: Γ – Δ – X – S – R – T – M – Σ – Γ
kdict = {
    # endpoints and path samples
    "G":        mp.Vector3(0.0, 0.0, 0.0),         # Γ
    # Δ line (Γ -> X) samples
    "Delta": mp.Vector3(0.0, 0.0, 0.25),
    # X (1/2,0,0)
    "X":        mp.Vector3(0.0, 0.0, 0.50),
    # S point often coincides with M in cubic (1/2,1/2,0); keep both tags for clarity
    "S":        mp.Vector3(0.20, 0.20, 0.50),
    # R (1/2,1/2,1/2)
    "R":        mp.Vector3(0.50, 0.50, 0.50),
    # T (1/2,0,1/2)
    "T":        mp.Vector3(0.50, 0.50, 0.20),
    # M (1/2,1/2,0)
    "M":        mp.Vector3(0.50, 0.50, 0.0),
    # Σ line (M -> Γ) a sample point
    "Sigma": mp.Vector3(0.25, 0.25, 0.0),       # halfway from M to Γ in the (h,h,0) plane
}

# ============ run per-k-point and export H field patterns ============
outdir = "h5_sc"
os.makedirs(outdir, exist_ok=True)

for tag, kp in kdict.items():
    print(f"Running MPB at {tag} = {kp}")
    ms = mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        k_points=[kp],
        resolution=resolution,
        num_bands=num_bands
    )
    # choose a per-k prefix to make files easy to group
    ms.filename_prefix = os.path.join(outdir, f"{tag}")
    # export H-field eigenmodes at this k (complex-valued)
    ms.run(mpb.output_at_kpoint(kp, mpb.fix_hfield_phase, mpb.output_hfield))

print("Done. H-field .h5 files are saved under:", outdir)