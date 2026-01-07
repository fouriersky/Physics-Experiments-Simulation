import os
import math
import meep as mp
from meep import mpb

# ---------- FCC primitive lattice ----------
# primitive real-space basis (length unit a=1):
# a1=(0,1,1), a2=(1,0,1), a3=(1,1,0)
geometry_lattice = mp.Lattice(
    basis_size=mp.Vector3(math.sqrt(0.5), math.sqrt(0.5), math.sqrt(0.5)),
    basis1=mp.Vector3(0, 1, 1),
    basis2=mp.Vector3(1, 0, 1),
    basis3=mp.Vector3(1, 1, 0)
)

# ---------- geometry: diamond (Fd-3m) two-sphere motif ----------
eps = 13.0
r   = 0.25
diel = mp.Medium(epsilon=eps)
geometry = [
    mp.Sphere(r, center=mp.Vector3( 0.125,  0.125,  0.125), material=diel),
    mp.Sphere(r, center=mp.Vector3(-0.125, -0.125, -0.125), material=diel),
]

resolution = 32
num_bands  = 10

# ---------- high-symmetry k-points in fractional (b1,b2,b3) coordinates ----------
G = mp.Vector3(0.0,   0.0,  0.0)          # Γ
X = mp.Vector3(0.0,   0.5,  0.5)          # X (face center)
W = mp.Vector3(0.25,  0.75, 0.5)          # W (edge midpoint)
L = mp.Vector3(0.5,   0.5,  0.5)          # L (corner)

# path: Γ -> X -> W -> L -> G 
path_points = [
    ("G", G),
    ("X", X),
    ("W", W),
    ("L", L),
    ("G", G)
]

# in addition, add a few interior samples along each segment for your symmetry tests
def lerp(v0, v1, t):  # linear interpolation
    return mp.Vector3(v0.x*(1-t)+v1.x*t, v0.y*(1-t)+v1.y*t, v0.z*(1-t)+v1.z*t)

sample_tags = []
# Γ->X
sample_tags += [("GX_25", lerp(G, X, 0.25))]
# X->W
sample_tags += [("XW_50", lerp(X, W, 0.50))]
# W->L
sample_tags += [("WL_50", lerp(W, L, 0.50))]

sample_tags += [("LG_50", lerp(L, G, 0.50))]

# ---------- run per-k-point and export H-fields ----------
outdir = "h5_fcc"
os.makedirs(outdir, exist_ok=True)

def run_one(tag, kpt):
    print(f"Running MPB at {tag} = {kpt}")
    ms = mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        k_points=[kpt],
        resolution=resolution,
        num_bands=num_bands
    )
    ms.filename_prefix = os.path.join(outdir, f"{tag}")
    ms.run(mpb.output_at_kpoint(kpt, mpb.fix_hfield_phase, mpb.output_hfield))

for tag, kp in path_points:
    run_one(tag, kp)

# a few samples along the path
for tag, kp in sample_tags:
    run_one(tag, kp)

print("Done. H-field .h5 files under:", outdir)