import os
import re
import sys
import h5py
import numpy as np

def list_h5_for_kpoint(h5_folder, tag):
    pat = re.compile(rf"^{re.escape(tag)}-h\.k\d+\.b(\d+)\.h5$")
    items = {}
    for fn in os.listdir(h5_folder):
        m = pat.match(fn)
        if m:
            b = int(m.group(1))
            items[b] = os.path.join(h5_folder, fn)
    return items  # {band: filepath}

def ensure_3d(arr):
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 1:
        return arr[:, None, None]
    return arr

def read_complex_ds(ds):
    arr = ds[...]
    if hasattr(arr, 'dtype') and arr.dtype.fields and all(k in arr.dtype.fields for k in ('r','i')):
        return arr['r'].astype(np.float64) + 1j * arr['i'].astype(np.float64)
    return arr.astype(np.complex128)

def load_hfield_raw(h5file):
    with h5py.File(h5file, 'r') as f:
        Hx = Hy = Hz = None
        if all(k in f for k in ('x.r','x.i','y.r','y.i','z.r','z.i')):
            Hx = f['x.r'][...].astype(np.float64) + 1j * f['x.i'][...].astype(np.float64)
            Hy = f['y.r'][...].astype(np.float64) + 1j * f['y.i'][...].astype(np.float64)
            Hz = f['z.r'][...].astype(np.float64) + 1j * f['z.i'][...].astype(np.float64)
        elif 'h' in f and isinstance(f['h'], h5py.Dataset):
            data = read_complex_ds(f['h'])
            if data.ndim == 4 and data.shape[0] == 3:
                Hx, Hy, Hz = data[0], data[1], data[2]
            elif data.ndim == 4 and data.shape[-1] == 3:
                Hx, Hy, Hz = data[..., 0], data[..., 1], data[..., 2]
            else:
                raise ValueError(f"Unsupported 'h' shape: {data.shape}")
        elif 'h' in f and isinstance(f['h'], h5py.Group):
            g = f['h']; keys = set(g.keys())
            if {'x','y','z'}.issubset(keys):
                Hx = read_complex_ds(g['x']); Hy = read_complex_ds(g['y']); Hz = read_complex_ds(g['z'])
            elif {'0','1','2'}.issubset(keys):
                Hx = read_complex_ds(g['0']); Hy = read_complex_ds(g['1']); Hz = read_complex_ds(g['2'])
        elif all(k in f for k in ('hx','hy','hz')):
            Hx = read_complex_ds(f['hx']); Hy = read_complex_ds(f['hy']); Hz = read_complex_ds(f['hz'])
        elif all(k in f for k in ('h_x','h_y','h_z')):
            Hx = read_complex_ds(f['h_x']); Hy = read_complex_ds(f['h_y']); Hz = read_complex_ds(f['h_z'])
        elif all(k in f for k in ('h-x','h-y','h-z')):
            Hx = read_complex_ds(f['h-x']); Hy = read_complex_ds(f['h-y']); Hz = read_complex_ds(f['h-z'])
        else:
            raise KeyError(f"No recognized H-field datasets.")
        Hx, Hy, Hz = ensure_3d(Hx), ensure_3d(Hy), ensure_3d(Hz)
        H = np.stack([Hx, Hy, Hz], axis=0).astype(np.complex128)
        return H

def save_outputs_bundle(base, tag, bands, H_full_stack):
    # 组合保存（多个带一起，仅 full）
    np.savez_compressed(f"{base}_{tag}_bands_{'-'.join(map(str,bands))}_H_full.npz", H=H_full_stack)

def main():
    if len(sys.argv) < 4:
        print("Usage: python readh5.py <h5_folder> <tag> <band1> [band2 band3 ...]")
        print("Example: python readh5.py h5 G 3 4 5")
        sys.exit(1)
    h5_folder, tag = sys.argv[1], sys.argv[2]
    bands = [int(x) for x in sys.argv[3:]]

    files = list_h5_for_kpoint(h5_folder, tag)
    missing = [b for b in bands if b not in files]
    if missing:
        print(f"Missing bands for tag {tag}: {missing}. Available:", sorted(files.keys()))
        sys.exit(2)

    H_full_list = []

    for b in bands:
        h5file = files[b]
        H = load_hfield_raw(h5file)          # (3, Nx, Ny, Nz)
        # 单带 full 保存
        base = os.path.splitext(os.path.basename(h5file))[0]
        np.savez_compressed(f"{base}_H_full.npz", H=H)
        H_full_list.append(H)

    # 组合堆叠为 (n_bands, 3, Nx, Ny, Nz)，并保存
    H_full_stack = np.stack(H_full_list, axis=0)
    bundle_base = "bundle"
    save_outputs_bundle(bundle_base, tag, bands, H_full_stack)

    print(f"完成：{tag} bands {bands}")
    print(f"单带文件：<tag>-h.k01.bNN 对应的 *_H_full.npz")
    print(f"组合文件：{bundle_base}_{tag}_bands_{'-'.join(map(str,bands))}_H_full.npz")

if __name__ == "__main__":
    main()
