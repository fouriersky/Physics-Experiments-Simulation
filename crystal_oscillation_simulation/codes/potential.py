import numpy as np
import os
import subprocess
import tempfile

def _to_frac(R, cell):
    inv = np.linalg.inv(cell)
    return (inv @ R.T).T  # (N,3)

def _from_frac(S, cell):
    return (cell @ S.T).T

def _wrap01(S):
    # wrap to [0,1)
    return S - np.floor(S)

def _align_positions_to_reference(R_new, R_ref, cell):
    """
    仅做全局平移+PBC 包裹的对齐（不做旋转/重排）。
    """
    S_new = _wrap01(_to_frac(np.asarray(R_new), cell))
    S_ref = _wrap01(_to_frac(np.asarray(R_ref), cell))
    # 用第一个原子作参考，求 fractional 平移增量
    delta = S_ref[0] - S_new[0]
    S_aligned = _wrap01(S_new + delta)
    return _from_frac(S_aligned, cell)

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


# ------------------------
# Potential interface & LJ implement
# ------------------------
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

import shutil

class DirectLAMMPSEAMPotential(PotentialBase):
    """
    直接调用本地 lmp.exe：
    - 生成 data 文件（atomic/metal/正交晶胞）
    - 生成 in 脚本（run 0）
    - dump per-atom forces 到文本并解析
    - thermo 打印势能到文件并解析

    仅支持正交晶胞（cell 为对角阵）。如果需要斜盒，需扩展 data 文件的倾斜项。
    """
    def __init__(self, eam_file, lmp_cmd="lmp.exe", element="Al",
                 mass=26.981538, pair_style="eam/alloy",
                 keep_tmp_files=False):
        self.eam_file = str(eam_file)
        self.lmp_cmd = str(lmp_cmd)
        self.element = str(element)
        self.mass = float(mass)
        self.pair_style = str(pair_style)   # "eam" 或 "eam/alloy"
        self.keep_tmp_files = bool(keep_tmp_files)

    def _check_exec(self):
        try:
            proc = subprocess.run([self.lmp_cmd, "-h"], capture_output=True, text=True, timeout=10)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr or proc.stdout)
        except Exception as e:
            raise RuntimeError(f"无法执行 LAMMPS 可执行文件：{self.lmp_cmd}") from e
        
    @staticmethod
    def _qr_upper_tri(cell):
        """
        将任意 3x3 晶胞矩阵 H（列为晶格向量）分解为 H = Q @ R，
        其中 Q 为正交矩阵，R 为上三角（对角线取正）。
        返回 Q, R。
        """
        H = np.asarray(cell, dtype=float)
        # numpy 的 QR: H = Q R
        Q, R = np.linalg.qr(H)
        # 保证 R 对角线为正，同时调整 Q 的列符号以保持 H = Q R
        for i in range(3):
            if R[i, i] < 0:
                R[i, :] *= -1.0
                Q[:, i] *= -1.0
        return Q, R
    
    @staticmethod
    def _write_data_atomic_triclinic(path, cell, positions, mass_amu=1.0):
        """
        将任意 cell 写成 LAMMPS triclinic data（tilt factors）。
        使用 QR 将 cell 旋到上三角 R，并把坐标旋到同一系下写入。
        返回 (Q, R)，用于运行后把坐标旋回原坐标系。
        """
        Q, R = DirectLAMMPSEAMPotential._qr_upper_tri(cell)
        # LAMMPS triclinic 参数（R 为上三角）
        lx = float(R[0, 0])
        xy = float(R[0, 1])
        xz = float(R[0, 2])
        ly = float(R[1, 1])
        yz = float(R[1, 2])
        lz = float(R[2, 2])

        # 坐标旋到 R 坐标系：r' = Q^T r
        positions = np.asarray(positions, dtype=float)
        pos_lmp = (Q.T @ positions.T).T

        N = len(pos_lmp)
        with open(path, "w", newline="\n") as f:
            f.write("LAMMPS data file (atomic/triclinic)\n\n")
            f.write(f"{N} atoms\n")
            f.write("1 atom types\n\n")
            # 盒子边界（原点在 0），以及 tilt 因子
            f.write(f"{0.0:.16g} {lx:.16g} xlo xhi\n")
            f.write(f"{0.0:.16g} {ly:.16g} ylo yhi\n")
            f.write(f"{0.0:.16g} {lz:.16g} zlo zhi\n")
            f.write(f"{xy:.16g} {xz:.16g} {yz:.16g} xy xz yz\n\n")
            f.write("Masses\n\n")
            #f.write(f"1 {26.981538:.6f}\n\n")  # 占位，主脚本里仍会设置 mass 1
            f.write(f"1 {float(mass_amu):.6f}\n\n")
            f.write("Atoms # atomic\n\n")
            for i, (x, y, z) in enumerate(pos_lmp, start=1):
                f.write(f"{i} 1 {x:.16g} {y:.16g} {z:.16g}\n")
        return Q, R

    @staticmethod
    def _is_orthorhombic(cell, tol=1e-12):
        cell = np.asarray(cell, dtype=float)
        off = cell - np.diag(np.diag(cell))
        return np.all(np.abs(off) < tol)
    
    @staticmethod
    def _parse_energy(path_Eout):
        with open(path_Eout, "r") as f:
            s = f.read().strip()
        return float(s)
    
    def _make_input_script(self, path_in, basename_pot, path_data, path_dump, path_Eout):
        lines = []
        lines += [
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            "box tilt large",   # 即使是正交盒，也兼容语法（不使用倾斜）
            f"read_data {path_data}",
            "",
            "### interactions",
            f"pair_style {self.pair_style}",
        ]
        # 势文件与映射
        if "alloy" in self.pair_style:
            # setfl：'* * 文件名 元素'
            lines.append(f"pair_coeff * * {basename_pot} {self.element}")
        else:
            # funcfl 单元素：'文件名'
            lines.append(f"pair_coeff * * {basename_pot}")
        lines += [
            f"mass 1 {self.mass:.6f}",
            "",
            "### run (single-point)",
            "neighbor 2.0 bin",
            "neigh_modify delay 0 every 1 check yes",
            "thermo_style custom pe",
            "thermo 1",
            # dump 文本输出，方便解析
            f"dump d1 all custom 1 {path_dump} id type x y z fx fy fz",
            'dump_modify d1 sort id format line "%d %d %.16g %.16g %.16g %.16g %.16g %.16g"',
            "run 0",
            "variable e equal pe",
            f'print "${{e}}" file {path_Eout} screen no',
            "undump d1",
            ""
        ]
        with open(path_in, "w", newline="\n") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _parse_positions_from_dump(path_dump, N):
        # 解析 write_dump/custom 输出（或 custom dump 的最后一帧）
        data = []
        with open(path_dump, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.lower().startswith("item:"):
                    continue
                parts = line.split()
                # id type x y z
                if len(parts) < 5:
                    continue
                _id = int(parts[0])
                x, y, z = map(float, parts[2:5])
                data.append((_id, x, y, z))
        if not data:
            raise RuntimeError("未在 dump 文件中解析到位置数据。")
        data.sort(key=lambda t: t[0])
        if len(data) < N:
            raise RuntimeError(f"dump 中原子数量不足：{len(data)} < {N}")
        R = np.array([[x, y, z] for (_id, x, y, z) in data[:N]], dtype=float)
        return R

    def relax_positions_fixed_cell(self, positions, cell,
                                   min_style="fire",
                                   e_tol=1e-10, f_tol=1e-6,
                                   maxiter=10000, maxeval=10000,
                                   align_to_input=True):
        """
        固定晶格常数，仅优化原子位置。返回 (positions_relaxed, energy_eV)
        """
        self._check_exec()
        positions = np.asarray(positions, dtype=float)
        cell = np.asarray(cell, dtype=float)

        N = len(positions)
        tmp_dir = tempfile.mkdtemp(prefix="direct_lmp_min_")
        try:
            # 势文件复制到临时目录
            pot_basename = os.path.basename(self.eam_file)
            pot_local = os.path.join(tmp_dir, pot_basename)
            shutil.copy2(self.eam_file, pot_local)

            # 写 data 与 in 脚本
            data_path = os.path.join(tmp_dir, "data.in")
            dump_final = os.path.join(tmp_dir, "dump_min_final.txt")
            Eout_path  = os.path.join(tmp_dir, "energy_min.out")
            in_path    = os.path.join(tmp_dir, "in.min")

            #self._write_data_atomic_ortho(data_path, cell, positions)
            Q, R = self._write_data_atomic_triclinic(data_path, cell, positions, mass_amu=self.mass)

            lines = []
            lines += [
                "units metal",
                "atom_style atomic",
                "boundary p p p",
                "box tilt large",
                f"read_data {data_path}",
                "",
                "### interactions",
                f"pair_style {self.pair_style}",
            ]
            if "alloy" in self.pair_style:
                lines.append(f"pair_coeff * * {pot_basename} {self.element}")
            else:
                lines.append(f"pair_coeff * * {pot_basename}")
            lines += [
                f"mass 1 {self.mass:.6f}",
                "",
                "### minimization (fixed cell)",
                "neighbor 2.0 bin",
                "neigh_modify delay 0 every 1 check yes",
                f"min_style {min_style}",
                "min_modify dmax 0.1 line quadratic",
                "reset_timestep 0",
                "thermo_style custom step pe fnorm",
                "thermo 50",
                f"minimize {e_tol:.3e} {f_tol:.3e} {int(maxiter)} {int(maxeval)}",
                # 写出最终一帧坐标
                f"write_dump all custom {dump_final} id type x y z",
                # 输出最终能量
                "variable e equal pe",
                f'print "${{e}}" file {Eout_path} screen no',
                ""
            ]
            with open(in_path, "w", newline="\n") as f:
                f.write("\n".join(lines))

            # 运行 LAMMPS
            proc = subprocess.run([self.lmp_cmd, "-in", in_path],
                                  cwd=tmp_dir, capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                stdout_tail = (proc.stdout or "")[-2000:]
                stderr_tail = (proc.stderr or "")[-2000:]
                raise RuntimeError(
                    f"最小化失败（返回码 {proc.returncode}）。\n--- STDOUT ---\n{stdout_tail}\n--- STDERR ---\n{stderr_tail}"
                )

            # 解析能量与位置
            E = self._parse_energy(Eout_path)
            R_lmp = self._parse_positions_from_dump(dump_final, N)
            R_cart = (Q @ R_lmp.T).T
            if align_to_input:
                R_cart = _align_positions_to_reference(R_cart, positions, cell)
            return R_cart, float(E)
        finally:
            if not self.keep_tmp_files:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

class DirectLAMMPSLCBOPPotential:
    """
    直接调用本地 LAMMPS（lmp.exe）计算 C/diamond 的 LCBOP 势。
    功能：与 DirectLAMMPSEAMPotential 相同，用于 relax_positions_fixed_cell。
    """
    def __init__(self, lcbop_file, lmp_cmd="lmp.exe", element="C",
                 mass=12.011, pair_style="lcbop",
                 keep_tmp_files=False):
        self.lcbop_file = str(lcbop_file)
        self.lmp_cmd = str(lmp_cmd)
        self.element = str(element)
        self.mass = float(mass)
        self.pair_style = str(pair_style)
        self.keep_tmp_files = bool(keep_tmp_files)

    def _check_exec(self):
        try:
            proc = subprocess.run([self.lmp_cmd, "-h"], capture_output=True, text=True, timeout=10)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr or proc.stdout)
        except Exception as e:
            raise RuntimeError(f"无法执行 LAMMPS 可执行文件：{self.lmp_cmd}") from e

    def relax_positions_fixed_cell(self, positions, cell,
                                   min_style="fire",
                                   e_tol=1e-10, f_tol=1e-6,
                                   maxiter=10000, maxeval=10000,
                                   align_to_input=True):
        """
        固定晶格常数，仅优化原子位置。
        返回 (positions_relaxed, energy_eV)
        """
        import tempfile, subprocess, os, numpy as np, shutil
        from pathlib import Path

        self._check_exec()
        positions = np.asarray(positions, dtype=float)
        cell = np.asarray(cell, dtype=float)
        N = len(positions)
        tmp_dir = tempfile.mkdtemp(prefix="direct_lmp_min_lcbop_")

        try:
            pot_basename = os.path.basename(self.lcbop_file)
            pot_local = os.path.join(tmp_dir, pot_basename)
            shutil.copy2(self.lcbop_file, pot_local)

            data_path = os.path.join(tmp_dir, "data.in")
            dump_final = os.path.join(tmp_dir, "dump_min_final.txt")
            Eout_path  = os.path.join(tmp_dir, "energy_min.out")
            in_path    = os.path.join(tmp_dir, "in.min")

            # 调用已有的 QR + triclinic 写出函数
            Q, R = DirectLAMMPSEAMPotential._write_data_atomic_triclinic(data_path, cell, positions, mass_amu=self.mass)

            lines = []
            lines += [
                "units metal",
                "atom_style atomic",
                "boundary p p p",
                "box tilt large",
                f"read_data {data_path}",
                "",
                "### interactions",
                f"pair_style {self.pair_style}",
                f"pair_coeff * * {pot_basename} {self.element}",
                f"mass 1 {self.mass:.6f}",
                "",
                "neighbor 2.0 bin",
                "neigh_modify delay 0 every 1 check yes",
                f"min_style {min_style}",
                "min_modify dmax 0.1 line quadratic",
                "reset_timestep 0",
                "thermo_style custom step pe fnorm",
                "thermo 50",
                f"minimize {e_tol:.3e} {f_tol:.3e} {int(maxiter)} {int(maxeval)}",
                f"write_dump all custom {dump_final} id type x y z",
                "variable e equal pe",
                f'print "${{e}}" file {Eout_path} screen no',
                ""
            ]
            with open(in_path, "w", newline="\n") as f:
                f.write("\n".join(lines))

            proc = subprocess.run([self.lmp_cmd, "-in", in_path],
                                  cwd=tmp_dir, capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                stdout_tail = (proc.stdout or "")[-2000:]
                stderr_tail = (proc.stderr or "")[-2000:]
                raise RuntimeError(
                    f"最小化失败（返回码 {proc.returncode}）。\n--- STDOUT ---\n{stdout_tail}\n--- STDERR ---\n{stderr_tail}"
                )

            E = DirectLAMMPSEAMPotential._parse_energy(Eout_path)
            R_lmp = DirectLAMMPSEAMPotential._parse_positions_from_dump(dump_final, N)
            R_cart = (Q @ R_lmp.T).T
            if align_to_input:
                R_cart = _align_positions_to_reference(R_cart, positions, cell)
            return R_cart, float(E)

        finally:
            if not self.keep_tmp_files:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _parse_stress_bar_to_eVA3(path_stress):
        # 文件一行6个数：pxx pyy pzz pxy pxz pyz（单位 bar，LAMMPS 压力为正=受压）
        with open(path_stress, "r") as f:
            parts = f.read().strip().split()
        vals_bar = list(map(float, parts))  # 6 floats
        # 转为 eV/Å^3，并翻转号：sigma = -pressure
        BAR_TO_EVA3 = 6.241509074e-7
        vals_eVA3 = [-v * BAR_TO_EVA3 for v in vals_bar]
        # 还原为 3x3 张量顺序: [pxx,pyy,pzz,pxy,pxz,pyz]
        sxx, syy, szz, sxy, sxz, syz = vals_eVA3
        sigma = np.array([[sxx, sxy, sxz],
                          [sxy, syy, syz],
                          [sxz, syz, szz]], dtype=float)
        return sigma

    def relax_energy_and_stress(self, positions, cell,
                                min_style="fire",
                                e_tol=1e-12, f_tol=1e-6,
                                maxiter=20000, maxeval=200000,
                                align_to_input=True):
        """
        固定晶格常数，原子弛豫后同时返回 (R_relaxed, E[eV], sigma[3x3] (eV/Å^3, 张量约定: 拉伸为正))
        """
        self._check_exec()
        positions = np.asarray(positions, dtype=float)
        cell = np.asarray(cell, dtype=float)
        N = len(positions)
        import shutil, os, subprocess, tempfile
        tmp_dir = tempfile.mkdtemp(prefix="direct_lmp_min_lcbop_")
        try:
            pot_basename = os.path.basename(self.lcbop_file)
            pot_local = os.path.join(tmp_dir, pot_basename)
            shutil.copy2(self.lcbop_file, pot_local)

            data_path  = os.path.join(tmp_dir, "data.in")
            dump_final = os.path.join(tmp_dir, "dump_min_final.txt")
            Eout_path  = os.path.join(tmp_dir, "energy_min.out")
            Sout_path  = os.path.join(tmp_dir, "stress.out")
            in_path    = os.path.join(tmp_dir, "in.min")

            # 任意晶胞：QR -> triclinic
            Q, R = DirectLAMMPSEAMPotential._write_data_atomic_triclinic(data_path, cell, positions, mass_amu=self.mass)

            lines = []
            lines += [
                "units metal",
                "atom_style atomic",
                "boundary p p p",
                "box tilt large",
                f"read_data {data_path}",
                "",
                "### interactions",
                f"pair_style {self.pair_style}",
                f"pair_coeff * * {pot_basename} {self.element}",
                f"mass 1 {self.mass:.6f}",
                "",
                "neighbor 2.0 bin",
                "neigh_modify delay 0 every 1 check yes",
                f"min_style {min_style}",
                "min_modify dmax 0.1 line quadratic",
                "reset_timestep 0",
                "thermo_style custom step pe fnorm",
                "thermo 50",
                f"minimize {e_tol:.3e} {f_tol:.3e} {int(maxiter)} {int(maxeval)}",
                # dump 最终坐标（在旋转系）
                f"write_dump all custom {dump_final} id type x y z",
                # 能量
                "variable e equal pe",
                f'print "${{e}}" file {Eout_path} screen no',
                # 整体 virial 压力张量（bar）
                "compute p all pressure NULL virial",
                'variable pxx equal c_p[1]',
                'variable pyy equal c_p[2]',
                'variable pzz equal c_p[3]',
                'variable pxy equal c_p[4]',
                'variable pxz equal c_p[5]',
                'variable pyz equal c_p[6]',
                f'print "${{pxx}} ${{pyy}} ${{pzz}} ${{pxy}} ${{pxz}} ${{pyz}}" file {Sout_path} screen no',
                ""
            ]
            with open(in_path, "w", newline="\n") as f:
                f.write("\n".join(lines))

            proc = subprocess.run([self.lmp_cmd, "-in", in_path],
                                  cwd=tmp_dir, capture_output=True, text=True, timeout=900)
            if proc.returncode != 0:
                raise RuntimeError(f"LAMMPS 最小化失败，返回码 {proc.returncode}\nSTDERR尾部:\n{(proc.stderr or '')[-1200:]}")

            # 读取 E 与 σ
            E = DirectLAMMPSEAMPotential._parse_energy(Eout_path)
            R_lmp = DirectLAMMPSEAMPotential._parse_positions_from_dump(dump_final, N)
            R_cart = (Q @ R_lmp.T).T
            if align_to_input:
                R_cart = _align_positions_to_reference(R_cart, positions, cell)
            sigma = self._parse_stress_bar_to_eVA3(Sout_path)
            return R_cart, float(E), sigma
        finally:
            if not self.keep_tmp_files:
                shutil.rmtree(tmp_dir, ignore_errors=True)

class DirectLAMMPSEAMself:
    """
    仅将 LAMMPS 用作 EAM 的能量/力计算器（run 0），支持任意晶胞（内部 QR -> triclinic）。
    返回 (energy_eV, forces[N,3])，forces 为原坐标系下。
    """
    def __init__(self, eam_file, lmp_cmd="lmp.exe", element="Al",
                 mass=26.981538, pair_style="eam/alloy",
                 keep_tmp_files=False):
        self.eam_file = str(eam_file)
        self.lmp_cmd = str(lmp_cmd)
        self.element = str(element)
        self.mass = float(mass)
        self.pair_style = str(pair_style)   # "eam" 或 "eam/alloy"
        self.keep_tmp_files = bool(keep_tmp_files)

    def _check_exec(self):
        try:
            subprocess.run([self.lmp_cmd, "-h"], capture_output=True, text=True, timeout=5)
        except Exception as e:
            raise RuntimeError(f"无法执行 LAMMPS 可执行文件: {self.lmp_cmd}") from e

    @staticmethod
    def _parse_energy(path_Eout):
        with open(path_Eout, "r") as f:
            return float(f.read().strip())

    @staticmethod
    def _parse_forces_from_dump(path_dump, N):
        forces = np.zeros((N, 3))
        with open(path_dump, "r") as f:
            for line in f:
                if line.startswith("ITEM") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                i = int(parts[0]) - 1
                fx, fy, fz = map(float, parts[5:8])
                forces[i] = [fx, fy, fz]
        return forces

    def energy_and_forces(self, positions, cell):
        """调用 LAMMPS (run 0) 计算能量与力，支持非正交晶胞"""
        self._check_exec()
        positions = np.asarray(positions, dtype=float)
        cell = np.asarray(cell, dtype=float)
        N = len(positions)

        tmpdir = tempfile.mkdtemp(prefix="eam_forcecalc_")
        try:
            pot_base = os.path.basename(self.eam_file)
            shutil.copy2(self.eam_file, os.path.join(tmpdir, pot_base))
            data_path = os.path.join(tmpdir, "data.in")
            dump_path = os.path.join(tmpdir, "dump_forces.txt")
            eout_path  = os.path.join(tmpdir, "energy.out")
            in_path    = os.path.join(tmpdir, "in.forcecalc")

            # 写 data：QR -> triclinic
            Q, R = DirectLAMMPSEAMPotential._write_data_atomic_triclinic(
                data_path, cell, positions, mass_amu=self.mass
            )

            # 写 LAMMPS 输入文件
            lines = [
                "units metal",
                "atom_style atomic",
                "boundary p p p",
                "box tilt large",
                f"read_data {data_path}",
                "",
                f"pair_style {self.pair_style}",
            ]
            if "alloy" in self.pair_style.lower():
                lines.append(f"pair_coeff * * {pot_base} {self.element}")
            else:
                lines.append(f"pair_coeff * * {pot_base}")
            lines += [
                f"mass 1 {self.mass:.6f}",
                "",
                "neighbor 2.0 bin",
                "neigh_modify delay 0 every 1 check yes",
                "thermo_style custom pe",
                "thermo 1",
                f"dump d1 all custom 1 {dump_path} id type x y z fx fy fz",
                'dump_modify d1 sort id format line "%d %d %.10g %.10g %.10g %.10g %.10g %.10g"',
                "run 0",
                "variable e equal pe",
                f'print "${{e}}" file {eout_path} screen no',
                "undump d1",
                ""
            ]
            with open(in_path, "w") as f:
                f.write("\n".join(lines))

            proc = subprocess.run([self.lmp_cmd, "-in", in_path],
                                  cwd=tmpdir, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"LAMMPS run0 失败，返回码 {proc.returncode}\n--- STDOUT ---\n{(proc.stdout or '')[-1500:]}\n"
                    f"--- STDERR ---\n{(proc.stderr or '')[-1500:]}"
                )

            energy = self._parse_energy(eout_path)
            forces_rot = self._parse_forces_from_dump(dump_path, N)
            # 旋回到原坐标系
            forces = (Q @ forces_rot.T).T
            return energy, forces

        finally:
            if not self.keep_tmp_files:
                shutil.rmtree(tmpdir, ignore_errors=True)

class DirectLAMMPSLCBOPself:
    def __init__(self, lcbop_file, lmp_cmd="lmp.exe", element="C",
                 mass=12.011, pair_style="lcbop",
                 keep_tmp_files=False):
        self.lcbop_file = str(lcbop_file)
        self.lmp_cmd = str(lmp_cmd)
        self.element = str(element)
        self.mass = float(mass)
        self.pair_style = str(pair_style)
        self.keep_tmp_files = bool(keep_tmp_files)

    def _check_exec(self):
        try:
            subprocess.run([self.lmp_cmd, "-h"],
                           capture_output=True, text=True, timeout=5)
        except Exception as e:
            raise RuntimeError(f"无法执行 LAMMPS 可执行文件: {self.lmp_cmd}") from e

    @staticmethod
    def _parse_energy(path_Eout):
        with open(path_Eout, "r") as f:
            return float(f.read().strip())

    @staticmethod
    def _parse_forces_from_dump(path_dump, N):
        forces = np.zeros((N, 3))
        with open(path_dump, "r") as f:
            for line in f:
                if line.startswith("ITEM") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                i = int(parts[0]) - 1
                fx, fy, fz = map(float, parts[5:8])
                forces[i] = [fx, fy, fz]
        return forces

    def energy_and_forces(self, positions, cell):
        """调用 LAMMPS (run 0) 计算能量与力，支持非正交晶胞"""
        self._check_exec()
        positions = np.asarray(positions)
        cell = np.asarray(cell)
        N = len(positions)

        tmpdir = tempfile.mkdtemp(prefix="lcbop_forcecalc_")
        try:
            pot_base = os.path.basename(self.lcbop_file)
            shutil.copy2(self.lcbop_file, os.path.join(tmpdir, pot_base))
            data_path = os.path.join(tmpdir, "data.in")
            dump_path = os.path.join(tmpdir, "dump_forces.txt")
            eout_path  = os.path.join(tmpdir, "energy.out")
            in_path    = os.path.join(tmpdir, "in.forcecalc")

            Q, R = DirectLAMMPSEAMPotential._write_data_atomic_triclinic(data_path, cell, positions,mass_amu=self.mass)

            # 写 LAMMPS 输入文件
            lines = [
                "units metal",
                "atom_style atomic",
                "boundary p p p",
                "box tilt large",  
                f"read_data {data_path}",
                "",
                f"pair_style {self.pair_style}",
                f"pair_coeff * * {pot_base} {self.element}",
                f"mass 1 {self.mass:.6f}",
                "",
                "neighbor 2.0 bin",
                "neigh_modify delay 0 every 1 check yes",
                "thermo_style custom pe",
                "thermo 1",
                f"dump d1 all custom 1 {dump_path} id type x y z fx fy fz",
                'dump_modify d1 sort id format line "%d %d %.10g %.10g %.10g %.10g %.10g %.10g"',
                "run 0",
                "variable e equal pe",
                f'print "${{e}}" file {eout_path} screen no',
                "undump d1",
                ""
            ]
            with open(in_path, "w") as f:
                f.write("\n".join(lines))

            # 运行 LAMMPS
            proc = subprocess.run([self.lmp_cmd, "-in", in_path],
                                  cwd=tmpdir, capture_output=True, text=True,timeout=120)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"LAMMPS run0 失败，返回码 {proc.returncode}\n--- STDOUT ---\n{(proc.stdout or '')[-1500:]}\n"
                    f"--- STDERR ---\n{(proc.stderr or '')[-1500:]}"
                )
            # 解析输出
            energy = self._parse_energy(eout_path)
            forces_rot = self._parse_forces_from_dump(dump_path, N)
            # 将力旋回到原坐标系：F = Q @ F'
            forces = (Q @ forces_rot.T).T
            return energy, forces

        finally:
            if not self.keep_tmp_files:
                shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":

    from opt_method import make_diamond,build_supercell,make_fcc

    eam_file = r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\Potentials\Al_zhou.eam.alloy"
    lcbop_file = r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\Potentials\C.lcbop"
    lmp_cmd  = r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\bin\lmp.exe"
#
    #a_Al = 4.20
    #lat = make_diamond(a_Al)
    #positions, cell, symbols = build_supercell(lat, 2, 2, 2)
    #pot_Al = DirectLAMMPSEAMPotential(eam_file=eam_file, lmp_cmd=lmp_cmd, element="Al",
    #                               mass=26.981538, pair_style="eam/alloy", keep_tmp_files=True)
    #R_relaxed, E_min = pot_Al.relax_positions_fixed_cell(positions, cell,
    #                                                  min_style="fire",
    #                                                  e_tol=1e-12, f_tol=1e-6,
    #                                                  maxiter=5000, maxeval=50000,
    #                                                  align_to_input=True)
    #print("relaxed Energy (eV) =", E_min)
    #print("relaxed positions (Å):\n", R_relaxed)

    #lat = make_diamond(3.567)
    #positions, cell, symbols = build_supercell(lat, 2, 2, 2)

    a_Al = 4.08
    lat = make_fcc(a_Al)
    positions, cell, symbols = build_supercell(lat, 2, 2, 2)

    pot = DirectLAMMPSEAMself(eam_file, lmp_cmd, keep_tmp_files=True)

    E, F = pot.energy_and_forces(positions, cell)
    print("Total potential energy (eV) =", E)
    print("Forces (eV/Å):\n", F)

    #from opt_method import make_diamond,build_supercell
    #a = 3.567  # diamond lattice constant
    #lat = make_diamond(a)
    #positions, cell, symbols = build_supercell(lat, 2, 2, 2)
    #pot_C = DirectLAMMPSLCBOPPotential(lcbop_file=lcbop_file, lmp_cmd=lmp_cmd,
    #                                   element="C", mass=12.011, keep_tmp_files=True)
    #R_relaxed, E_min = pot_C.relax_positions_fixed_cell(
    #    positions, cell, min_style="fire",
    #    e_tol=1e-12, f_tol=1e-6, maxiter=5000, maxeval=50000, align_to_input=True
    #)
    #print("relaxed Energy (eV) =", E_min)
    #print("relaxed positions (Å):\n", R_relaxed)