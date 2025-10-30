import os, sys
import numpy as np
from typing import Tuple

# 先配置 LAMMPS Python 封装环境（Windows 必须在 import 前设置 DLL 路径）
BASE = r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python"
try:
    os.add_dll_directory(os.path.join(BASE, "bin"))
except Exception:
    pass
py_dir = os.path.join(BASE, "Python")
if py_dir not in sys.path:
    sys.path.append(py_dir)

from lammps import lammps

# --------- 工具：把任意 cell 转为 LAMMPS triclinic 形参，并返回旋转矩阵 ----------
def _qr_upper_tri(cell: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H = np.asarray(cell, dtype=float)
    Q, R = np.linalg.qr(H)
    for i in range(3):
        if R[i, i] < 0:
            R[i, :] *= -1.0
            Q[:, i] *= -1.0
    return Q, R  # H = Q @ R, R 为上三角

def _lmp_prism_from_R(R: np.ndarray):
    # R 上三角 => 直接映射为 LAMMPS triclinic 盒参数
    lx = float(R[0, 0]); ly = float(R[1, 1]); lz = float(R[2, 2])
    xy = float(R[0, 1]); xz = float(R[0, 2]); yz = float(R[1, 2])
    return lx, ly, lz, xy, xz, yz

# --------- 基类：嵌入式 LAMMPS 持久计算器（固定盒） ----------
class _LAMMPSEmbeddedCalculatorBase:
    def __init__(self, pair_style: str, pair_coeff: str, mass_amu: float,
                 element: str, potfile: str):
        self.pair_style = str(pair_style)
        self.pair_coeff = str(pair_coeff)     # 完整的 pair_coeff 行，如 'pair_coeff * * file Element'
        self.mass_amu = float(mass_amu)
        self.element = str(element)
        self.potfile = str(potfile)

        self.lmp = lammps()                   # 持久实例
        self.N = None
        self.Q = None
        self.cell = None
        self._setup_done = False

    def setup(self, positions: np.ndarray, cell: np.ndarray):
        positions = np.asarray(positions, dtype=float)
        cell = np.asarray(cell, dtype=float)
        self.N = len(positions)
        self.cell = cell.copy()

        # 旋到上三角坐标系
        Q, R = _qr_upper_tri(cell)
        self.Q = Q
        pos_rot = (Q.T @ positions.T).T
        lx, ly, lz, xy, xz, yz = _lmp_prism_from_R(R)

        lmp = self.lmp
        # 基本设置
        lmp.command("clear")
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command("boundary p p p")
        lmp.command("box tilt large")

        # 建立 triclinic 盒
        # region prism xlo xhi ylo yhi zlo zhi xy xz yz units box
        lmp.command(f"region box prism 0 {lx:.16g} 0 {ly:.16g} 0 {lz:.16g} {xy:.16g} {xz:.16g} {yz:.16g} units box")
        lmp.command("create_box 1 box")
        lmp.command(f"mass 1 {self.mass_amu:.6f}")

        # 创建 N 个原子（先占位，立即用 scatter 覆盖坐标）
        # 为避免 N 条 create_atoms 命令的开销，使用 lattice none + create_atoms single 循环一次即可；
        # 但对大系统，循环也只在 setup 阶段执行一次，代价可接受。
        for i in range(self.N):
            lmp.command(f"create_atoms 1 single 0 0 0")

        # 设置相互作用
        lmp.command(f"pair_style {self.pair_style}")
        lmp.command(self.pair_coeff)
        lmp.command("neighbor 2.0 bin")
        lmp.command("neigh_modify delay 0 every 1 check yes")
        lmp.command("thermo_style custom pe")
        lmp.command("thermo 1")

        # 将坐标散布到 LAMMPS（旋转系下）
        x = np.asarray(pos_rot, dtype=float).ravel()
        lmp.scatter_atoms("x", 1, 3, x)

        # 预构建邻居表，验证能量
        lmp.command("run 0 post no")
        self._setup_done = True

    def compute(self, positions: np.ndarray, cell: np.ndarray = None):
        if not self._setup_done:
            raise RuntimeError("calculator 未 setup()")

        if cell is not None:
            # 当前实现要求固定盒；若传入 cell 改变，抛错或重建
            if not np.allclose(cell, self.cell, atol=1e-10):
                raise ValueError("当前 calculator 固定晶胞；传入的 cell 已变化，请重建 calculator。")

        # 旋转坐标到 LAMMPS 旋转系
        pos = np.asarray(positions, dtype=float)
        pos_rot = (self.Q.T @ pos.T).T
        x = np.asarray(pos_rot, dtype=float).ravel()

        # 更新坐标并计算
        lmp = self.lmp
        lmp.scatter_atoms("x", 1, 3, x)
        lmp.command("run 0 post no")

        # 提取能量与力（在旋转系）
        E = float(lmp.get_thermo("pe"))
        f_rot = np.array(lmp.gather_atoms("f", 1, 3), dtype=float).reshape(self.N, 3)

        # 旋回到原坐标系：F = Q @ F'
        F = (self.Q @ f_rot.T).T
        return E, F

    def close(self):
        try:
            self.lmp.close()
        except Exception:
            pass

# --------- 具体计算器：LCBOP / EAM ----------
class LAMMPSLCBOPCalculator(_LAMMPSEmbeddedCalculatorBase):
    def __init__(self, lcbop_file: str, element: str = "C"):
        pair_style = "lcbop"
        pair_coeff = f"pair_coeff * * {os.path.abspath(lcbop_file)} {element}"
        super().__init__(pair_style, pair_coeff, mass_amu=12.011, element=element, potfile=lcbop_file)

class LAMMPSEAMCalculator(_LAMMPSEmbeddedCalculatorBase):
    def __init__(self, eam_file: str, element: str = "Al", pair_style: str = "eam/alloy", mass_amu: float = 26.981538):
        ps = pair_style.strip().lower()
        if "alloy" in ps:
            pair_coeff = f"pair_coeff * * {os.path.abspath(eam_file)} {element}"
        else:
            pair_coeff = f"pair_coeff * * {os.path.abspath(eam_file)}"
        super().__init__(pair_style, pair_coeff, mass_amu=mass_amu, element=element, potfile=eam_file)

# --------- 使用示例（请根据需要调用） ----------
if __name__ == "__main__":
    from opt_method import make_fcc, make_diamond, build_supercell

    # 示例：EAM / Al
    a = 4.05
    lat = make_fcc(a)
    R, H, syms = build_supercell(lat, 2, 2, 2)
    calc = LAMMPSEAMCalculator(r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\Potentials\Al_zhou.eam.alloy",
                               element="Al", pair_style="eam/alloy", mass_amu=26.981538)
    calc.setup(R, H)
    E, F = calc.compute(R)  # 固定盒
    print("E(eV)=", E, " | |F|max=", np.abs(F).max())
    calc.close()

