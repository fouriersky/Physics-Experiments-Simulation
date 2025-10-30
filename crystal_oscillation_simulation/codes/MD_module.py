"""
模块 2: 分子动力学(MD)模块

 - length: Angstrom (Å)
 - time: femtosecond (fs) -> LAMMPS units metal uses ps; we convert fs->ps
 - energy: eV
 - mass: amu (atomic mass unit)
 - stress returned in eV/Å^3 ; multiply by 160.21766208 -> GPa
"""
import numpy as np
import os, shutil, subprocess, tempfile
from potential import DirectLAMMPSEAMPotential

# Physical constants / conversion
kB_eV_per_K = 8.617333262145e-5   # eV / K
amu_to_eVfs2_A2 = 1.036427e-2     # 1 amu * (Å/fs)^2 = 1.036427e-2 eV
BAR_TO_EVA3 = 6.241509074e-7      # 1 bar = 6.2415e-7 eV/Å^3
EVA3_TO_GPA = 160.21766208        # eV/Å^3 -> GPa

def _apply_strain(cell, positions, eta, symmetric=True):
    """
    仿射应变：R' = F R, H' = F H, 其中 F = I + eta
    symmetric=True 且 i!=j 时请预先将 eta 设置为对称剪切（外部调用者负责）。
    """
    cell = np.asarray(cell, dtype=float)
    positions = np.asarray(positions, dtype=float)
    F = np.eye(3) + np.array(eta, dtype=float)
    cell_s = F @ cell
    pos_s = (F @ positions.T).T
    return cell_s, pos_s

def _resolve_lammps_interactions(backend, pot_basename):
    """
    从 DirectLAMMPSEAMPotential 或 DirectLAMMPSLCBOPPotential 提取 pair_style/coeff/mass。
    要求 backend 拥有:
      - pair_style (str)
      - element (str)
      - mass (float, amu)
    EAM:
      - pair_style 包含 'eam'；'* * filename Element'
    LCBOP:
      - pair_style 通常为 'lcbop' 或相应插件；同样 '* * filename Element'
    返回: list[str] (pair_style/pair_coeff), mass_amu
    """
    if not hasattr(backend, "pair_style"):
        raise RuntimeError("后端缺少 pair_style 属性")
    if not hasattr(backend, "element"):
        raise RuntimeError("后端缺少 element 属性")
    if not hasattr(backend, "mass"):
        raise RuntimeError("后端缺少 mass 属性")

    lines = [f"pair_style {backend.pair_style}"]
    ps = backend.pair_style.strip().lower()
    elem = str(backend.element)
    if "eam" in ps:
        # EAM/ALLOY 或 EAM：EAM alloy 需要元素映射
        if "alloy" in ps:
            lines.append(f"pair_coeff * * {pot_basename} {elem}")
        else:
            lines.append(f"pair_coeff * * {pot_basename}")
    else:
        # 一般多体 C 势（如 lcbop/rebo/airebo 等）：也多用 * * filename 元素
        lines.append(f"pair_coeff * * {pot_basename} {elem}")

    return lines, float(backend.mass)

def _get_backend_potfile(backend):
    """
    返回 (potfile_path, basename)。支持 backend.eam_file 或 backend.lcbop_file。
    """
    if hasattr(backend, "eam_file"):
        p = getattr(backend, "eam_file")
    elif hasattr(backend, "lcbop_file"):
        p = getattr(backend, "lcbop_file")
    else:
        raise RuntimeError("未找到势文件路径（期望 backend.eam_file 或 backend.lcbop_file）")
    base = os.path.basename(p)
    return p, base

def run_nvt_md_blocks(backend, positions, cell,
                      temperature_K=300.0,
                      dt_fs=1.0,
                      total_steps=20000,
                      block_steps=100,
                      equil_steps=5000,
                      strain_eta=None, symmetric=True,
                      seed=12345,
                      keep_tmp_files=False,
                      tdamp_ps=None):
    """
    使用 LAMMPS 做 NVT (Nosé-Hoover) 分块平均:
    - 先等温平衡 equil_steps
    - 然后采样 (total_steps - equil_steps)，每 block_steps 步输出一次块平均
    输出:
      dict {
        'time_ps': (nblocks,),  # 每个 block 末尾的时间
        'E_block_eV': (nblocks,),  # 块平均 Etot = <pe>+<ke>
        'T_block_K': (nblocks,),
        'sigma_block_eVA3': (nblocks,3,3),  # 张量，拉伸为正
        'sigma_block_GPa': (nblocks,3,3),
        'cell_used': (3,3),
        'tmp_dir': path or None
      }
    说明:
      - 支持任意 cell（内部 QR -> triclinic），单原子类型体系
      - 对剪切应力：请在 strain_eta 里设置对称剪切（eta_ij=eta_ji）
    """
    if not hasattr(backend, "lmp_cmd"):
        raise RuntimeError("后端缺少 lmp_cmd，可执行路径")
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)

    # 施加应变
    if strain_eta is not None:
        cell, positions = _apply_strain(cell, positions, strain_eta, symmetric=symmetric)

    # 基本参数
    dt_ps = float(dt_fs) * 1e-3
    total_steps = int(total_steps)
    block_steps = max(1, int(block_steps))
    equil_steps = max(0, int(equil_steps))
    sample_steps = max(0, total_steps - equil_steps)
    nblocks = sample_steps // block_steps

    if nblocks <= 0:
        raise ValueError("采样步数不足：确保 total_steps > equil_steps 且剩余步数至少一个 block_steps。")

    tmp_dir = tempfile.mkdtemp(prefix="md_nvt_")
    try:
        # 准备势文件与 data
        potfile_path, pot_basename = _get_backend_potfile(backend)
        shutil.copy2(potfile_path, os.path.join(tmp_dir, pot_basename))

        data_path = os.path.join(tmp_dir, "data.in")
        # 写入 triclinic data（任意晶胞），同时记录 Q 以便将应力旋回到原坐标系
        Q, _R = DirectLAMMPSEAMPotential._write_data_atomic_triclinic(
            data_path, cell, positions, mass_amu=getattr(backend, "mass", 12.0)
        )
        # LAMMPS 脚本
        in_eq = os.path.join(tmp_dir, "in.eq")
        in_prod = os.path.join(tmp_dir, "in.prod")
        avg_path = os.path.join(tmp_dir, "avg_block.txt")

        # 交互项
        pair_lines, mass_amu = _resolve_lammps_interactions(backend, pot_basename)

        # 公共头
        common_head = [
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            #"box tilt large",
            f"read_data {data_path}",
            "",
            "### interactions",
            *pair_lines,
            f"mass 1 {mass_amu:.6f}",
            "",
            "neighbor 2.0 bin",
            "neigh_modify delay 0 every 1 check yes",
            # Thermo 仅用于屏幕观察
            "thermo_style custom step temp pe ke etotal pxx pyy pzz pxy pxz pyz",
            f"thermo {block_steps}",
            "",
            "### computes for averaging",
            "compute cT all temp",
            "compute cPE all pe",
            "compute cKE all ke",
            "compute cP all pressure cT",  # bar
            "",
            "velocity all create ${T} ${SEED} mom yes rot yes dist gaussian",
        ]

        # NVT 参数
        Tdamp_ps = float(tdamp_ps) if tdamp_ps is not None else max(0.1, 100.0 * dt_ps)  # 经验：100*dt 或至少 0.1 ps
        eq_lines = [
            f"variable T equal {float(temperature_K):.16g}",
            f"variable SEED equal {int(seed)}",
            *common_head,
            f"fix nvtfix all nvt temp ${{T}} ${{T}} {Tdamp_ps:.6g}",
            f"timestep {dt_ps:.16g}",
            f"run {equil_steps}",
            "unfix nvtfix",
        ]
        with open(in_eq, "w", newline="\n") as f:
            f.write("\n".join(eq_lines) + "\n")

        prod_lines = [
            f"variable T equal {float(temperature_K):.16g}",
            f"variable SEED equal {int(seed)+1}",
            *common_head,
            f"fix nvtfix all nvt temp ${{T}} ${{T}} {Tdamp_ps:.6g}",
            f"timestep {dt_ps:.16g}",
            # 分块时间平均：每 block_steps 步输出一次，平均窗口=block_steps
            # 输出：cP[1..6], cT, cPE, cKE
            f"fix avg all ave/time 1 {block_steps} {block_steps} "
            f"c_cP[1] c_cP[2] c_cP[3] c_cP[4] c_cP[5] c_cP[6] c_cT c_cPE c_cKE "
            f"file {avg_path}",
            f"run {nblocks*block_steps}",
            "unfix avg",
            "unfix nvtfix",
        ]
        with open(in_prod, "w", newline="\n") as f:
            f.write("\n".join(prod_lines) + "\n")

        # 运行 LAMMPS
        def _run_in(inp):
            proc = subprocess.run([backend.lmp_cmd, "-in", inp],
                                  cwd=tmp_dir, capture_output=True, text=True, timeout=3600)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"LAMMPS 运行失败，返回码 {proc.returncode}\n--- STDOUT ---\n{(proc.stdout or '')[-1200:]}\n"
                    f"--- STDERR ---\n{(proc.stderr or '')[-1200:]}"
                )

        if equil_steps > 0:
            _run_in(in_eq)
        _run_in(in_prod)

        # 解析分块平均
        # 文件行：step cP1 cP2 cP3 cP4 cP5 cP6 cT cPE cKE
        steps, p1, p2, p3, p4, p5, p6, tK, pe, ke = [], [], [], [], [], [], [], [], [], []
        with open(avg_path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or s.lower().startswith("time"):
                    continue
                parts = s.split()
                if len(parts) < 10:
                    continue
                try:
                    step_i = int(float(parts[0]))
                except Exception:
                    # 有些版本不包含步数，则当作没步号
                    step_i = None
                    # 并将索引从 0 开始
                vals = list(map(float, parts[-9:]))  # 取最后9列
                _p1, _p2, _p3, _p4, _p5, _p6, _t, _pe, _ke = vals
                steps.append(step_i if step_i is not None else 0)
                p1.append(_p1); p2.append(_p2); p3.append(_p3)
                p4.append(_p4); p5.append(_p5); p6.append(_p6)
                tK.append(_t); pe.append(_pe); ke.append(_ke)

        if len(steps) != nblocks:
            # 容错：只取完整块
            nuse = min(len(steps), nblocks)
        else:
            nuse = nblocks

        steps = np.array(steps[:nuse], dtype=float)
        p1 = np.array(p1[:nuse], dtype=float)
        p2 = np.array(p2[:nuse], dtype=float)
        p3 = np.array(p3[:nuse], dtype=float)
        p4 = np.array(p4[:nuse], dtype=float)
        p5 = np.array(p5[:nuse], dtype=float)
        p6 = np.array(p6[:nuse], dtype=float)
        tK = np.array(tK[:nuse], dtype=float)
        pe = np.array(pe[:nuse], dtype=float)
        ke = np.array(ke[:nuse], dtype=float)

        # 压力(bar) -> 应力(eV/Å^3) 并取负号
        sxx = -p1 * BAR_TO_EVA3
        syy = -p2 * BAR_TO_EVA3
        szz = -p3 * BAR_TO_EVA3
        sxy = -p4 * BAR_TO_EVA3
        sxz = -p5 * BAR_TO_EVA3
        syz = -p6 * BAR_TO_EVA3
        # 在旋转系下的 σ（nblocks,3,3）
        sigma_rot = np.stack([
            np.stack([sxx, sxy, sxz], axis=-1),
            np.stack([sxy, syy, syz], axis=-1),
            np.stack([sxz, syz, szz], axis=-1),
        ], axis=-2)
        # 旋回到原坐标系：σ_lab = Q · σ_rot · Q^T
        QT = Q.T
        sigma = np.einsum("ia,nab,jb->nij", Q, sigma_rot, QT)

        Etot = pe + ke
        # 时间轴（每个块末尾时间，包含前面平衡）
        # 如果文件没有步号，用等间隔计算
        if np.all(steps == 0):
            idx = np.arange(1, nuse + 1, dtype=float)
            time_ps = (equil_steps + idx * block_steps) * dt_ps
        else:
            time_ps = steps * dt_ps

        out = dict(
            time_ps=time_ps,
            E_block_eV=Etot,
            T_block_K=tK,
            sigma_block_eVA3=sigma,
            sigma_block_GPa=sigma * EVA3_TO_GPA,
            cell_used=cell,
            Q=Q,
            tmp_dir=(tmp_dir if keep_tmp_files else None)
        )
        return out
    finally:
        if not keep_tmp_files:
            shutil.rmtree(tmp_dir, ignore_errors=True)

def run_npt_md_blocks(backend, positions, cell,
                      temperature_K=300.0,
                      pressure_bar=0.0,
                      dt_fs=1.0,
                      total_steps=20000,
                      block_steps=100,
                      equil_steps=5000,
                      strain_eta=None, symmetric=True,
                      seed=12345,
                      keep_tmp_files=False,
                      tdamp_ps=None,
                      pdamp_ps=None):
    """
    使用 LAMMPS 做 NPT (Nosé-Hoover) 分块平均（triclinic: tri 压力耦合）:
    - 先等温等压平衡 equil_steps
    - 然后采样 (total_steps - equil_steps)，每 block_steps 步输出一次块平均
    输出:
      dict {
        'time_ps': (nblocks,),
        'E_block_eV': (nblocks,),
        'T_block_K': (nblocks,),
        'sigma_block_eVA3': (nblocks,3,3),  # 拉伸为正
        'sigma_block_GPa': (nblocks,3,3),
        'L_block_A': (nblocks,3),           # [Lx, Ly, Lz]
        'L_avg_A': (3,),                    # 每个方向的平均盒长
        'cell_used': (3,3),
        'Q': (3,3),
        'tmp_dir': path or None
      }
    说明:
      - 支持任意 cell（内部 QR -> triclinic）
      - tri 压力耦合适用于 triclinic 盒；对角初始盒可同样使用 tri
    """
    if not hasattr(backend, "lmp_cmd"):
        raise RuntimeError("后端缺少 lmp_cmd，可执行路径")
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)

    # 施加应变（仿射）
    if strain_eta is not None:
        cell, positions = _apply_strain(cell, positions, strain_eta, symmetric=symmetric)

    # 基本参数
    dt_ps = float(dt_fs) * 1e-3
    total_steps = int(total_steps)
    block_steps = max(1, int(block_steps))
    equil_steps = max(0, int(equil_steps))
    sample_steps = max(0, total_steps - equil_steps)
    nblocks = sample_steps // block_steps
    if nblocks <= 0:
        raise ValueError("采样步数不足：确保 total_steps > equil_steps 且剩余步数至少一个 block_steps。")

    tmp_dir = tempfile.mkdtemp(prefix="md_npt_")
    try:
        # 准备势文件与 data
        potfile_path, pot_basename = _get_backend_potfile(backend)
        shutil.copy2(potfile_path, os.path.join(tmp_dir, pot_basename))

        data_path = os.path.join(tmp_dir, "data.in")
        # 写入 triclinic data（任意晶胞），并记录旋转矩阵 Q
        Q, _R = DirectLAMMPSEAMPotential._write_data_atomic_triclinic(
            data_path, cell, positions, mass_amu=getattr(backend, "mass", 12.0)
        )

        # LAMMPS 脚本
        in_eq = os.path.join(tmp_dir, "in.eq")
        in_prod = os.path.join(tmp_dir, "in.prod")
        avg_path = os.path.join(tmp_dir, "avg_block.txt")

        # pair_style/pair_coeff
        pair_lines, mass_amu = _resolve_lammps_interactions(backend, pot_basename)

        # 公共头（与 NVT 相同）
        common_head = [
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            f"read_data {data_path}",
            "",
            "### interactions",
            *pair_lines,
            f"mass 1 {mass_amu:.6f}",
            "",
            "neighbor 2.0 bin",
            "neigh_modify delay 0 every 1 check yes",
            "thermo_style custom step temp press pe ke etotal pxx pyy pzz pxy pxz pyz lx ly lz",
            f"thermo {block_steps}",
            "",
            "### computes for averaging",
            "compute cT all temp",
            "compute cPE all pe",
            "compute cKE all ke",
            "compute cP all pressure cT",  # bar
            "",
            "velocity all create ${T} ${SEED} mom yes rot yes dist gaussian",
        ]

        # 控制参数
        Tdamp_ps = float(tdamp_ps) if tdamp_ps is not None else max(0.1, 100.0 * dt_ps)
        Pdamp_ps = float(pdamp_ps) if pdamp_ps is not None else max(1.0, 1000.0 * dt_ps)  # 压力阻尼通常更大一些

        # 等温等压平衡阶段
        eq_lines = [
            f"variable T equal {float(temperature_K):.16g}",
            f"variable P equal {float(pressure_bar):.16g}",  # bar
            f"variable SEED equal {int(seed)}",
            *common_head,
            # triclinic 压力耦合：tri Px Py Pz Pxdamp Pydamp Pzdamp
            f"fix nptfix all npt temp ${{T}} ${{T}} {Tdamp_ps:.6g} tri ${{P}} ${{P}} ${{P}} {Pdamp_ps:.6g} {Pdamp_ps:.6g} {Pdamp_ps:.6g}",
            f"timestep {dt_ps:.16g}",
            f"run {equil_steps}",
            "unfix nptfix",
        ]
        with open(in_eq, "w", newline="\n") as f:
            f.write("\n".join(eq_lines) + "\n")

        # 产出阶段：分块平均 cP 与盒长（Lx,Ly,Lz）
        prod_lines = [
            f"variable T equal {float(temperature_K):.16g}",
            f"variable P equal {float(pressure_bar):.16g}",
            f"variable SEED equal {int(seed)+1}",
            *common_head,
            f"fix nptfix all npt temp ${{T}} ${{T}} {Tdamp_ps:.6g} tri ${{P}} ${{P}} ${{P}} {Pdamp_ps:.6g} {Pdamp_ps:.6g} {Pdamp_ps:.6g}",
            f"timestep {dt_ps:.16g}",
            # 将盒长关键字包为变量，供 ave/time 使用
            "variable Lx equal lx",
            "variable Ly equal ly",
            "variable Lz equal lz",
            # 块时间平均：Lx Ly Lz + 压力张量 + 温度/能量
            f"fix avg all ave/time 1 {block_steps} {block_steps} "
            f"v_Lx v_Ly v_Lz "
            f"c_cP[1] c_cP[2] c_cP[3] c_cP[4] c_cP[5] c_cP[6] "
            f"c_cT c_cPE c_cKE "
            f'file "{avg_path}"',
            f"run {nblocks*block_steps}",
            "unfix avg",
            "unfix nptfix",
        ]
        with open(in_prod, "w", newline="\n") as f:
            f.write("\n".join(prod_lines) + "\n")

        # 运行 LAMMPS
        def _run_in(inp):
            proc = subprocess.run([backend.lmp_cmd, "-in", inp],
                                  cwd=tmp_dir, capture_output=True, text=True, timeout=3600)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"LAMMPS 运行失败，返回码 {proc.returncode}\n--- STDOUT ---\n{(proc.stdout or '')[-1200:]}\n"
                    f"--- STDERR ---\n{(proc.stderr or '')[-1200:]}"
                )
        if equil_steps > 0:
            _run_in(in_eq)
        _run_in(in_prod)

        # 解析分块平均
        # 文件列：step Lx Ly Lz cP1..cP6 cT cPE cKE  => 共 1 + 3 + 6 + 3 = 13 列
        steps, Lx, Ly, Lz = [], [], [], []
        p1, p2, p3, p4, p5, p6 = [], [], [], [], [], []
        tK, pe, ke = [], [], []
        with open(avg_path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or s.lower().startswith("time"):
                    continue
                parts = s.split()
                if len(parts) < 13:
                    continue
                try:
                    step_i = int(float(parts[0]))
                except Exception:
                    step_i = None
                vals = list(map(float, parts[-12:]))  # 取最后12列：Lx Ly Lz p1..p6 T PE KE
                _Lx, _Ly, _Lz, _p1, _p2, _p3, _p4, _p5, _p6, _t, _pe, _ke = vals
                steps.append(step_i if step_i is not None else 0)
                Lx.append(_Lx); Ly.append(_Ly); Lz.append(_Lz)
                p1.append(_p1); p2.append(_p2); p3.append(_p3)
                p4.append(_p4); p5.append(_p5); p6.append(_p6)
                tK.append(_t);  pe.append(_pe); ke.append(_ke)

        nuse = min(len(steps), nblocks)
        steps = np.array(steps[:nuse], dtype=float)
        Lx = np.array(Lx[:nuse], dtype=float)
        Ly = np.array(Ly[:nuse], dtype=float)
        Lz = np.array(Lz[:nuse], dtype=float)
        p1 = np.array(p1[:nuse], dtype=float)
        p2 = np.array(p2[:nuse], dtype=float)
        p3 = np.array(p3[:nuse], dtype=float)
        p4 = np.array(p4[:nuse], dtype=float)
        p5 = np.array(p5[:nuse], dtype=float)
        p6 = np.array(p6[:nuse], dtype=float)
        tK = np.array(tK[:nuse], dtype=float)
        pe = np.array(pe[:nuse], dtype=float)
        ke = np.array(ke[:nuse], dtype=float)

        # 压力(bar) -> 应力(eV/Å^3), 拉伸为正
        sxx = -p1 * BAR_TO_EVA3
        syy = -p2 * BAR_TO_EVA3
        szz = -p3 * BAR_TO_EVA3
        sxy = -p4 * BAR_TO_EVA3
        sxz = -p5 * BAR_TO_EVA3
        syz = -p6 * BAR_TO_EVA3
        sigma_rot = np.stack([
            np.stack([sxx, sxy, sxz], axis=-1),
            np.stack([sxy, syy, syz], axis=-1),
            np.stack([sxz, syz, szz], axis=-1),
        ], axis=-2)
        # 旋回到原坐标系
        sigma = np.einsum("ia,nab,jb->nij", Q, sigma_rot, Q.T)

        Etot = pe + ke
        if np.all(steps == 0):
            idx = np.arange(1, nuse + 1, dtype=float)
            time_ps = (equil_steps + idx * block_steps) * dt_ps
        else:
            time_ps = steps * dt_ps

        out = dict(
            time_ps=time_ps,
            E_block_eV=Etot,
            T_block_K=tK,
            sigma_block_eVA3=sigma,
            sigma_block_GPa=sigma * EVA3_TO_GPA,
            L_block_A=np.stack([Lx, Ly, Lz], axis=-1),
            L_avg_A=np.array([Lx.mean(), Ly.mean(), Lz.mean()], dtype=float),
            cell_used=cell,
            Q=Q,
            tmp_dir=(tmp_dir if keep_tmp_files else None)
        )
        return out
    finally:
        if not keep_tmp_files:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _tensor_to_voigt_eps(eps_tensor, convention="engineering"):
    """
    将对称应变张量(3x3)转为 Voigt 向量:
      convention='tensor': [exx, eyy, ezz, eyz, exz, exy]
      convention='engineering': [exx, eyy, ezz, 2*eyz, 2*exz, 2*exy]
    """
    e = np.array(eps_tensor, dtype=float)
    if e.shape != (3,3):
        raise ValueError("eps_tensor 必须是 3x3")
    v = np.array([e[0,0], e[1,1], e[2,2], e[1,2], e[0,2], e[0,1]], dtype=float)
    if convention == "engineering":
        v[3:] *= 2.0
    return v

def _tensor_to_voigt_sigma(sig_tensor):
    """
    将对称应力张量(3x3)转为 Voigt 向量（剪切不乘 2）:
      [sxx, syy, szz, syz, sxz, sxy]
    单位保持与输入一致（eV/Å^3 或 GPa）。
    """
    s = np.array(sig_tensor, dtype=float)
    if s.shape != (3,3):
        raise ValueError("sig_tensor 必须是 3x3")
    return np.array([s[0,0], s[1,1], s[2,2], s[1,2], s[0,2], s[0,1]], dtype=float)

def solve_stiffness_least_squares(samples, convention="tensor", symmetrize=True):
    """
    用最小二乘解 6x6 刚度矩阵 C，使得 σ_vec ≈ C · ε_vec。
    参数:
      samples: list[dict]，每个元素包含
        {
          'eps_tensor': (3,3) numpy, 施加的应变张量（对称），
          'sigma_avg':  (3,3) numpy, 对应 MD 稳态/分块的平均应力张量
        }
      convention: 'tensor' 或 'engineering'（决定剪切分量是否乘2）
      symmetrize: 返回时做 (C + C^T)/2 的微小对称化，降噪
    返回:
      C: (6,6) numpy 数组（单位与 sigma 相同：若传 GPa 则 C 为 GPa）
    """
    if len(samples) < 6:
        raise ValueError("至少需要 6 组独立应变样本来拟合 6x6 刚度矩阵")
    E_rows = []
    S_rows = []
    for item in samples:
        eps_v = _tensor_to_voigt_eps(item['eps_tensor'], convention=convention)
        sig_v = _tensor_to_voigt_sigma(item['sigma_avg'])
        E_rows.append(eps_v)
        S_rows.append(sig_v)
    E = np.asarray(E_rows, dtype=float)   # (K,6)
    S = np.asarray(S_rows, dtype=float)   # (K,6)
    # 解 E · C^T ≈ S 亦可；这里直接解 E X ≈ S，X 即为 C
    C, _, _, _ = np.linalg.lstsq(E, S, rcond=None)
    if symmetrize:
        C = 0.5 * (C + C.T)
    return C

def solve_stiffness_from_pure_components(samples_by_comp, convention="tensor", symmetrize=True):
    """
    针对“每次仅施加一个非零应变分量”的设计，分别拟合 C 的每一列:
      输入:
        samples_by_comp: dict[int -> list[ (eps_scalar, sigma_avg_tensor) ]]
          键为 Voigt 索引 j ∈ {0..5}（按本文件的 Voigt 顺序）
          值为若干对 (ε_j, σ_avg)，用于线性拟合：σ ≈ C[:,j] * ε_j
      返回:
        C: (6,6)
    说明:
      - 若每列至少有两个不同的 ε_j（含正负），鲁棒性更好。
      - 若使用 engineering 约定，请确保传入的 ε_j 已乘以 2（剪切）。
    """
    C = np.zeros((6,6), dtype=float)
    for j, pairs in samples_by_comp.items():
        eps_list = []
        sig_list = []
        for eps_j, sig_tensor in pairs:
            eps_list.append(float(eps_j))
            sig_list.append(_tensor_to_voigt_sigma(sig_tensor))
        eps_arr = np.asarray(eps_list, dtype=float).reshape(-1, 1)  # (K,1)
        S = np.asarray(sig_list, dtype=float)                       # (K,6)
        # 解 eps_arr * c_col^T ≈ S -> 最小二乘 c_col^T
        # 即每个分量独立线性回归：σ_i ≈ c_i_j * ε_j
        col, _, _, _ = np.linalg.lstsq(eps_arr, S, rcond=None)      # (1,6)
        C[:, j] = col.ravel()
    if symmetrize:
        C = 0.5 * (C + C.T)
    return C

def _sigma_avg_from_res(res, drop_frac=0.5, use_GPa=True):
    """从 run_nvt_md_blocks 的结果取稳态块均值应力张量。"""
    sig = res['sigma_block_GPa'] if use_GPa else res['sigma_block_eVA3']
    i0 = int(drop_frac * len(sig))
    return sig[i0:].mean(axis=0)  # 3x3

def _voigt_sigma_from_tensor(sig_tensor):
    return _tensor_to_voigt_sigma(sig_tensor)

def build_C_by_central_difference(backend, pos0, cell0, temperature_K=300,
                                  dt_fs=1.0, total_steps=30000, block_steps=100,
                                  equil_steps=3000, strain_eps=2e-3, seed=12345,
                                  drop_frac=0.5,convention="engineering",
                                  eps_mags=None, repeats=1, same_seed_pm=True,
                                  tdamp_ps=None, enforce_cubic=False):
    """
    采用中心差分法构建 6x6 刚度矩阵:
      - 支持多个应变幅值 eps_mags=[ε,2ε,...]，每个做中心差分后线性回归/平均
      - 支持重复次数 repeats（独立随机种子）做平均
      - same_seed_pm=True 时，±ε 使用相同 seed（强烈推荐，显著降噪）
      - tdamp_ps 可显式设置热浴时间常数，保持温度更稳
      - enforce_cubic=True 时，将结果投影到立方对称
      - NVT ensemble
    返回:
      C_GPa: (6,6)
    """
    if eps_mags is None:
        eps_mags = [float(strain_eps)]
    C = np.zeros((6,6), dtype=float)
    for j in range(6):
        col_samples = []
        for m, em in enumerate(eps_mags):
            for r in range(repeats):
                seed_base = seed + 100*j + 10*m + r
                seed_plus  = seed_base
                seed_minus = seed_base if same_seed_pm else seed_base + 1
                # +em
                eta_p = eps_tensor_from_voigt(j, +em, convention=convention)
                res_p = run_nvt_md_blocks(
                    backend=backend, positions=pos0, cell=cell0,
                    temperature_K=temperature_K, dt_fs=dt_fs,
                    total_steps=total_steps, block_steps=block_steps,
                    equil_steps=equil_steps, strain_eta=eta_p, symmetric=True,
                    seed=seed_plus, keep_tmp_files=False, tdamp_ps=tdamp_ps
                )
                # -em
                eta_m = eps_tensor_from_voigt(j, -em, convention=convention)
                res_m = run_nvt_md_blocks(
                    backend=backend, positions=pos0, cell=cell0,
                    temperature_K=temperature_K, dt_fs=dt_fs,
                    total_steps=total_steps, block_steps=block_steps,
                    equil_steps=equil_steps, strain_eta=eta_m, symmetric=True,
                    seed=seed_minus, keep_tmp_files=False, tdamp_ps=tdamp_ps
                )
                sig_p = _voigt_sigma_from_tensor(_sigma_avg_from_res(res_p, drop_frac=drop_frac, use_GPa=True))
                sig_m = _voigt_sigma_from_tensor(_sigma_avg_from_res(res_m, drop_frac=drop_frac, use_GPa=True))
                slope = (sig_p - sig_m) / (2.0 * em)
                col_samples.append(slope)
        C[:, j] = np.mean(np.vstack(col_samples), axis=0)
    C = 0.5 * (C + C.T)
    if enforce_cubic:
        C = project_to_cubic(C)
    return C

def build_C_by_central_difference_NPT(backend, pos0, cell0, temperature_K=300,
                                  dt_fs=1.0, total_steps=30000, block_steps=100,
                                  equil_steps=3000, strain_eps=2e-3, seed=12345,
                                  drop_frac=0.5,convention="engineering",
                                  eps_mags=None, repeats=1, same_seed_pm=True,
                                  tdamp_ps=None, enforce_cubic=False):
    if eps_mags is None:
        eps_mags = [float(strain_eps)]
    C = np.zeros((6,6), dtype=float)
    for j in range(6):
        col_samples = []
        for m, em in enumerate(eps_mags):
            for r in range(repeats):
                seed_base = seed + 100*j + 10*m + r
                seed_plus  = seed_base
                seed_minus = seed_base if same_seed_pm else seed_base + 1
                # +em
                eta_p = eps_tensor_from_voigt(j, +em, convention=convention)
                res_p = run_npt_md_blocks(
                    backend=backend,
                    positions=pos0, cell=cell0,
                    temperature_K=temperature_K, 
                    dt_fs=dt_fs,
                    total_steps=total_steps, block_steps=block_steps,
                    equil_steps=equil_steps, strain_eta=eta_p, symmetric=True,
                    seed=seed_plus, keep_tmp_files=False, tdamp_ps=tdamp_ps, pdamp_ps=1.0, pressure_bar=1.0,
                )
                # -em
                eta_m = eps_tensor_from_voigt(j, -em, convention=convention)
                res_m = run_npt_md_blocks(
                    backend=backend,
                    positions=pos0, cell=cell0,
                    temperature_K=temperature_K, 
                    dt_fs=dt_fs,
                    total_steps=total_steps, block_steps=block_steps,
                    equil_steps=equil_steps, strain_eta=eta_m, symmetric=True,
                    seed=seed_minus, keep_tmp_files=False, tdamp_ps=tdamp_ps, pdamp_ps=1.0, pressure_bar=1.0,
                )
                sig_p = _voigt_sigma_from_tensor(_sigma_avg_from_res(res_p, drop_frac=drop_frac, use_GPa=True))
                sig_m = _voigt_sigma_from_tensor(_sigma_avg_from_res(res_m, drop_frac=drop_frac, use_GPa=True))
                slope = (sig_p - sig_m) / (2.0 * em)
                col_samples.append(slope)
        C[:, j] = np.mean(np.vstack(col_samples), axis=0)
    C = 0.5 * (C + C.T)
    if enforce_cubic:
        C = project_to_cubic(C)
    return C

def project_to_cubic(C):
    """
    将任意 6x6 C 投影到立方晶系（3 个独立常数 C11, C12, C44）。
    """
    C = np.asarray(C, dtype=float)
    C11 = (C[0,0] + C[1,1] + C[2,2]) / 3.0
    C12 = (C[0,1] + C[0,2] + C[1,2]) / 3.0
    C44 = (C[3,3] + C[4,4] + C[5,5]) / 3.0
    Cc = np.zeros((6,6), dtype=float)
    for i in range(3):
        Cc[i,i] = C11
    for i in range(3):
        for k in range(3):
            if i != k:
                Cc[i,k] = C12
    for i in range(3,6):
        Cc[i,i] = C44
    return Cc

# ---------------- 使用示例：将你已有的 MD 结果拼成样本并求 C ----------------
# 对 6 个独立应变分量分别做了 NVT，拿到了稳态块平均应力 <σ>，构造 samples:
def example_build_C_from_runs(md_results):
    """
    md_results: list[dict]，每个元素应包含：
      {
        'eps_tensor': 3x3 numpy (施加应变),
        'sigma_avg':  3x3 numpy (将 res['sigma_block_GPa'] 做后期平均得到),
      }
    返回:
      C_GPa: 6x6 numpy（GPa）
    """
    C = solve_stiffness_least_squares(md_results, convention="engineering", symmetrize=True)
    return C

def eps_tensor_from_voigt(j, eps, convention="engineering"):
    """
    生成对称应变张量(3x3)，Voigt 顺序: 0:xx, 1:yy, 2:zz, 3:yz, 4:xz, 5:xy
    剪切分量采用“tensor 约定”εij（非工程剪切），并设置对称：εij=εji=eps。
    """
    E = np.zeros((3,3), dtype=float)
    if j == 0:
        E[0,0] = eps
    elif j == 1:
        E[1,1] = eps
    elif j == 2:
        E[2,2] = eps
    elif j == 3:
        E[1,2] = E[2,1] = eps
        val = eps*0.5 if convention == "engineering" else eps
        E[1,2] = E[2,1] = val
    elif j == 4:
        E[0,2] = E[2,0] = eps
        val = eps*0.5 if convention == "engineering" else eps
        E[0,2] = E[2,0] = val
    elif j == 5:
        E[0,1] = E[1,0] = eps
        val = eps*0.5 if convention == "engineering" else eps
        E[0,1] = E[1,0] = val
    else:
        raise ValueError("Voigt 索引应为 0..5")
    return E

# ------------------------
# Example helper for running short NVT (do not run on import)
# ------------------------
if __name__ == "__main__":
    # 示例：对已构型的 Al 或 C 体系，在 300K 下做 20 ps NVT，block=0.5 ps
    from potential import DirectLAMMPSLCBOPPotential
    from opt_method import strain_stress_0K_pipeline,make_diamond
    nx, ny, nz = 2, 2, 2
    a_guess = 3.56
    
    pot = DirectLAMMPSLCBOPPotential(
        lcbop_file=r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\Potentials\C.lcbop",
        lmp_cmd =r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\bin\lmp.exe",
        element="C", pair_style="lcbop", keep_tmp_files=False
    )
    out = strain_stress_0K_pipeline(
        lattice_factory=make_diamond, a=a_guess, nx=nx, ny=ny, nz=nz, potential=pot,
        strain_eps=1e-3, optimize_a=True, verbose=True,
        relax_params=dict(f_tol=1e-6, maxit=2000),  
        method='external'
    )

    C_GPa = build_C_by_central_difference(
        backend=pot,
        pos0=out['pos0'], cell0=out['cell0'],
        temperature_K=300, dt_fs=1.0,
        total_steps=30000, block_steps=100, equil_steps=3000,
        strain_eps=1e-3, seed=12345, drop_frac=0.3,convention="engineering"
    )
    print("C (GPa) via central difference =\n", C_GPa)


    #print("pos0:",out['pos0'])
    #print("cell0:",out['cell0'])

    #res = run_nvt_md_blocks(
    #    backend=pot,                 # DirectLAMMPSEAMPotential 或 DirectLAMMPSLCBOPPotential
    #    positions=out['pos0'],
    #    cell=out['cell0'],
    #    temperature_K=300,
    #    dt_fs=1.0,
    #    total_steps=30000,           # 总步数
    #    block_steps=100,             # 每 0.1 ps 取一个 block 平均（若 dt=1 fs）
    #    equil_steps=3000,            # 前 5 ps 作为预热，不计入统计
    #    strain_eta=None,             # 或 np.array([[exx, exy,...]])
    #    symmetric=True,
    #    seed=12345,
    #    keep_tmp_files=False
    #)

    """
    samples = []
    runs=[]
    for j in range(6):
        eps_tensor_k = eps_tensor_from_voigt(j, strain_eps)
        # 传入仿射应变的“形变梯度”增量 η = ε（对角与剪切都直接用对称 εij）
        eta = eps_tensor_k.copy()
        res_k = run_nvt_md_blocks(
            backend=pot,
            positions=out['pos0'],
            cell=out['cell0'],
            temperature_K=300,
            dt_fs=1.0,
            total_steps=30000,
            block_steps=100,
            equil_steps=3000,
            strain_eta=eta,
            symmetric=True,      # 剪切保持对称
            seed=12345 + j,      # 变更随机种子
            keep_tmp_files=False
        )
        runs.append((eps_tensor_k, res_k))

    for eps_tensor_k, res_k in runs:  
        sigma_blocks = res_k['sigma_block_GPa']  # (nblocks,3,3)
        sigma_avg = sigma_blocks[int(0.5*len(sigma_blocks)):].mean(axis=0)  # 丢弃前50%做平均
        samples.append(dict(eps_tensor=eps_tensor_k, sigma_avg=sigma_avg))

    C_GPa=example_build_C_from_runs(samples)
    print("C (GPa) =\n", C_GPa)
    """

    """
    # 画能量随时间（每 block）
    t = res['time_ps']; E = res['E_block_eV']; T = res['T_block_K']
    sigma_GPa = res['sigma_block_GPa']   # (nblocks,3,3)

    # 可视化与保存
    import matplotlib.pyplot as plt
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "md_plots")
    os.makedirs(out_dir, exist_ok=True)

    # 1) 能量-时间
    plt.figure(figsize=(6,4))
    plt.plot(t, E, lw=1.5)
    plt.xlabel("time (ps)")
    plt.ylabel("Etot per block (eV)")
    plt.title("Block-averaged total energy")
    plt.tight_layout()
    fig_E_path = os.path.join(out_dir, "md_E_vs_time.png")
    plt.savefig(fig_E_path, dpi=150)

    # 2) 温度-时间
    plt.figure(figsize=(6,4))
    plt.plot(t, T, color="tab:orange", lw=1.5)
    plt.xlabel("time (ps)")
    plt.ylabel("Temperature per block (K)")
    plt.title("Block-averaged temperature")
    plt.tight_layout()
    fig_T_path = os.path.join(out_dir, "md_T_vs_time.png")
    plt.savefig(fig_T_path, dpi=150)

    # 3) 应力-时间（GPa）
    # 提取 6 个分量
    sxx = sigma_GPa[:,0,0]
    syy = sigma_GPa[:,1,1]
    szz = sigma_GPa[:,2,2]
    sxy = sigma_GPa[:,0,1]
    sxz = sigma_GPa[:,0,2]
    syz = sigma_GPa[:,1,2]
    plt.figure(figsize=(7.5,5))
    plt.plot(t, sxx, label="σ_xx", lw=1.3)
    plt.plot(t, syy, label="σ_yy", lw=1.3)
    plt.plot(t, szz, label="σ_zz", lw=1.3)
    plt.plot(t, sxy, label="σ_xy", lw=1.3)
    plt.plot(t, sxz, label="σ_xz", lw=1.3)
    plt.plot(t, syz, label="σ_yz", lw=1.3)
    plt.xlabel("time (ps)")
    plt.ylabel("Stress per block (GPa)")
    plt.title("Block-averaged stress components")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    fig_S_path = os.path.join(out_dir, "md_sigma_vs_time.png")
    plt.savefig(fig_S_path, dpi=150)

    # 4) 简单统计
    import numpy as np
    print("[MD blocks] averages over sampled blocks:")
    print(f"  <Etot> = {np.mean(E):.6f} eV,  std = {np.std(E):.6f} eV")
    print(f"  <T>    = {np.mean(T):.2f} K,   std = {np.std(T):.2f} K")
    sig_mean = np.mean(sigma_GPa, axis=0)
    print("  <sigma> (GPa) =")
    print(sig_mean)
    print("Saved figures:")
    print(" ", fig_E_path)
    print(" ", fig_T_path)
    print(" ", fig_S_path)
    """