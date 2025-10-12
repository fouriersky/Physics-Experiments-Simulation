import subprocess, os
cmd = r"F:\lammps\LAMMPS 64-bit 22Jul2025 with Python\bin\lmp.exe"
print(os.path.isfile(cmd), os.access(cmd, os.X_OK))
print(subprocess.run([cmd, "-h"], capture_output=True, text=True))

