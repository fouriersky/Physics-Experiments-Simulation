import matplotlib.pyplot as plt
T = [10, 100, 300, 500, 700, 900, 1100]

avg_ii = [1047.8362, 974.863661, 943.49290067, 908.36518767, 858.9074 , 851.91333,  847.81535157]
avg_44_66 = [688.1539, 655.55285333, 675.24419833, 678.63479333, 645.0538, 618.323655,  617.23387256]
avg_12_13_23 = [ 133.8066, 167.95022,  176.789342, 187.528847, 197.2592, 187.51766533,  204.63308739]

plt.figure(figsize=(8, 4.8))
plt.plot(T, avg_ii, marker='o', lw=2, label='C11')
plt.plot(T, avg_44_66, marker='s', lw=2, label='C44')
plt.plot(T, avg_12_13_23, marker='^', lw=2, label='C12')

plt.xlabel('Temperature (K)')
plt.ylabel('Cij (GPa)')
plt.ylim(0,1100)
plt.xlim(0,1200)
plt.title('Temperature dependence of elastic constants')
plt.grid(True, ls='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig('./C_averages_vs_T.png', dpi=300)
