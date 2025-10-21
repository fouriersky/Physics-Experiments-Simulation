import numpy as np
import matplotlib.pyplot as plt

w0 = 2

def Re(w):
    value = 1+ ((w0**2-w**2)-1/3)/(((w0**2-w**2)-1/3)**2+w**2)
    return value

def Im(w):
    value = w/(((w0**2-w**2)-1/3)**2+w**2)
    return value

w = np.linspace(0,10,400)

fig ,axs =plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
y_re = Re(w)
y_im = Im(w)

# 绘制 Re
axs[0].plot(w, y_re, color='C0')
axs[0].axhline(1,ls='--')
axs[0].set_title('Real part ')
axs[0].set_xlabel(r'$\omega$')
axs[0].set_ylabel(r'Re[$\epsilon$]')
axs[0].grid(True, ls='--', alpha=0.6)

# 绘制 Im
axs[1].plot(w, y_im, color='C1')
axs[1].set_title('Imaginary part')
axs[1].set_xlabel(r'$\omega$')
axs[1].set_ylabel(r'Im[$\epsilon$]')
axs[1].grid(True, ls='--', alpha=0.6)

plt.savefig('./e-w.png',dpi=300)
plt.show()



