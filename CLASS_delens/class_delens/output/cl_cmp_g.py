import numpy as np
import matplotlib.pyplot as plot

import subprocess as sp
import shlex

#sp.run(shlex.split('wsl ./../'))

#nldl = np.loadtxt('lcdm_scatter_nl_dl_cl.dat')[:, 1]
lcdm = np.loadtxt('explanatory_class_delens_cl.dat')[:, 1]
g28 = np.loadtxt('gamma_28_ecdl_cl.dat')[:, 1]
p7 = np.loadtxt('explanatory_class_delensp7_cl.dat')[:, 1]

l = np.arange(2, len(lcdm)+2, 1)

#print(nldl)
#print(dl)
print(l)

plot.semilogx(l, lcdm, label='$\Lambda$CDM')
plot.semilogx(l, g28, label='$\Gamma = 10^{-24}$')
plot.semilogx(l, p7, label='$p_{ann} = 10^{-7}$')
plot.xlim([0, l[-1]])
#plot.title('Change in $C_\ell$ by $p_{ann}$ Step Size in $m^3/s/kg$')
plot.xlabel('$\ell$')
plot.ylabel('$\ell(\ell+1)C^{TT}_\ell$ [K$^2$]')
plot.legend()
plot.show()