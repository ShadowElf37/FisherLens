import numpy as np
import matplotlib.pyplot as plot

#nldl = np.loadtxt('lcdm_scatter_nl_dl_cl.dat')[:, 1]
lcdm = np.loadtxt('explanatory_class_delens_cl.dat')[:, 1]
p7 = np.loadtxt('explanatory_class_delensp7_cl.dat')[:, 1]
p8 = np.loadtxt('explanatory_class_delensp8_cl.dat')[:, 1]
p9 = np.loadtxt('explanatory_class_delensp9_cl.dat')[:, 1]
p10 = np.loadtxt('explanatory_class_delensp10_cl.dat')[:, 1]

l = np.arange(2, len(lcdm)+2, 1)

#print(nldl)
#print(dl)
print(l)

plot.loglog(l, -(p7-lcdm)/lcdm, label='Step Size $10^{-7}$')
plot.loglog(l, -(p8-lcdm)/lcdm, label='Step Size $10^{-8}$')
plot.loglog(l, -(p9-lcdm)/lcdm, label='Step Size $10^{-9}$')
plot.loglog(l, -(p10-lcdm)/lcdm, label='Step Size $10^{-10}$')
plot.xlim([10,l[-1]])
plot.title('Change in $C_\ell$ by $p_{ann}$ Step Size in $m^3/s/kg$')
plot.xlabel('$\ell$')
plot.ylabel('$(LCDM-C_\ell)/LCDM$')
plot.legend()
plot.show()