import numpy as np
from generate_sigma_for_paper import step_table, STEP_TABLE_OFFSET
from classWrapTools import start_class_subproc

l_range = np.arange(2, 3001, 1)

m = -6
n = 0

mi = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3].index(m)
ni = [0,2,4].index(n)
sigma_table = [[1.836214287890062e-27, 2.0518390683079987e-26, 3.1112206655907685e-26], [3.2010521914769854e-27, 2.806166040585091e-25, 1.7774326767821104e-23], [3.66129331573554e-27, 3.1693518560830103e-24, 1.5912332246267115e-21], [7.580515107049344e-27, 3.7152765357903646e-23, 4.1822914003989664e-20], [8.369979060016429e-27, 4.245646772447925e-22, 4.498792369870086e-18], [8.670939028540317e-27, 4.8626892442488664e-21, 1.3491303574151182e-16], [1.2228485748278109e-26, 5.0860873379914125e-20, 5.130687432907451e-16], [7.012118140872936e-26, 5.482189222055548e-19, 2.5750864610618834e-14], [6.506048589094239e-25, 5.599551426166845e-18, 2.433412001132613e-13], [6.451914578768011e-24, 5.614395849537843e-17, 2.7884056915156963e-12]]
sigma_95pct =  sigma_table[mi][ni] #8.4e-24#

sigma_ini = step_table[(m, n)]+STEP_TABLE_OFFSET

pull_column = 1

cl_LCDM = np.loadtxt("CLASS_delens/class_delens/output/lcdm_scatter_nl_dl_cl_lensed.dat").transpose()[pull_column][:2999]

ini_fp = 'CLASS_delens/data/compare_residuals_buffer/scat_resid.ini'
scat_ini_old = open(ini_fp).read().split('\n')
scat_ini_new = []
for line in scat_ini_old:
    if line.startswith('sigma_dmeff'):
        scat_ini_new.append(f'sigma_dmeff = {sigma_95pct}')
    elif line.startswith('log10m_dmeff'):
        scat_ini_new.append(f'log10m_dmeff = {m}')
    elif line.startswith('npow_dmeff'):
        scat_ini_new.append(f'npow_dmeff = {n}')
    else:
        scat_ini_new.append(line)
open(ini_fp, 'w').write('\n'.join(scat_ini_new))

scat_proc = start_class_subproc(ini_fp, cwd='CLASS_delens/class_delens/')
cl_scat = np.loadtxt("CLASS_delens/data/compare_residuals_buffer/scat_resid_cl_lensed.dat").transpose()[pull_column]

cmbs4_nl = np.load('noise_draft.pkl', allow_pickle=True)
print(cmbs4_nl)

import matplotlib.pyplot as plot

#overlay = plot.imread("overlay.png")

fig, ax = plot.subplots()

#ax.imshow(overlay, extent=[2, 3000, -3, 3], aspect='auto')
#ax.set_axis_off()

#ax2 = plot.axes(ax.get_position(True))
#ax2.set_frame_on(False)
ax.plot(l_range, 100*(cl_LCDM-cl_scat[:2999])/cl_LCDM, label="$2\sigma$ signal for CMB-S4",linewidth=3.0, color='#08f')
ax.set_xscale('log')
ax.set_ylim(-3, 3)
ax.set_xlim([2, 5000])
ax.set_ylabel(r'$100 \times (C_\ell^{\Lambda CDM} - C_\ell)/C_\ell^{\Lambda CDM}$')
ax.set_xlabel(r'$\ell$')

L = np.arange(2, 4999, 1)
ax.fill_between(L, -100*2*cmbs4_nl['cl_TT'][2:5001], 100*2*cmbs4_nl['cl_TT'][2:5001], color='#aaa')

plot.legend(loc='upper left')
plot.title(f'Power Spectrum Residual for $m=10^{{{m}}}$ GeV and $n={n}$')
plot.show()
