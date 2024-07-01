import numpy as np
import matplotlib.pyplot as plot

ANN = False
DECAY = False # not impl
SCATTER = True

c = 2.997e8
f_sky = 0.4
TAU_PRIOR = True

if SCATTER:
    data = np.load('CLASS_delens/results/step-25/scatter_m2_n0.pkl', allow_pickle=True)
elif ANN:
    data = np.load('CLASS_delens/results/ann.pkl', allow_pickle=True)
elif DECAY:
    data = np.load('CLASS_delens/results/decay.pkl', allow_pickle=True)
else:
    raise Exception('bro?')
print(data['fisherGaussian'])
fish = data['fisherGaussian']['delensed']

if TAU_PRIOR:
    fish[4, :] = 0
    fish[:, 4] = 0
    fish[4,4] = 1/0.0074**2
cov = np.linalg.inv(fish)/f_sky
#print(np.sqrt(cov[4,4]))

if ANN:
    # fix pann factors of c
    units_conversion = 9.e16  # 5.61e20
    cov[-1, :] *= units_conversion#c ** 2
    cov[:, -1] *= units_conversion#c ** 2

    print('pann 95% =', np.sqrt(cov[-1,-1])*2)
elif SCATTER:
    #print('m 95% = 1 ±', np.sqrt(cov[-2,-2])*2, 'GeV')
    print('σ 95% =', np.sqrt(cov[-1, -1]) * 2, 'cm^2')
elif DECAY:
    #print('m 95% = 1 ±', np.sqrt(cov[-2,-2])*2, 'GeV')
    print('Γ 95% =', np.sqrt(cov[-1, -1]) * 2, 'cm^2')

#print(cov.tolist())
print(np.sqrt(np.diagonal(cov)))

# graph
import plot_triangle

if ANN:
    SCALES = [0, 0, 9, 0, 0, 0, 7]
    FID = np.array([0.1197, 0.0222, 2.196e-9, 0.9655, 0.054, 0.010409*100, 0])
    LABELS = [r'$\Omega_{DM} h^2$', r'$\Omega_b h^2$', r'$10^{9}A_s$', r'$n_s$', r'$\tau_{reio}$', r'$10^{2}\theta_s$',r'$p_{\mathrm{ann}}$ ($10^{-7}\mathrm{m}^3\mathrm{s}^{-1}\mathrm{kg}^{-1}$)']
elif SCATTER:
    SCALES = [0, 0, 9, 0, 0, 0, 26]
    FID = np.array([0.1197, 0.0222, 2.196e-9, 0.9655, 0.054, 0.010409*100, 0])
    LABELS = [r'$\Omega_{DM} h^2$', r'$\Omega_b h^2$', r'$10^{9}A_s$', r'$n_s$', r'$\tau_{reio}$', r'$10^{2}\theta_s$', r'$\sigma\ \ (10^{-26}\mathrm{cm}^2)$'] # r'$\log_{10}(m_\chi)$',
elif DECAY:
    SCALES = [0, 0, 9, 0, 0, 0, 28]
    FID = np.array([0.1197, 0.0222, 2.196e-9, 0.9655, 0.054, 0.010409*100, 0])
    LABELS = [r'$\Omega_{DM} h^2$', r'$\Omega_b h^2$', r'$10^{9}A_s$', r'$n_s$', r'$\tau_{reio}$', r'$10^{2}\theta_s$', r'$\Gamma\ \ (10^{-28}\mathrm{s}^{-1})$'] # r'$\log_{10}(m_\chi)$',

plot_triangle.ONE_SIDED.append(LABELS[-1])

plot_triangle.triplot(LABELS, SCALES, FID, cov)
#plot.gcf().set_dpi(100)
plot.gcf().set_size_inches(12, 9)
plot.gcf().savefig('figure.png')

plot.show()