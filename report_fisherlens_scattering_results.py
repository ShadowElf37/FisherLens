import numpy as np
import matplotlib.pyplot as plot

ANN = False
DECAY = False # not impl
SCATTER = True

c = 2.997e8
f_sky = 0.4
TAU_PRIOR = True

covs = []

for m in range(-5, 4):
    for n in range(0,1):
        fish = np.load(f'CLASS_delens/results/scatter_m{m}_n{n}.pkl', allow_pickle=True)['fisherGaussian']['delensed']
        if TAU_PRIOR:
            fish[4, :] = 0
            fish[:, 4] = 0
            fish[4, 4] = 1 / 0.0074 ** 2
        cov = np.linalg.inv(fish) / f_sky
        print(f'm = 10^{m} GeV\tn = {n}\tσ <=', np.sqrt(cov[-1, -1]) * 2, 'cm^2')
        covs.append(cov)
