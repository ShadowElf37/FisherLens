import subprocess as sp
import numpy as np
import matplotlib.pyplot as plot

step_table = {
    (-6, 0): 27,
    (-5, 0): 27,
    (-4, 0): 27,
    (-3, 0): 26,
    (-2, 0): 26,
    (-1, 0): 26,
    (0, 0): 24,
    (1, 0): 23,
    (2, 0): 22,
    (3, 0): 21,

    (-6, 2): 25,
    (-5, 2): 24,
    (-4, 2): 23,
    (-3, 2): 22,
    (-2, 2): 21,
    (-1, 2): 20,
    (0, 2): 19,
    (1, 2): 18,
    (2, 2): 17,
    (3, 2): 16,

    (-6, 4): 24,
    (-5, 4): 22,
    (-4, 4): 20,
    (-3, 4): 19,
    (-2, 4): 17,
    (-1, 4): 16,
    (0, 4): 16,
    (1, 4): 14,
    (2, 4): 13,
    (3, 4): 12,
}

f_sky = 0.4
TAU_PRIOR = True

mode = 2
constraints = []

if mode == 1:
    for ss_exp in range(-15, -5):
        try:
            cmd = ['python', 'fisherGenerateDataClass_example.py', str(10**ss_exp)]
            print(' '.join(cmd))
            proc = sp.Popen(cmd)
            proc.wait()
        except Exception as e:
            print(e)
        #print(f'{m} joined!')
elif mode == 2:
    for ss_exp in range(-15, -5):
        f = f'ann_p{str(abs(float(ss_exp)))}'
        try:
            print(f'{f}.pkl')
            fish = np.load(f'CLASS_delens/results/{f}.pkl', allow_pickle=True)['fisherGaussian']['delensed']
            if TAU_PRIOR:
                fish[4, :] = 0
                fish[:, 4] = 0
                fish[4, 4] = 1 / 0.0074 ** 2
            cov = np.linalg.inv(fish) / f_sky
            constraints.append(np.sqrt(cov[-1, -1]) * 2)
        except:
            print(f'Skipping 10^{ss_exp}')
    print(constraints)
    plot.loglog(10.**np.array(list(range(-15, -5)))/9e16, constraints)
    plot.xlabel('Step Size in Derivative')
    plot.ylabel('Constraint from Forecast')
    plot.axhline(y=1e-6/9e16, color='r', linestyle='--')
    plot.legend(['CMB-S4 (Fisher)', 'Planck'])
    plot.show()
