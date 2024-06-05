import numpy as np

c = 2.997e8
LMAX = 5000
l = np.arange(0, LMAX + 1)
dfac = l * (l + 1) / 2 / np.pi

ADD_PLANCK = False

# get DRAFT noise curves
ppfile = 'CMB-S4_DRAFT/products/20220726/s4wide_ilc_galaxy0_27-39-93-145-225-278_TT-EE_for7years.npy'
pp = np.load(ppfile, allow_pickle=1, encoding='latin1').item()['cl_residual']
print(pp)
print(len(pp['TT']))
draft_noise = dict(cl_TT=np.abs(pp['TT'][:-1]), cl_EE=np.abs(pp['EE'][:-1]),cl_TE=np.zeros_like(l), cl_BB=np.zeros_like(l), l=l)

print(len(draft_noise['cl_TT']))

# add planck white noise for TT only
arcmin_to_radian = np.pi / 60. / 180.
planck_noise_T = np.zeros_like(l, dtype=float)
planck_noise_P = np.zeros_like(l, dtype=float)
for s, theta in zip([45,149,137,65,43,66,200], [33,23,14,10,7,5,5]):
    s *= arcmin_to_radian
    theta *= arcmin_to_radian
    planck_noise_T += s ** -2 * np.exp(-l * (l + 1) * theta ** 2 / (8 * np.log(2)))
for s, theta in zip([450,103,81,134,406], [14,10,7,5,5]):
    s *= arcmin_to_radian
    theta *= arcmin_to_radian
    planck_noise_P += s ** -2 * np.exp(-l * (l + 1) * theta ** 2 / (8 * np.log(2)))
planck_noise_T[2501:] = 0
planck_noise_P[2501:] = 0
planck_noise_P[:31] = 0
draft_noise['cl_TT'][draft_noise['cl_TT'] == 0] = 1e100
draft_noise['cl_EE'][draft_noise['cl_EE'] == 0] = 1e100
if ADD_PLANCK:
    draft_noise['cl_TT'] = 1/(1/draft_noise['cl_TT'] + planck_noise_T)
    #draft_noise['cl_EE'] = 1 / (1 / draft_noise['cl_EE'] + planck_noise_P)
#draft_noise['cl_TT'][0] = 0
#draft_noise['cl_EE'][:2] = 0

for key in ('cl_TT', 'cl_EE', 'cl_TE', 'cl_BB'):
    draft_noise[key] = draft_noise[key][2:]
    try:
        draft_noise['d' + key[1:]] = draft_noise[key] * dfac
    except:
        print(key)

print(draft_noise)

import pickle
with open('/Users/yovel/Desktop/fisher_test/FisherLens/noise_draft.pkl', 'wb') as f:
    pickle.dump(draft_noise, f)