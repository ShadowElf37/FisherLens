import numpy as np
import matplotlib.pyplot as plot

fsky = 0.4

files = (
    ('CLASS_delens/results/step-25/scatter_m0_n0.pkl', 'sigma_dmeff', r'\sigma_0', 'Scattering'),
    ('CLASS_delens/results/ann.pkl', 'pann', r'p_{ann}', 'Scattering'),
    (r'CLASS_delens/results/decay_g26.0.pkl', 'DM_decay_Gamma', r'\Gamma', 'Scattering'),
    #(r'C:\Users\yovel\Desktop\FisherLensFresh\CLASS_delens\results\fisher_8p_nodelensg.pkl', 'DM_decay_Gamma', r'\Gamma', 'Scattering')
)

tickfont = {'family': 'sans serif',
                'weight': 'normal',
                'size': 18}
plot.rc('font', **tickfont)

f, ax = plot.subplots(len(files), 1, sharex=True)
plot.subplots_adjust(hspace=0.0)
f.set_size_inches(14.0, 6.0*len(files))
#f.suptitle('Fisher Information per $\ell$, by Channel Combination')

l = np.arange(2, 4501)
for i, (file, interesting_parameter, param, title) in enumerate(files):
    data = np.load(file, allow_pickle=True)

    try:
        cl = data['powersFid']['delensed'].copy()
        nl = data['cmbNoiseSpectra'].copy()
        dcl = data['paramDerivs'][interesting_parameter]['delensed']

        cl['cl_dd'] = data['powersFid']['lensing']['cl_dd']
        for key in nl.keys():
            n = nl[key][:len(cl[key])]
            cl[key] += n
        cl['cl_phiphi'] = cl['cl_dd'] / (l*(l+1))
        dcl['cl_phiphi'] = data['paramDerivs'][interesting_parameter]['lensing']['cl_dd'] / (l*(l+1))
    except KeyError:
        #print(data['powersFid'].keys())
        cl = data['powersFid'][0]['delensed'].copy()
        nl = data['cmbNoiseSpectra'].copy()
        dcl = data['paramDerivs'][0][interesting_parameter]['delensed']

        for key in nl.keys():
            n = nl[key][:len(cl[key])]
            cl[key] += n
        cl['cl_dd'] = data['powersFid'][0]['lensing']['cl_dd']
        cl['cl_phiphi'] = cl['cl_dd'] / (l * (l + 1))
        dcl['cl_phiphi'] = data['paramDerivs'][0][interesting_parameter]['lensing']['cl_dd'] / (l * (l + 1))

    Fxx_TT = (l+0.5)*fsky*(1/cl['cl_TT'])*dcl['cl_TT']*(1/cl['cl_TT'])*dcl['cl_TT']

    Fxx_TT_TE_EE = np.zeros_like(Fxx_TT)
    for ell in l:
        #print(ell)
        ell -= 2
        C = np.matrix([
            [cl['cl_TT'][ell], cl['cl_TE'][ell]],
            [cl['cl_TE'][ell], cl['cl_EE'][ell]],
        ])
        Ci = np.linalg.inv(C)
        dC = np.matrix([
            [dcl['cl_TT'][ell], dcl['cl_TE'][ell]],
            [dcl['cl_TE'][ell], dcl['cl_EE'][ell]],
        ])
        ell += 2
        Fxx_TT_TE_EE[ell-2] = (ell+0.5)*fsky*np.trace(Ci*dC*Ci*dC)

    Fxx_TT_TE_EE_PP = np.zeros_like(Fxx_TT)
    for ell in l:
        #print(ell)
        ell -= 2
        C = np.matrix([
            [cl['cl_TT'][ell], cl['cl_TE'][ell], 0],
            [cl['cl_TE'][ell], cl['cl_EE'][ell], 0],
            [0, 0, cl['cl_phiphi'][ell]]
        ])
        Ci = np.linalg.inv(C)
        dC = np.matrix([
            [dcl['cl_TT'][ell], dcl['cl_TE'][ell], 0],
            [dcl['cl_TE'][ell], dcl['cl_EE'][ell], 0],
            [0, 0, dcl['cl_phiphi'][ell]]
        ])
        ell += 2
        Fxx_TT_TE_EE_PP[ell-2] = (ell+0.5)*fsky*np.trace(Ci*dC*Ci*dC)



    Fxx_TE = (l+0.5)*fsky*(1/cl['cl_TE'])*dcl['cl_TE']*(1/cl['cl_TE'])*dcl['cl_TE']
    Fxx_EE = (l+0.5)*fsky*(1/cl['cl_EE'])*dcl['cl_EE']*(1/cl['cl_EE'])*dcl['cl_EE']
    Fxx_PP = (l+0.5)*fsky*(1/cl['cl_phiphi'])*dcl['cl_phiphi']*(1/cl['cl_phiphi'])*dcl['cl_phiphi']

    w = 5
    lmin = 100
    ax[i].plot(l[(lmin-2):], Fxx_TT[(lmin-2):], label='TT', linewidth=w)
    ax[i].plot(l[(lmin-2):], Fxx_TT_TE_EE[(lmin-2):], label='TT+TE+EE', linewidth=w)
    ax[i].plot(l[(lmin-2):], Fxx_TT_TE_EE_PP[(lmin-2):], label=r'TT+TE+EE+$\phi\phi$', linewidth=w, linestyle='--')
    ax[i].set_ylabel(r'$F_{%s%s}$ per $\ell$' % (param, param), fontsize=24)
    ax[i].set_xlabel(r'$\ell$', fontsize=22)
    ax[i].legend()
    #ax[i].set_title(title)

plot.savefig('fish_info.png', bbox_inches='tight')
plot.show()

# VERIFIED THESE NUMBERS ADD TO THE FINAL FISHER MATRIX PROVIDED BY FISHERLENS
# CODE FOR THIS VERIFICATION IS BELOW
# NOTE YOU MUST SET fsky = 1.0
"""
print(l)
full_fisher = 0
for ell in l:
    #print(ell)
    ell -= 2
    C = np.matrix([
        [cl['cl_TT'][ell], cl['cl_TE'][ell], 0],
        [cl['cl_TE'][ell], cl['cl_EE'][ell], 0],
        [0, 0, cl['cl_phiphi'][ell]]
    ])
    Ci = np.linalg.inv(C)
    dC = np.matrix([
        [dcl['cl_TT'][ell], dcl['cl_TE'][ell], 0],
        [dcl['cl_TE'][ell], dcl['cl_EE'][ell], 0],
        [0, 0, dcl['cl_phiphi'][ell]]
    ])
    ell += 2

    full_fisher += (ell+0.5)*fsky*np.trace(Ci*dC*Ci*dC)


print(full_fisher)
"""
