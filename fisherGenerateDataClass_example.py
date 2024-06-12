import sys
#import cambWrapTools
import classWrapTools
import fisherTools
import pickle
import scipy
import numpy
import os
import numpy as np

import copy

rank = 0
size = 1

# CHOOSE NOISE HERE - CV is 0 for all, including delensing noise
PLANCK_ONLY = False # DONT USE
DRAFT_PLANCK = False # DONT USE

DRAFT_ONLY = True
CV = False

# PLANCK_ONLY uses Zack's parameters for TT and EE and the delensing noise is computed iteratively by class_delens
# CV is 0 for all, including delensing noise 
# DRAFT_PLANCK uses DRAFT with Planck added only in TT
# DRAFT_ONLY uses DRAFT

# CHOOSE DM MODEL HERE
ANN = False
SCATTER = True
DECAY = False


if not (ANN ^ SCATTER ^ DECAY) and not (ANN and SCATTER and DECAY):
    raise ValueError('Please select one and only one of scattering, annihilation, or decay.')


###  Set of experiments  ###
# Results will be indexed by experiment number, starting from 0
expNames = [0]
nExps = len(expNames)
if PLANCK_ONLY:
    lmax = 2000
else:
    lmax = 4500
lmaxTT = lmax
lmin = 2

# need to sort out l
# put in TT white noise and DRAFT noise, compare white noise to fishchips

lbuffer = 0
lmax_calc = lmax+lbuffer

expNamesThisNode = numpy.array_split(numpy.asarray(expNames), size)[rank]

# Directory where CLASS_delens is located
classExecDir = 'CLASS_delens/class_delens/'
# Directory where you would like the output
classDataDir = 'CLASS_delens/'
outputDir = classDataDir + 'results/'

classDataDirThisNode = classDataDir + 'data/Node_' + str(rank) + '/'
# Base name to use for all output files
if SCATTER:
    fileBase = 'scattercv' if CV else 'scatter'
elif ANN:
    fileBase = 'anncv' if CV else 'ann'
elif DECAY:
    fileBase = 'decaycv' if CV else 'decay'

if not os.path.exists(classDataDirThisNode):
    os.makedirs(classDataDirThisNode)
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# Spectrum types and polarizations to include
spectrumTypes = ['unlensed', 'lensed', 'delensed', 'lensing']
polCombs = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd']

#######################################################################################3
#LOAD PARAMS AND GET POWER SPECTRA

extra_params = dict()

#Fiducial values and step sizes taken from arXiv:1509.07471 Allison et al
c = 2.997e8
if ANN:
    print('Calculating with annihilation...')
    cosmoFid = {'omega_c_h2' : 0.1197,
                'omega_b_h2': 0.0222,
                #'N_eff': 3.046, \
                'A_s' : 2.196e-9,
                'n_s' : 0.9655,
                'tau' : 0.054,
                #'H0' : 67.5, \
                'theta_s' : 0.010409,
                #'Yhe' : 0.25, \
                #'r'   : 0.01, \
                #'mnu' : 0.06,
                'pann': 0
                }
    #cosmoFid['n_t'] = - cosmoFid['r'] / 8.0 * (2.0 - cosmoFid['n_s'] - cosmoFid['r'] / 8.0)

    stepSizes = {'omega_c_h2' : 0.0030,
                'omega_b_h2': 0.0008,
                #'N_eff': .080, \
                'A_s' : 0.1e-9,
                'n_s' : 0.010,
                'tau' : 0.002,
                #'H0' : 1.2, \
                'theta_s' : 0.000050,
                #'mnu' : 0.02, \
                #'r'   : 0.001, \
                #'n_t' : cosmoFid['n_t'], \
                #'Yhe' : 0.0048,
                'pann': 1.0e-8/9e16
                }
elif DECAY:
    print('Calculating with decay...')
    cosmoFid = {'omega_c_h2' : 0.1197,
                'omega_b_h2': 0.0222,
                #'N_eff': 3.046, \
                'A_s' : 2.196e-9,
                'n_s' : 0.9655,
                'tau' : 0.054,
                #'H0' : 67.5, \
                'theta_s' : 0.010409,
                #'Yhe' : 0.25, \
                #'r'   : 0.01, \
                #'mnu' : 0.06,
                'DM_decay_Gamma': 0
                }
    #cosmoFid['n_t'] = - cosmoFid['r'] / 8.0 * (2.0 - cosmoFid['n_s'] - cosmoFid['r'] / 8.0)

    stepSizes = {'omega_c_h2' : 0.0030,
                'omega_b_h2': 0.0008,
                #'N_eff': .080, \
                'A_s' : 0.1e-9,
                'n_s' : 0.010,
                'tau' : 0.002,
                #'H0' : 1.2, \
                'theta_s' : 0.000050,
                #'mnu' : 0.02, \
                #'r'   : 0.001, \
                #'n_t' : cosmoFid['n_t'], \
                #'Yhe' : 0.0048,
                'DM_decay_Gamma': 1.e-30
                }
    extra_params['DM_decay_fraction'] = 1.
elif SCATTER:
    print('Calculating with scattering...')
    cosmoFid = {'omega_dmeff': 0.12011,
                'omega_b': 0.022383,
                'A_s': 2.196e-9,
                'n_s': 0.9655,
                'tau': 0.054,
                'theta_s': 0.010409,
                #'log10m_dmeff': 0,  # 1e-5 1e-4 1e-3 1e-2 1e-1 1 10 100 1000
                #'log10sigma_dmeff': -25
                'sigma_dmeff': 1.e-100
                }
    stepSizes = {'omega_dmeff': 0.0030,
                 'omega_b': 0.0008,
                 'A_s': 0.1e-9,
                 'n_s': 0.010,
                 'tau': 0.002,
                 'theta_s': 0.000050,
                 #'log10m_dmeff': 0.1,
                 #'log10sigma_dmeff': 1
                 'sigma_dmeff': 1.e-40
                 }
    extra_params['log10m_dmeff'] = -5  # -5 -4 -3 -2 -1 0 1 2 3
    extra_params['npow_dmeff'] = 0

cosmoParams = list(cosmoFid.keys())
delta_l_max = 5000-lmax
ell = numpy.arange(2,lmax_calc+1+delta_l_max)
lvec = numpy.arange(2,lmax+1)


# Mask the \ells you do not want included in lensing reconstruction
# Keys can be added as e.g. 'lmin_T', 'lmax_T', etc.
reconstructionMask = dict()
reconstructionMask['lmax_T'] = lmaxTT


#extra_params['delensing_verbose'] = 3
#extra_params['output_spectra_noise'] = 'no'
#extra_params['write warnings'] = 'y'
extra_params['delta_l_max'] = delta_l_max
if SCATTER:
    fileBase += '_m' + str(extra_params['log10m_dmeff']) + '_n' + str(extra_params['npow_dmeff'])


# Specify \ells to keep when performing Fisher matrix sum
ellsToUse = {'cl_TT': [lmin, lmaxTT], 'cl_TE': [lmin, lmax], 'cl_EE': [lmin, lmax], 'cl_dd': [2, lmax]}
ellsToUseNG = {'cl_TT': [lmin, lmaxTT], 'cl_TE': [lmin, lmax], 'cl_EE': [lmin, lmax], 'cl_dd': [2, lmax], 'lmaxCov': lmax_calc}

cmbNoiseSpectra = dict()
deflectionNoises = dict()
paramDerivs = dict()
powersFid = dict()
invCovDotParamDerivs_delensed = dict()
invCovDotParamDerivs_lensed = dict()
paramDerivStack_delensed = dict()
paramDerivStack_lensed = dict()
fisherGaussian = dict()
fisherNonGaussian_delensed = dict()
fisherNonGaussian_lensed = dict()

# Flags for whether to include NonGaussian covariances, and derivatives wrt unlensed spectra
includeUnlensedSpectraDerivatives = False

# Calculations begin
print('Node ' + str(rank) + ' working on experiment ' + str(expNames[0]))


#================
# start create noise
#================
if DRAFT_ONLY:
    cmbNoiseSpectra = pickle.load(open('noise_draft.pkl', 'rb'))
if DRAFT_PLANCK:
    cmbNoiseSpectra = pickle.load(open('noise_draftplanck.pkl', 'rb'))

if CV or PLANCK_ONLY:
    cmbNoiseSpectra = {'l' : 0,
                'cl_TT' : 0,
                'cl_EE' : 0,
                'cl_TE' : 0,
                'cl_BB' : 0,

                'cl_dd' : 0,

                'dl_TT' : 0,
                'dl_EE' : 0,
                'dl_TE' : 0,
                'dl_BB' : 0
                }

    for key in cmbNoiseSpectra.keys():
        cmbNoiseSpectra[key] = np.zeros_like(ell)

# start planck only
if PLANCK_ONLY:
    arcmin_to_radian = np.pi / 60. / 180.
    l = ell
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
    planck_noise_P[:28] = 1e100
    cmbNoiseSpectra['cl_TT'] = 1/planck_noise_T
    cmbNoiseSpectra['dl_TT'] = l * (l + 1) / 2 / np.pi * cmbNoiseSpectra['cl_TT']
    cmbNoiseSpectra['cl_EE'] = 1/planck_noise_P
    cmbNoiseSpectra['dl_EE'] = l * (l + 1) / 2 / np.pi * cmbNoiseSpectra['cl_EE']
# end planck only


cmbNoiseSpectra['l'] = ell
print(cmbNoiseSpectra)


#================
# end create noise
#================

if not CV:
    powersFid, cmbNoiseSpectra['cl_dd'] = classWrapTools.class_generate_data(cosmoFid,
                                     cmbNoise = cmbNoiseSpectra,
                                     extraParams = extra_params,
                                     rootName = fileBase,
                                     lmax = lmax_calc,
                                     classExecDir = classExecDir,
                                     classDataDir = classDataDirThisNode,
                                     reconstructionMask = reconstructionMask)

    print('Computed Nl_dd = ', cmbNoiseSpectra['cl_dd'])
else:
    powersFid, _ = classWrapTools.class_generate_data(cosmoFid,
                                     cmbNoise = cmbNoiseSpectra,
                                     deflectionNoise = cmbNoiseSpectra['cl_dd'],
                                     extraParams = extra_params,
                                     rootName = fileBase,
                                     lmax = lmax_calc,
                                     classExecDir = classExecDir,
                                     classDataDir = classDataDirThisNode,
                                     reconstructionMask = reconstructionMask)


# BEGIN FISHER
paramDerivs = fisherTools.getPowerDerivWithParams(cosmoFid = cosmoFid, \
                        extraParams = extra_params, \
                        stepSizes = stepSizes, \
                        polCombs = polCombs, \
                        cmbNoiseSpectraK = cmbNoiseSpectra, \
                        deflectionNoisesK = cmbNoiseSpectra['cl_dd'], \
                        useClass = True, \
                        lmax = lmax_calc, \
                        fileNameBase = fileBase, \
                        classExecDir = classExecDir, \
                        classDataDir = classDataDirThisNode)

fisherGaussian = fisherTools.getGaussianCMBFisher(powersFid = powersFid, \
                        paramDerivs = paramDerivs, \
                        cmbNoiseSpectra = cmbNoiseSpectra, \
                        deflectionNoises = cmbNoiseSpectra['cl_dd'], \
                        cosmoParams = cosmoParams, \
                        spectrumTypes = ['unlensed', 'lensed', 'delensed'], \
                        polCombsToUse = polCombs, \
                        ellsToUse = ellsToUse)
    
print('Node ' + str(rank) + ' finished all experiments')

forecastData = {'cmbNoiseSpectra' : cmbNoiseSpectra,
                'powersFid' : powersFid,
                'paramDerivs': paramDerivs,
                'fisherGaussian': fisherGaussian,
                'deflectionNoises' : cmbNoiseSpectra['cl_dd']}

print('Node ' + str(rank) + ' saving data')

filename = classDataDirThisNode + fileBase + '.pkl'
delensedOutput = open(filename, 'wb')
pickle.dump(forecastData, delensedOutput, -1)
delensedOutput.close()
print('Node ' + str(rank) + ' saving data complete')

if rank==0:
    print('Node ' + str(rank) + ' collecting data')
    for irank in range(1,size):
        print('Getting data from node ' + str(irank))
        filename = classDataDir + 'data/Node_' + str(irank) + '/' + fileBase + '_' + str(irank) + '.pkl'
        nodeData = open(filename, 'rb')
        nodeForecastData = pickle.load(nodeData)
        nodeData.close()
        for key in list(forecastData.keys()):
            forecastData[key].update(nodeForecastData[key])

    print('Node ' + str(rank) + ' reading script')
    f = open(os.path.abspath(__file__), 'r')
    script_text = f.read()
    f.close()

    forecastData['script_text'] = script_text

    forecastData['cosmoFid'] = cosmoFid
    forecastData['cosmoParams'] = cosmoParams

    print('Node ' + str(rank) + ' saving collected data')
    filename = outputDir + fileBase + '.pkl'
    delensedOutput = open(filename, 'wb')
    pickle.dump(forecastData, delensedOutput, -1)
    delensedOutput.close()
