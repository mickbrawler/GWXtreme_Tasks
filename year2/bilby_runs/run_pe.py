import numpy as np

import bilby

import lalsimulation as lalsim
import lal
import argparse
import os
import glob
import sys

def parse_cmd():
    parser=argparse.ArgumentParser()
    parser.add_argument("--luminosityDistance", help = "luminosity distance of injected source")
    parser.add_argument("--mass1", help = "mass of the primary component injected source")
    parser.add_argument("--mass2", help = "mass of the secondary component injected source")
    parser.add_argument("--waveform-approximant", help = "waveform approximant")
    parser.add_argument("--eos", help = "eos")
    parser.add_argument("--psd-dir", help = "directory containing power spectral densities of detectors")
    parser.add_argument("--ra",help= "injected right ascension")
    parser.add_argument("--dec",help= "injected declination")
    parser.add_argument("--inc",help= "injected inclination")
    parser.add_argument("--psi",help= "injected polarization")
    parser.add_argument("--observing-run", help='observing run')
    parser.add_argument("--snr-range", help="range of injected snrs")
    args=parser.parse_args()
    return args
args=parse_cmd()

D=args.luminosityDistance
m1=float(args.mass1)
m2=float(args.mass2)
ra=float(args.ra)
dec=float(args.dec)
inc=float(args.inc)
psi=float(args.psi)
obr=args.observing_run
approximant=args.waveform_approximant
eos=args.eos
psd_files=glob.glob(args.psd_dir+'*.txt')
mc=(m1*m2)**(3./5.)/(m1+m2)**(1./5.)>1.9 or (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
if mc<0.8 or mc>2.7:
    
    sys.exit()

# Change outdir to our own. Make appropriate sub directories
#outdir = '/home/anarya.ray/gwxtreme-project/injection_run/outdir2/O4_new9roq_equal/'+approximant+'/'+eos+'/'+args.snr_range+'_10low/'+str(D)+'_'+str(int(m1*100)/100.)+'_'+str(int(m2*100)/100.)+'/'
if(not os.path.exists(outdir)):
    os.makedirs(outdir)

file_to_det={'H1':"aligo",'L1':'aligo','V1':'avirgo','K1':'kagra'}

def lambda_from_piecewise(p,m):
    EOS=lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(p[0],p[1],p[2],p[3])
    FAM=lalsim.CreateSimNeutronStarFamily(EOS)                                                                                                                                                                                         
    m_max=lalsim.SimNeutronStarMaximumMass(FAM)/lal.MSUN_SI
    m_max = int(m_max*1000)/1000
    rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, FAM)
    kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, FAM)
    cc = m*lal.MRSUN_SI/rr
    return (2/3)*kk/(cc**5)

def lambda_from_spectral(p,m):
    EOS=lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(p[0],p[1],p[2],p[3])
    FAM=lalsim.CreateSimNeutronStarFamily(EOS)
    m_max=lalsim.SimNeutronStarMaximumMass(FAM)/lal.MSUN_SI
    m_max = int(m_max*1000)/1000
    rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, FAM)
    kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, FAM)
    cc = m*lal.MRSUN_SI/rr
    return (2/3)*kk/(cc**5)


def lambda_from_name(m,name):
    EOS=lalsim.SimNeutronStarEOSByName(name)
    FAM=lalsim.CreateSimNeutronStarFamily(EOS)
    m_max=lalsim.SimNeutronStarMaximumMass(FAM)/lal.MSUN_SI
    m_max = int(m_max*1000)/1000
    rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, FAM)
    kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, FAM)
    cc = m*lal.MRSUN_SI/rr
    return (2/3)*kk/(cc**5)

spectral_inj=np.array([+0.8651, +0.1548, -0.0151, -0.0002])
picewise_inj=np.array([])

if eos=='spectral':
    l1=lambda_from_spectral(spectral_inj,m1)
    l2=lambda_from_spectral(spectral_inj,m2)
elif eos=='piecewise':
    l1=lambda_from_piecewise(piecewise_inj,m1)
    l2=lambda_from_spectral(piecewise_inj,m2)
else:
    l1=lambda_from_name(m1,eos)
    l2=lambda_from_name(m2,eos)
print(bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(l1,l2,m1,m2))



#outdir = 'pe_dir'
label = 'bns_example'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

minimum_frequency = 20
reference_frequency = 100
duration = 512
sampling_frequency = 4096.
approximant = "TaylorF2"

# set up simulated data
np.random.seed(88170235)
injection_parameters = dict(
    chirp_mass=(m1*m2)**(3./5.)/(m1+m2)**(1./5.), mass_ratio=m2/m1, chi_1=0., chi_2=0., lambda_1=l1, lambda_2=l2,ra=ra, dec=dec, luminosity_distance=float(D), theta_jn=inc, psi=psi, phase=1.3, geocent_time=1187008882)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=dict(
        waveform_approximant=approximant,
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency
    )
)
if obr=='O4':
    interferometers =bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1','K1'])

    for ifo in interferometers:
        for fn in psd_files:
            if file_to_det[ifo.name] in fn:
                print(ifo.name,fn)
                ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=fn)
elif obr=='O3':
    interferometers =bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])

    for ifo in interferometers:
        for fn in psd_files:
            if ifo.name in fn:
                print(ifo.name,fn)
                ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=fn)
elif obr=='O2':
    interferometers =bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])

    for ifo in interferometers:
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=psd_files[0])
else:
    raise
for interferometer in interferometers:
    interferometer.minimum_frequency = minimum_frequency
interferometers.set_strain_data_from_zero_noise(sampling_frequency, duration, start_time=injection_parameters['geocent_time'] - duration + 2.)
interferometers.inject_signal(
    parameters=injection_parameters,
    waveform_generator=waveform_generator
)

# set up priors
priors = bilby.gw.prior.BNSPriorDict()
priors.pop("mass_1")
priors.pop("mass_2")

mc=injection_parameters['chirp_mass']
priors['chirp_mass'].minimum = mc * 0.95
priors['chirp_mass'].maximum = mc * 1.05
priors["mass_ratio"].minimum = 0.125
priors["mass_ratio"].maximum = 1
priors['chi_1'] = bilby.core.prior.Uniform(minimum=-0.05, maximum=0.05, name="chi_1")
priors['chi_2'] = bilby.core.prior.Uniform(minimum=-0.05, maximum=0.05, name="chi_2")
priors.pop('lambda_1')
priors.pop('lambda_2')
priors['lambda_tilde'] = bilby.core.prior.Uniform(0, 5000, name='lambda_tilde')
priors['delta_lambda'] = bilby.core.prior.Uniform(-5000, 5000, name='delta_lambda')
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1,
    maximum=injection_parameters['geocent_time'] + 0.1,
    name='geocent_time', latex_label='$t_c$', unit='$s$'
)
priors["luminosity_distance"] = bilby.core.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=min(10,float(D)-10), maximum=max(100,float(D)+100), unit='Mpc', latex_label='$d_L$')

# set up ROQ likelihood
search_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.binary_neutron_star_roq,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=dict(
        waveform_approximant=approximant,
        reference_frequency=reference_frequency
    )
)
roq_params = np.array(
    [(minimum_frequency, sampling_frequency / 2, duration, 0.8, 1.9, 0)],
    dtype=[("flow", float), ("fhigh", float), ("seglen", float), ("chirpmassmin", float), ("chirpmassmax", float), ("compmin", float)]
)
likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient(
    interferometers, search_waveform_generator, priors,
    linear_matrix="/home/anarya.ray/gwxtreme-project/injection_run/roq/ROQdata/basis.hdf5", quadratic_matrix="/home/anarya.ray/gwxtreme-project/injection_run/roq/ROQdata/basis.hdf5", roq_params=roq_params,
    distance_marginalization=True, phase_marginalization=True
)

# sampling
npool = 8
nact = 10
nlive = 2000
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', use_ratio=True,
    nlive=nlive, walks=100, maxmcmc=5000, nact=nact, npool=npool,
    injection_parameters=injection_parameters, outdir=outdir, label=label,
)    #conversion_function=bilby.gw.conversion.generate_all_bns_parameters


result.plot_corner()
