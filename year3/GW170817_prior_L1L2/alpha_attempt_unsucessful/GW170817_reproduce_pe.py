# This is a script shared by Anarya Ray that will do a uniform Prior (L1,L2) GW170817 bilby run
import multiprocessing
import os
import optparse

import bilby
import numpy as np


parser = optparse.OptionParser()
parser.add_option("--nact", dest="nact", default=10, type="int")
parser.add_option("--nlive", dest="nlive", default=2000, type="int")
parser.add_option("--npool", dest="npool", default=100, type="int")
parser.add_option("--no-phase-marg", action="store_true", default=False, dest="no_phase_marg")
parser.add_option("--low-spin", action="store_true", default=False, dest="low_spin")
(options, args) = parser.parse_args()

path = "/home/michael/projects/eos/GWXtreme_Tasks/year3/GW170817_prior_L1L2"
outdir = path+"/outdir/2nd_attempt/" 
Label = "GW170817"

time_of_event = 1187008882.4   # GPS time of the event informed by detection pipelines (in unit of seconds)
duration = 128.   # Duration of data used for parameter estimation (in unit of seconds)
minimum_frequency = 23.   # minimum frequency of the integration for a likelihood evaluation (in unit of Hz)
sampling_frequency = 4096.    # twice the maximum frequency of the integration for a likelihood evaluation (in unit of Hz)
maximum_frequency=sampling_frequency/2.0#1200.0
post_trigger_duration = 2   # time offset padded at the end of data to include merger-ringdown part (in unit of seconds)
end_time = time_of_event + post_trigger_duration
start_time = end_time - duration

# PSDs were downloaded from https://dcc.ligo.org/LIGO-P1900011/public.
psds = np.loadtxt(path+"/anarya.ray/gwxtreme-project/170817/17noneos/data/GWTC1_GW170817_PSDs.dat").T
psd_frequencies = psds[0]
psd_dict = {"H1": psds[1], "L1": psds[2], "V1": psds[3]}

# H1 and L1 data can be downloaded from https://www.gw-openscience.org/eventapi/html/GWTC-1-confident/GW170817/v3.
# L1 data near GW170817 contains a loud glitch. Special frames with the glitch removed can be downloaded from
# https://dcc.ligo.org/LIGO-T1700406/public. 
#
# The data files are ignored by .gitignore, as they have large sizes. You can download them easily by moving
# to ./gw170817_data and running `source download_framedata.sh`.
data_dict = {
    "H1": path+"/anarya.ray/gwxtreme-project/170817/17noneos/data/H-H1_GWOSC_16KHZ_R1-1187006835-4096.gwf",
    "L1": path+"/anarya.ray/gwxtreme-project/170817/17noneos/data/L-L1_CLEANED_HOFT_C02_T1700406_v3-1187008667-4096.gwf",
    "V1": path+"/anarya.ray/gwxtreme-project/170817/17noneos/data/V-V1_GWOSC_16KHZ_R1-1187006835-4096.gwf"}
channel_dict = {
    "H1": "H1:GWOSC-16KHZ_R1_STRAIN",
    "L1": "L1:DCH-CLEAN_STRAIN_C02_T1700406_v3",
    "V1": "V1:GWOSC-16KHZ_R1_STRAIN"}

# Set up interferometers
interferometers = bilby.gw.detector.InterferometerList([])
for det in ["H1", "L1", "V1"]:
    interferometer = bilby.gw.detector.get_empty_interferometer(det)
    interferometer.minimum_frequency = minimum_frequency
    interferometer.set_strain_data_from_frame_file(
        data_dict[det], sampling_frequency, duration, start_time, channel_dict[det])
    interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd_frequencies, psd_array=psd_dict[det])
    interferometers.append(interferometer)

# Set up priors
priors = bilby.gw.prior.BNSPriorDict(filename=path+"/anarya.ray/gwxtreme-project/170817/17noneos/data/standard_ul1l2.prior")
if options.low_spin:
    priors["chi_1"].maximum = 0.05
    priors["chi_2"].maximum = 0.05
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=time_of_event - 0.1, maximum=time_of_event + 0.1,
    name='geocent_time', latex_label='$t_c$', unit='$s$')

# Add calibration errors. Calibration errors were downloaded from https://dcc.ligo.org/LIGO-P1900040/public.
spline_calibration_envelope_dict = {
    "H1": path+"/anarya.ray/gwxtreme-project/170817/17noneos/data/GWTC1_GW170817_H_CalEnv.txt",
    "L1": path+"/anarya.ray/gwxtreme-project/170817/17noneos/data/GWTC1_GW170817_L_CalEnv.txt",
    "V1": path+"/anarya.ray/gwxtreme-project/170817/17noneos/data/GWTC1_GW170817_V_CalEnv.txt"
}
spline_calibration_nodes = 10
for ifo in interferometers:
    ifo.maximum_frequency = maximum_frequency
for det in ["H1", "L1", "V1"]:
    cal_prior = bilby.gw.prior.CalibrationPriorDict.from_envelope_file(
        spline_calibration_envelope_dict[det],
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,# sampling_frequency/2.
        n_nodes=spline_calibration_nodes,
        label=det
    )
    priors.update(cal_prior)
for ifo in interferometers:
    ifo.calibration_model = bilby.gw.calibration.CubicSpline(
        prefix=f"recalib_{ifo.name}_",
        minimum_frequency=ifo.minimum_frequency,
        maximum_frequency=ifo.maximum_frequency,
        n_points=spline_calibration_nodes
    )

# Construct likelihood
approximant = "TaylorF2"   # Waveform model
reference_frequency = 100.   # Reference frequency for spin parameters
waveform_generator_mb = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.binary_neutron_star_frequency_sequence,
    waveform_arguments=dict(waveform_approximant=approximant, reference_frequency=reference_frequency),
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
)

#if not os.path.exists(outdir):
#    os.makedirs(outdir)

likelihood_mb = bilby.gw.likelihood.MBGravitationalWaveTransient(
    interferometers=interferometers,
    waveform_generator=waveform_generator_mb,
    priors=priors,
    reference_chirp_mass=1.15,#priors['chirp_mass'].minimum,
    distance_marginalization=True,
    phase_marginalization=True,
    distance_marginalization_lookup_table=os.path.join(outdir,'.distance_marginalization_lookup_{}.npz'.format(Label))
)

# Sampling
npool = min(options.npool, multiprocessing.cpu_count())
result = bilby.run_sampler(
    likelihood=likelihood_mb, priors=priors, sampler='dynesty', use_ratio=True,
    nlive=options.nlive, nact=options.nact, npool=npool, outdir=outdir, label=Label)

# Make a corner plot
result.plot_corner()

