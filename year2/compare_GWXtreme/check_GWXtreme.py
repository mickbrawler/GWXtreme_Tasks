from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import json
import pylab as pl

class checker:

    def __init__(self, env):

        self.env = env
        self.ref_EoS = "SLY"
        self.EoS_list = ["APR4_EPP", "H4", "MS1"]
        self.posterior_files = ["posterior_files/posterior_samples_broad_spin_prior.dat",
                                "posterior_files/posterior_samples_narrow_spin_prior.dat"]

        # Parametrized bestfits
        with open("../parametric_tests/files/basic_runs/1_piecewise_EoS_bestfits.json", "r") as f:
            self.piecewise_EoS = json.load(f)
        with open("../parametric_tests/files/basic_runs/1_spectral_EoS_bestfits.json", "r") as f:
            self.spectral_EoS = json.load(f)

    def makeFiles(self)
        # Makes MRK and ML files for each EoS

        m_min = 1.0
        N = 1000
        for EoS in self.EoS_list:

            eos = lalsim.SimNeutronStarEOSByName(EoS)
            fam = lalsim.CreateSimNeutronStarFamily(eos)
            max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
            max_mass = int(max_mass*1000)/1000
            masses = np.linspace(m_min, max_mass, N)
            masses = masses[masses <= max_mass]

            Lambdas = []
            gravMass = []
            radii = []
            kappas = []
            for m in masses:

                try:
                    rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
                    kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
                    cc = m*lal.MRSUN_SI/rr
                    Lambdas = np.append(Lambdas, (2/3)*kk/(cc**5))
                    gravMass = np.append(gravMass, m)
                    radii.append(rr)
                    kappas.append(kk)
                except RuntimeError:
                    break

            Lambdas = np.array(Lambdas)
            gravMass = np.array(gravMass)
            radii = np.array(radii)
            kappas = np.array(kappas)

            MRK = np.array([gravMass,radii,kappas])
            ML = np.array([gravMass,Lambdas])
            np.savetxt("comparison_files/MRK/{}.txt".format(EoS))
            np.savetxt("comparison_files/ML/{}.txt".format(EoS))

    def get_eos_BF(self):
        # Produces file of a dictionary of each EoS' Bayes factor.
        # Different Bayes factor for each version of an EoS (named, MRK, ML,
        # piecewise, spectral)

        increment = 0
        Type = ["narrow", "broad"]
        for posterior_file in posterior_files:

            modsel = ems.Model_selection(posteriorFile=posterior_file, spectral=False)
            s_modsel = ems.Model_selection(posteriorFile=posterior_file, spectral=True)
            
            named_BFs = {}
            MRK_BFs = {}
            ML_BFs = {}
            piecewise_BFs = {}
            spectral_BFs = {}
            for EoS in self.EoS_list:

                BF = modsel.computeEvidenceRatio(EoS1=EoS, EoS2=self.ref_EoS)
                named_BFs.update({EoS:BF})
                BF = modsel.computeEvidenceRatio(EoS1="comparison_files/MRK/{}.txt".format(EoS), EoS2=self.ref_EoS)
                MRK_BFs.update({EoS:BF})
                BF = modsel.computeEvidenceRatio(EoS1="comparison_files/ML/{}.txt".format(EoS), EoS2=self.ref_EoS)
                ML_BFs.update({EoS:BF})
                BF = modsel.computeEvidenceRatio(EoS1=self.piecewise_EoS[EoS], EoS2=self.ref_EoS)
                piecewise_BFs.update({EoS:BF})
                BF = s_modsel.computeEvidenceRatio(EoS1=self.spectral_EoS[EoS], EoS2=self.ref_EoS)
                spectral_BFs.update({EoS:BF})

            with open("comparison_files/named/{}_{}_BF".format(Type[increment],self.env), "w") as f:
                json.dump(named_BFs, f, indent=2, sort_keys=True)
            with open("comparison_files/MRK/{}_{}_BF".format(Type[increment],self.env), "w") as f:
                json.dump(MRK_BFs, f, indent=2, sort_keys=True)
            with open("comparison_files/ML/{}_{}_BF".format(Type[increment],self.env), "w") as f:
                json.dump(ML_BFs, f, indent=2, sort_keys=True)
            with open("comparison_files/piecewise/{}_{}_BF".format(Type[increment],self.env), "w") as f:
                json.dump(piecewise_BFs, f, indent=2, sort_keys=True)
            with open("comparison_files/spectral/{}_{}_BF".format(Type[increment],self.env), "w") as f:
                json.dump(spectral_BFs, f, indent=2, sort_keys=True)

            increment += 1

    def get_eos_stack_BF(self):
        # Produces file of a dictionary of each EoS' stacked Bayes factor.
        # Different Bayes factor for each version of an EoS (named, MRK, ML,
        # piecewise, spectral)
        
        stackobj = ems.Stacking(self.posterior_files, spectral=False)
        s_stackobj = ems.Stacking(self.posterior_files, spectral=True)
        
        named_stack_BFs = {}
        MRK_stack_BFs = {}
        ML_stack_BFs = {}
        piecewise_stack_BFs = {}
        spectral_stack_BFs = {}
        for EoS in self.EoS_list:

            stack_BF = stackobj.stack_events(EoS, self.ref_EoS)
            named_stack_BFs.update({EoS:stack_BF})
            stack_BF = stackobj.stack_events("comparison_files/MRK/{}.txt".format(EoS), self.ref_EoS)
            MRK_stack_BFs.update({EoS:stack_BF})
            stack_BF = stackobj.stack_events("comparison_files/ML/{}.txt".format(EoS), self.ref_EoS)
            ML_stack_BFs.update({EoS:stack_BF})
            stack_BF = stackobj.stack_events(self.piecewise_EoS[EoS], EoS2=self.ref_EoS)
            piecewise_stack_BFs.update({EoS:stack_BF})
            stack_BF = s_stackobj.stack_events(self.spectral_EoS[EoS], EoS2=self.ref_EoS)
            spectral_stack_BFs.update({EoS:stack_BF})

        with open("comparison_files/named/stack_{}_BF".format(self.env), "w") as f:
            json.dump(named_stack_BFs, f, indent=2, sort_keys=True)
        with open("comparison_files/MRK/stack_{}_BF".format(self.env), "w") as f:
            json.dump(MRK_stack_BFs, f, indent=2, sort_keys=True)
        with open("comparison_files/ML/stack_{}_BF".format(self.env), "w") as f:
            json.dump(ML_stack_BFs, f, indent=2, sort_keys=True)
        with open("comparison_files/piecewise/stack_{}_BF".format(self.env), "w") as f:
            json.dump(piecewise_stack_BFs, f, indent=2, sort_keys=True)
        with open("comparison_files/spectral/stack_{}_BF".format(self.env), "w") as f:
            json.dump(spectral_stack_BFs, f, indent=2, sort_keys=True)

    def get_perc_error(self):
        # Produce dictionary of each eos' percect error (between GWXtreme & Anarya's build)
        # Gets percent error between the files of the two env (base & anarya)
        # Each env has files for each type of input (named, MRK, ML) for modsel
        # and stacking.

        # Function needs redesign to be capable of testing every type of file

        versions = ["named", "MRK", "ML", "piecewise", "spectral"]
        for version in versions:

            tests = ["narrow", "broad", "stack"]
            for test in tests:

                with open("comparison_files/{}/{}_base_BF.json","r") as f:
                    base_dict = json.load(f)

                with open("comparison_files/{}/{}_anarya_BF.json","r") as f:
                    anarya_dict = json.load(f)

                base_vals = np.array(list(base_dict.values()))
                anarya_vals = np.array(list(anarya_dict.values()))
                perc_error = (np.abs(base_vals - anarya_vals) / base_vals) * 100

                pl.clf()
                pl.figure(figsize=(25, 10))
                pl.bar(eos_names, perc_error)
                pl.title("Percent Error of Anarya's Build")
                pl.xlabel("EoS Names")
                pl.ylabel("Percent Error")
                pl.xticks(rotation=45, ha='right', fontsize=5)
                pl.tight_layout()
                pl.savefig("comparison_files/{}/{}_comparison.png")

