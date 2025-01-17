from GWXtreme import eos_model_selection as ems
import lalsimulation as lalsim
import lal
import numpy as np
import json
import pylab as pl

class checker:

    def __init__(self, env):
        # env : names used are "base" for default GWXtreme version and "anarya"
        #       for Anarya's version. This is solely used to name resultant file
        #       names.

        self.env = env
        self.ref_EoS = "SLY"
        self.EoS_list = ["APR4_EPP", "H4", "MS1"]
        self.posterior_files = ["posterior_samples/posterior_samples_narrow_spin_prior.dat"] # Just one file for now

        # Parametrized bestfits
        with open("1_piecewise_EoS_bestfits.json", "r") as f:
            self.piecewise_EoS = json.load(f)
        with open("1_spectral_EoS_bestfits.json", "r") as f:
            self.spectral_EoS = json.load(f)

    def makeFiles(self):
        # Makes MRK and ML files for each EoS

        N = 1000
        for EoS in self.EoS_list:

            eos = lalsim.SimNeutronStarEOSByName(EoS)
            fam = lalsim.CreateSimNeutronStarFamily(eos)
            max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
            max_mass = int(max_mass*1000)/1000
            min_mass = lalsim.SimNeutronStarFamMinimumMass(fam)/lal.MSUN_SI
            masses = np.linspace(min_mass, max_mass, N)
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

            MRK = np.array([gravMass,radii,kappas]).T
            ML = np.array([gravMass,Lambdas]).T
            np.savetxt("comparison_files/MRK/{}.txt".format(EoS), MRK)
            np.savetxt("comparison_files/ML/{}.txt".format(EoS), ML)

    def get_EoS_BF(self):
        # Produces file of a dictionary of each EoS' Bayes factor.
        # Different Bayes factor for each version of an EoS (named, MRK, ML,
        # piecewise, spectral)

        increment = 0
        tests = ["narrow"] # only this file for now. Incrementer is present incase we wanted to use broad as well.
        for posterior_file in self.posterior_files:

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

            with open("comparison_files/named/{}_{}_BF.json".format(tests[increment],self.env), "w") as f:
                json.dump(named_BFs, f, indent=2, sort_keys=True)
            with open("comparison_files/MRK/{}_{}_BF.json".format(tests[increment],self.env), "w") as f:
                json.dump(MRK_BFs, f, indent=2, sort_keys=True)
            with open("comparison_files/ML/{}_{}_BF.json".format(tests[increment],self.env), "w") as f:
                json.dump(ML_BFs, f, indent=2, sort_keys=True)
            with open("comparison_files/piecewise/{}_{}_BF.json".format(tests[increment],self.env), "w") as f:
                json.dump(piecewise_BFs, f, indent=2, sort_keys=True)
            with open("comparison_files/spectral/{}_{}_BF.json".format(tests[increment],self.env), "w") as f:
                json.dump(spectral_BFs, f, indent=2, sort_keys=True)

            increment += 1

    def get_EoS_stack_BF(self):
        # Produces file of a dictionary of each EoS' stacked Bayes factor.
        # Different Bayes factor for each version of an EoS (named, MRK, ML,
        # piecewise, spectral)
 
        # Uncomment below lines if you want to fully use Stacking class "joint" multiplication processes
        #self.posterior_files = ["posterior_samples/posterior_samples_narrow_spin_prior.dat",
        #                        "posterior_samples/posterior_samples_broad_spin_prior.dat"]

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

        with open("comparison_files/named/stack_{}_BF.json".format(self.env), "w") as f:
            json.dump(named_stack_BFs, f, indent=2, sort_keys=True)
        with open("comparison_files/MRK/stack_{}_BF.json".format(self.env), "w") as f:
            json.dump(MRK_stack_BFs, f, indent=2, sort_keys=True)
        with open("comparison_files/ML/stack_{}_BF.json".format(self.env), "w") as f:
            json.dump(ML_stack_BFs, f, indent=2, sort_keys=True)
        with open("comparison_files/piecewise/stack_{}_BF.json".format(self.env), "w") as f:
            json.dump(piecewise_stack_BFs, f, indent=2, sort_keys=True)
        with open("comparison_files/spectral/stack_{}_BF.json".format(self.env), "w") as f:
            json.dump(spectral_stack_BFs, f, indent=2, sort_keys=True)

### Postprocessing ### Postprocessing ### Postprocessing ### Postprocessing ### Postprocessing

    def get_perc_error(self):
        # Produce dictionary of each eos' percect error (between GWXtreme & Anarya's build)
        # Gets percent error between the files of the two env (base & anarya)
        # Each env has files for each type of input (named, MRK, ML) for modsel
        # and stacking.

        versions = ["named", "MRK", "ML", "piecewise", "spectral"]
        for version in versions:

            tests = ["narrow", "stack"]
            #tests = ["narrow", "broad", "stack"]
            for test in tests:

                with open("comparison_files/{}/{}_base_BF.json".format(version,test),"r") as f:
                    base_dict = json.load(f)

                with open("comparison_files/{}/{}_anarya_BF.json".format(version,test),"r") as f:
                    anarya_dict = json.load(f)

                base_vals = np.array(list(base_dict.values()))
                anarya_vals = np.array(list(anarya_dict.values()))
                perc_error = (np.abs(base_vals - anarya_vals) / base_vals) * 100

                pl.clf()
                pl.bar(self.EoS_list, perc_error)
                pl.title("Percent Error of Anarya's Build")
                pl.xlabel("EoS Names")
                pl.ylabel("Percent Error")
                pl.xticks(rotation=45, ha='right')
                pl.savefig("comparison_files/{}/error_plot/{}_comparison.png".format(version,test))

    # Check for plotting script doesn't need to be advanced. Maybe not needed at all.
    # It depends on modsel and stacking, which we already test.
    def get_EoS_plot(self):
        # Simply runs plot_func for every type of file possible

        increment = 0
        Type = ["narrow"]
        for posterior_file in self.posterior_files:

            modsel = ems.Model_selection(posteriorFile=posterior_file, spectral=False)
            s_modsel = ems.Model_selection(posteriorFile=posterior_file, spectral=True)
            
            for EoS in self.EoS_list:

                print("EoS Name")
                modsel.plot_func([EoS],filename="comparison_files/named/EoS_plot/{}_{}_{}.png".format(self.env,Type[increment],EoS))
                print("MRK")
                modsel.plot_func(["comparison_files/MRK/{}.txt".format(EoS)],filename="comparison_files/MRK/EoS_plot/{}_{}_{}.png".format(self.env,Type[increment],EoS))
                print("ML")
                modsel.plot_func(["comparison_files/ML/{}.txt".format(EoS)],filename="comparison_files/ML/EoS_plot/{}_{}_{}.png".format(self.env,Type[increment],EoS))
                print("piecewise")
                modsel.plot_func([self.piecewise_EoS[EoS]],filename="comparison_files/piecewise/EoS_plot/{}_{}_{}.png".format(self.env,Type[increment],EoS))
                print("spectral")
                s_modsel.plot_func([self.spectral_EoS[EoS]],filename="comparison_files/spectral/EoS_plot/{}_{}_{}.png".format(self.env,Type[increment],EoS))

        stackobj = ems.Stacking(self.posterior_files, spectral=False)
        s_stackobj = ems.Stacking(self.posterior_files, spectral=True)

        for EoS in self.EoS_list:

            print("EoS Name")
            stackobj.plot_stacked_bf(eos_list=[EoS],filename="comparison_files/named/EoS_plot/stack_{}_{}.png".format(self.env,EoS))
            print("MRK")
            stackobj.plot_stacked_bf(eos_list=["comparison_files/MRK/{}.txt".format(EoS)],filename="comparison_files/MRK/EoS_plot/stack_{}_{}.png".format(self.env,EoS))
            print("ML")
            stackobj.plot_stacked_bf(eos_list=["comparison_files/ML/{}.txt".format(EoS)],filename="comparison_files/ML/EoS_plot/stack_{}_{}.png".format(self.env,EoS))
            print("piecewise")
            stackobj.plot_stacked_bf(eos_list=[self.piecewise_EoS[EoS]],filename="comparison_files/piecewise/EoS_plot/stack_{}_{}.png".format(self.env,EoS))
            print("spectral")
            s_stackobj.plot_stacked_bf(eos_list=[self.spectral_EoS[EoS]],filename="comparison_files/spectral/EoS_plot/stack_{}_{}.png".format(self.env,EoS))

