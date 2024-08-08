

for EoS in nest_data.keys():                                                                                                                                                        [12/1985]
    EoS1 = nest_data[EoS][0]
    EoS2 = nest_data['SLY'][0]
    BF = EoS1/EoS2

    EoS1err = nest_data[EoS][-1]
    EoS2err = nest_data['SLY'][-1]

    EoS1min, EoS1max = EoS1-EoS1err, EoS1+EoS1err
    EoS2min, EoS2max = EoS2-EoS2err, EoS2+EoS2err

    ErrMin = EoS1max/EoS2min
    ErrMax = EoS1min/EoS2max
    ErrMax - ErrMin

    Phenom_EoS_BF_err[EoS] = [BF,[err]]
4626/34: data['IMRPhenom LALInference_Nest'] = Phenom_EoS_BF_err
4626/35:
with open("data/BNS/BFs/GW170817_2D_3D_BFs.json","w") as f:
    json.dump(data, f, indent=2, sort_keys=True)
