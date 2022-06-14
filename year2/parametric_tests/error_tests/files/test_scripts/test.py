import time
import argparse

def Main(samples, core, Bool):
    Time = 2
    print(samples)
    time.sleep(Time)
    print(core)
    time.sleep(Time)
    print(Bool)
    time.sleep(Time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", help="Number of samples", type=float)
    parser.add_argument("core", help="Core number", type=float)
    parser.add_argument("new", help="True-(make files) : False-(use old files)", type=bool)
    args = parser.parse_args()

    Main(args.samples,args.core,args.new)
