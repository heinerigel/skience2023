import os
import matplotlib.pyplot as plt
from scipy import interpolate
import warnings
import logbook
import argparse
import sys

warnings.filterwarnings('ignore')
os.environ["CARTOPY_USER_BACKGROUNDS"] = "BG/"
from ww32seismo import *
import dynamic_yaml

def main(loglevel="INFO", show = False, outfile=True):
    parser = argparse.ArgumentParser(description = 'Extract SeismoRMS from WW3 P2L model')
    parser.add_argument('--Q', help = "List of Q values to compute separated with commas, without space")
    args = parser.parse_args(sys.argv[1:])
    Qs = args.Q
    logger = logbook.Logger("ww3")
    logger.info('*** Starting: Extracting spectra from P2L ***')
    # Import configurations and Settings
    with open("config.yml", 'r') as f:
        configs = dynamic_yaml.load(f)
    target = configs.params.station
    target_lat = configs.params.station_lat
    target_lon = configs.params.station_lon
    rhos = configs.params.rhos
    beta = configs.params.beta
    Rg = configs.params.Rg
    if not Qs:
        Qs = [configs.params.Q]
    else:
        Qs = [int(Qi) for Qi in Qs.split(',')]
    Re = 4.0e7/(2*np.pi)
    depth_file = configs.files.depth
    #  needed files: Rayleigh_source.txt, depth file, and 2 models with and without reflection /!\
    dataset = netcdf_dataset(r"{}".format(depth_file))
    dpt = pd.DataFrame(np.asarray(dataset.variables["dpt"])[50,:,:], columns=dataset.variables["longitude"], index=dataset.variables["latitude"])
    dpt[dpt==-32767] *= 0.0
    dpt[dpt<=0.0] = 0.0
    if not os.path.isdir("DATA"):
        os.mkdir("DATA")
    if not os.path.isdir("FIGURES"):
        os.mkdir("FIGURES")
    ## Plot depth and distance
    lats, lons, distance_df = get_distance(configs, dataset, dpt, plot=False)

    df = pd.read_csv(r'{}'.format(configs.files.noise_source_term), header=None, delim_whitespace=True, index_col=0)
    df.index *= np.pi
    df = df.fillna(0.0)
    C_base = (df[:8]**2).sum(axis=1)
    C_base.at[C_base.index[-1]+0.01] = 0.0
    C_base.at[-1.0] = 0.0
    C_base.at[20.0] = 0.0
    C_base = C_base.sort_index()
    Cf = interpolate.interp1d(C_base.index, C_base)
    for Q in Qs:
        dfF_fs = get_ww3(configs, Q, 10, lats, lons, Re, dpt, Cf, distance_df, plot=False)

        fig = plt.figure(figsize=(16,4), facecolor="w")
        cmap = plt.get_cmap('viridis')
        psd = 10* np.log10(dfF_fs)
        plt.pcolormesh(dfF_fs.columns, 1./dfF_fs.index, psd, cmap=cmap, vmin = -150, vmax =-110)
        plt.colorbar().set_label("Amplitude (dB)")
        plt.ylabel("Period (s)")
        plt.yscale('log')
        plt.title("Spectrogram {} Q{}".format(target,int(Q)))
        fig.autofmt_xdate()
        if outfile:
            outfile = "Spectrogram_{}_Q{}.png".format(target,int(Q))
            print("output to:", outfile)
            plt.savefig(os.path.join("FIGURES",outfile))
        if show:
            plt.show()
        fig = plt.figure(figsize=(16,5), facecolor="w")
        integ = np.sqrt(scipy.integrate.trapz(dfF_fs.fillna(0), dfF_fs.index, axis=0))
        plt.plot(dfF_fs.columns, integ)
        plt.ylabel("Amplitude")
        plt.title("dRMS {} Q{}".format(target,int(Q)))
        fig.autofmt_xdate()
        plt.xlim(dfF_fs.columns[0],dfF_fs.columns[-1])
        if outfile:
            outfile = "dRMS_{}_Q{}.png".format(target,int(Q))
            print("output to:", outfile)
            plt.savefig(os.path.join("FIGURES",outfile))
        if show:
            plt.show()
if __name__ == "__main__":
    main()