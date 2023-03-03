import ftplib
import tqdm
import dynamic_yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset as netcdf_dataset
import netCDF4
import numpy as np
import scipy.integrate
from obspy.geodetics.base import gps2dist_azimuth
from obspy import UTCDateTime
from obspy.geodetics.base import kilometer2degrees, locations2degrees
import numpy.matlib
from datetime import datetime
from cartopy import config
import cartopy.crs as ccrs
import cmasher as cmr

    
def download_p2l():
    """
    Downloads the P2L files directly from the IFREMER ftp for the year in the config.yml file
    """
    with open("config.yml", 'r') as f:
        configs = dynamic_yaml.load(f)
    # Fill Required Information
    HOSTNAME = "ftp.ifremer.fr"
    # Connect FTP Server
    ftp_server = ftplib.FTP(HOSTNAME)
    ftp_server.login() # Anonymous login
    # force UTF-8 encoding
    ftp_server.encoding = "utf-8"
    #from config
    sismo_path = configs.download.sismo_path
    year = configs.download.year
    p2l_save = configs.download.p2l_save
    if not os.path.isdir(p2l_save):
        os.mkdir(p2l_save)

    ref_paths = []

    try:
        ref_paths = ftp_server.nlst(sismo_path) # Get initial work directories with and without reflection
    except ftplib.error_perm as resp:
        if str(resp) == "550 No files found":
            print("No files in this directory")
        else:
            raise

    for p in ref_paths:
        print(p)
        f_list = ftp_server.nlst(os.path.join(p,"{}".format(year), "FIELD_NC"))
        p2l_list = [i for i in f_list if "_p2l.nc" in i] # Only list the p2l files
        pbar = tqdm.tqdm(p2l_list, desc="Downloading {}".format(p))
        for p2l in pbar:
            fn = p2l.split("/SISMO/")[1].split("FIELD_NC/")[1]
            save_dir = os.path.join(p2l_save, p2l.split("/SISMO/")[1].split("/LOPS")[0])
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            fpath = os.path.join(save_dir, fn)
            if not os.path.isfile(fpath):
                ftp_server.retrbinary('RETR {}'.format(p2l),open(fpath, 'wb').write)
    ftp_server.quit()
    
def plot_spec(dfF_fs, station, Q):
    """
    Plots the spectrogram for the station if the corresponding dataframe if already available
    """
    plt.rcParams['figure.figsize'] = (16,6)
    plt.rcParams['axes.facecolor'] = "w"
    fig, ax = plt.subplots()


    cmap = plt.get_cmap('viridis')
    cmin=-160
    cmax=-110
    ymin=1
    ymax=12

    psd = 10* np.log10(dfF_fs)
    plt.pcolormesh(dfF_fs.columns, 1./dfF_fs.index, psd, cmap=cmap, vmin = cmin, vmax =cmax)
#    plt.axvline(pd.to_datetime("2021-10-25"), color="w", ls="--")
#    plt.axvline(pd.to_datetime("2021-11-05"), color="w", ls="--")
    cb = plt.colorbar(ax=ax).set_label("Amplitude [$m^2/s^4/Hz$] [dB]")
    plt.ylabel("Period (s)")
    plt.yscale('log')
    fig.autofmt_xdate()
#    plt.ylim(0.1,ymax)
    plt.title("{}, Q = {}".format(station,int(Q)))
    plt.show()

def plot_rms(dfF_fs, station):
    """
    Plots the seismic RMS for the station if the corresponding dataframe if already available
    """    
    fig = plt.figure(figsize=(16,6), facecolor="w")
    integ = np.sqrt(scipy.integrate.trapz(dfF_fs.fillna(0), dfF_fs.index, axis=0))
    plt.plot(dfF_fs.columns, integ)
    plt.ylabel("Amplitude")
    fig.autofmt_xdate()
    plt.title(station)
    plt.xlim(dfF_fs.columns[0],dfF_fs.columns[-1])
    plt.show()    
    
    
    
def dispNewtonTH(f,dep):
    """inverts the linear dispersion relation (2*pi*f)^2=g*k*tanh(k*dep) to get
    k from f and dep. 2 Arguments: f and dep.
    """
    eps = 0.000001
    g = 9.81
    sig = 2.*np.pi*f
    Y = dep*sig**2./g
    X = np.sqrt(Y)
    I=1
    F = np.ones(dep.shape)
    
    while np.abs(np.max(F)) > eps:
        H = np.tanh(X)
        F = Y-X*H
        FD = -H-X/np.cosh(X)**2
        X -= F/FD
        F = F.values
    return X/dep

def compute_depth_correction(f, dep):
    wnum = dispNewtonTH(f, dep)
    kd = wnum*dep
    depth_correction = (np.tanh(kd))**2.0*(1.0 + 2.0*(kd/np.sinh(2*kd)))
    return depth_correction

def get_omegaoverbeta(fs, dpt, beta):
    omega = 2 * np.pi * fs * 2
    omehoverbeta=omega * dpt / beta     # omega*h/beta
    omehoverbeta[omehoverbeta>=10] = 10.0
    omehoverbeta[omehoverbeta<=0] = 0.0
    return omehoverbeta

def get_C(fs, dpt, beta, Cf):
    """Returns the value of ^C for fs
    """
    ob = get_omegaoverbeta(fs, dpt,beta)
    return Cf(ob)

def plot_surfarea(lons,lats,dA, Re):
    plt.figure(figsize=(18,8),)
    ax = plt.subplot(111, projection=ccrs.Robinson())
    ax.coastlines()
    plt.pcolormesh(lons, lats, dA, transform=ccrs.PlateCarree())
    plt.colorbar(shrink=0.7).set_label("Surface Area (m**2)")
    plt.show()

def plot_corrsinalpha(lons,lats, Re, alpha, alpha2):
    plt.figure(figsize=(12,4))
    plt.suptitle("Correction of sin(alpha) * Re")
    ax = plt.subplot(121, aspect='equal', projection=ccrs.Robinson())
    plt.pcolormesh(lons, lats[lats<=80], (Re*np.sin(alpha)), transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.title("Re sin(alpha)")
    plt.colorbar(shrink=0.95, orientation="horizontal")
    ax = plt.subplot(122, aspect='equal', projection=ccrs.Robinson())
    plt.pcolormesh(lons, lats[lats<=80], (Re*np.sin(alpha2)), transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.title("Re sin(min(alpha, pi/2.))")
    plt.colorbar(shrink=0.95, orientation="horizontal")
    plt.show()

def get_distance(configs, dataset, dpt, plot=True):
    target = configs.params.station
    target_lat = configs.params.station_lat
    target_lon = configs.params.station_lon
    lats = dataset.variables['latitude'][:]
    lons = dataset.variables['longitude'][:]

    if not os.path.isfile("DATA/distance_to_%s.csv"%target):
        df = pd.DataFrame(np.zeros((len(lats), len(lons))), index=lats, columns=lons)
        df = dpt.stack(dropna=False).to_frame()
        distances = [locations2degrees(target_lat, target_lon, lat, lon) for (lat, lon), row in df.iterrows()]
        distance_df = pd.DataFrame(distances, index=df.index, columns=["distance"])
        distance_df.to_csv("DATA/distance_to_%s.csv" % target)
    else:
        distance_df = pd.read_csv("DATA/distance_to_%s.csv" % target, index_col=[0,1])

    distance_df = distance_df.unstack()

    if plot:
        cmap = cmr.lavender
        ## Depth map
        dpt.unstack().to_csv("DATA/depth_unstacked.csv")
        plt.figure(figsize=(18,8),)
        ax = plt.subplot(111, projection=ccrs.Robinson())
        plt.title("Depth")
        ax.coastlines(color="w")
        plt.pcolormesh(dpt.columns, dpt.index, dpt, cmap=cmap, transform=ccrs.PlateCarree())
        cb = plt.colorbar(shrink=0.7).set_label("Depth (m)")
        plt.savefig("FIGURES/depth_map.png")

        ## Distance map
        plt.figure(figsize=(18,8),)
        ax = plt.subplot(111, projection=ccrs.Robinson())
        plt.title("Distance to station %s" % target)
        ax.coastlines(color="w")
        plt.pcolormesh(distance_df.columns.get_level_values(1), distance_df.index, distance_df, cmap='viridis', transform=ccrs.PlateCarree(), vmin=0, vmax=180)
        cb = plt.colorbar(shrink=0.7)
        plt.contour(distance_df.columns.get_level_values(1), distance_df.index, distance_df, np.arange(0,180, 20), c="w", cmap='viridis_r', transform=ccrs.PlateCarree())
        plt.colorbar(cax=cb.ax)
        cb.set_label("Distance (deg)")
        plt.savefig("FIGURES/distance to {}.png".format(target))

    return lats, lons, distance_df


def alpha_distance(configs,Re, distance_df):
    rhos = configs.params.rhos
    beta = configs.params.beta
    factor1= np.pi*(1.0/rhos)**2.0/(beta**5*Re)
    # distance_df_degrees = kilometer2degrees(distance_df/1.0e3, radius=Re/1e3)
    distance_df_degrees = distance_df
    distance_df_degrees = distance_df_degrees.loc[-78:83]#RAPH had to adjust the size of the array
    alpha = np.deg2rad(distance_df_degrees.values)
    alpha2 = alpha.copy()
    alpha2[alpha2>=np.pi/2.] = np.pi/2.
    return factor1,alpha, alpha2

def get_ww3(configs, Q, month, lats, lons, Re, dpt, Cf, distance_df, plot=False):
    """
    Computes synthetic seismic spectra at the station in configs.yml
    :param configs: configuration instance
    :param Q: attenuation
    :param month: month of analysis
    :param lats: 
    :param lons: 
    :param Re: Radius of the Earth in m
    :param dpt: DataFrame of the bathymetry
    :param Cf: 
    :param distance_df: DataFrame of teh distance to the station in configs.yml
    :param plot: BOOL to plot the amplification map
    :return: dfF_fs, the synthetic seismic spectra at the station
    """
    beta = configs.params.beta
    CgR = configs.params.Rg
    target = configs.params.station
    P = configs.params.P
    factor1, alpha, alpha2 = alpha_distance(configs,Re, distance_df)
    fn = os.path.join(configs.files.p2l_dir,"LOPS_WW3-GLOB-30M_2021{}_p2l.nc".format(month))
    fname = r"{}".format(fn)
    dataset = netcdf_dataset(fname)
    lats = dataset.variables['latitude'][:]
    lons = dataset.variables['longitude'][:]
    times = dataset.variables['time']
    times = netCDF4.num2date(times[:],times.units)
    freqs = dataset.variables['f'][:] # ocean wave frequency
    #Qf = np.ones(len(freqs)) * Q
    Qf = (0.4+0.4*(1-np.tanh(15*(2*freqs-0.14))))*Q # from IFREMER matlab code "seismic_calculation_synthetic_settings.m"

    F_fs = {}
    try:
        del coeff_all
    except:
        pass
    lats = lats[lats<=80]
    nx = len(lons)
    ny = len(lats)
    ## Elementary surface
    dlon=(lons[-1]-lons[0])/(nx-1)## RAPH find the other coordinates to use here
    dlat=(lats[-1]-lats[0])/(ny-1)
    coslat = numpy.matlib.repmat(np.cos(np.deg2rad(lats)),len(lons),1).T
    dA=Re**2.*coslat*(dlon*dlat*(np.pi/180)**2) # Area of surface element
    
    pbar = tqdm.tqdm(times, desc="Processing %s" % times[0])
#    for p2l in pbar:       
    for ti, t in enumerate(pbar):
        if t in F_fs:
            continue
        pbar.set_description("Processing %s" % t)
#        print("Processing %s" % t)
        if "coeff_all" not in locals():
            print("First time step: Computing attenuation (Q = {}) & amplification".format(int(Q)))
            omega=(2.0*np.pi)*freqs*2  # seismic radian frequency omega=2*2*pi/T
            nf = len(freqs)
            coeff = np.zeros((nf, ny, nx))
            attenuation = np.zeros((nf, ny, nx))

            for fi, fs in enumerate(freqs):
    #            print(fi, fs)
                depth_correction = compute_depth_correction(fs, dpt).fillna(1.0)
                coeff[fi, :, :] = factor1 * omega[fi] * get_C(fs, dpt.loc[-78:80], beta, Cf) * depth_correction.loc[-78:80] / np.sin(alpha2)
                attenuation[fi, :,:] = np.exp(-1.0 * omega[fi] * alpha * Re / (np.abs(CgR)*Qf[fi]))
            dA3D = np.tile(dA[np.newaxis,:,:], (nf,1,1))## RAPH
            coeff_all=coeff*attenuation*dA3D

        P2f = 10.0**(dataset.variables['p2l'][ti, :, :, :]*dataset.variables['p2l'].scale)-(1e-12-1e-16)
    #    source = coeff_all * P2f
        source = coeff_all * P2f[:,:-6,:]
        source = np.reshape(source, (nf, nx*ny))
        F_fs[t] = source.sum(axis=1)

    if plot:
        plot_surfarea(lons,lats,dA, Re)
        plt.figure(figsize=(18,8))
        ax = plt.subplot(111, projection=ccrs.Robinson())
        ax.coastlines(color="w")
        plt.title('Coeff_all')
        plt.pcolormesh(lons, lats, coeff_all[5], vmax=1e-22, transform=ccrs.PlateCarree())
        plt.colorbar(orientation="vertical", shrink=0.7)
        plt.savefig("FIGURES/Coeff_all")
    dfF_fs = pd.DataFrame(F_fs, index=freqs * 2)
    dfF_fs_index = []
    for i, header in enumerate(dfF_fs.columns): dfF_fs_index.append(
        datetime.strptime(str(dfF_fs.columns[i]), '%Y-%m-%d %H:%M:%S'))
    dfF_fs_index
    dfF_fs.columns = pd.DatetimeIndex(pd.DatetimeIndex(dfF_fs_index))
    dfF_fs = dfF_fs.sort_index()
    dfF_fs.to_pickle("DATA/Q/{}_Q{}.pkl".format(target, int(Q)))
    return dfF_fs

def read_p2ls(year, months, p2l_dir):
    times = []
    p2ls = []
    for month in months:
        fname = r"%s/LOPS_WW3-GLOB-30M_%i%02i_p2l.nc"%(p2l_dir, year, month)
        dataset = netcdf_dataset(fname)

        lats = dataset.variables['latitude'][:]
        lons = dataset.variables['longitude'][:]
        currtimes = dataset.variables['time'][:]
        freqs = dataset.variables['f'][:]
        df = np.diff(freqs)
        df = np.append(df[0],df)
        for i, t in enumerate(currtimes):
            mask = dataset.variables['p2l'][i, 0, : , :].mask
            p2l = np.sum([(10**(dataset.variables['p2l'][i, fi, : , :]-1e-12))*df[fi] for fi in range(len(df))], axis=0) #Already in Pa2m2s?, sum all freqs
            p2ls.append(np.ma.array(p2l, mask=mask))
            times.append(t)
    print("Done reading the P2L file(s)")
    return times, p2ls, lats, lons



def get_corr(O, M):
    correlation = (((O-O.mean())*(M-M.mean())).sum())/(np.sqrt(((O-O.mean())**2).sum())*np.sqrt(((M-M.mean())**2).sum()))
    return correlation
def get_misfit(O, M):
    misfit = np.sum(abs(O-M)/abs(O).values)/len(O)
    return misfit

def dfrms(a):
    return np.sqrt(np.trapz(a.values, a.index))

def df_rms(d, freqs, output="DISP"):
    d = d.dropna(axis=1, how='all')
    RMS = {}
    for fmin, fmax in freqs:
        
        ix = np.where((d.columns>=fmin) & (d.columns<=fmax))[0]
        spec = d.iloc[:,ix]
        f = d.columns[ix]
        
        w2f = (2.0 * np.pi * f)

        # The acceleration power spectrum (dB to Power! = divide by 10 and not 20!)
        amp = 10.0**(spec/10.) 
        if output == "ACC":
            RMS["%.1f-%.1f"%(fmin, fmax)] = amp.apply(dfrms, axis=1)
            continue
        
        # velocity power spectrum (divide by omega**2)
        vamp = amp / w2f**2
        if output == "VEL":
            RMS["%.1f-%.1f"%(fmin, fmax)] = vamp.apply(dfrms, axis=1)
            continue
                
        # displacement power spectrum (divide by omega**2)
        damp = vamp / w2f**2
       
        RMS["%.1f-%.1f"%(fmin, fmax)] = damp.apply(dfrms, axis=1)

    return pd.DataFrame(RMS, index=d.index)