from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.visualization import ZScaleInterval
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np
import os

from hciplot import plot_frames, plot_cubes
from vip_hci.var import frame_center
from vip_hci.metrics import completeness_curve, contrast_curve, detection, \
    inverse_stim_map, significance, snr, snrmap, stim_map, throughput
from vip_hci.fm import cube_planet_free, firstguess
from vip_hci.psfsub import pca, pca_annular

import redux_utils
from redux_vip import prep

def plot_redux(name, zscale=True, save=True):

    data_path = "out/%s.fits"%name
    out_path = "out/%s_plot.png"%name

    with fits.open(data_path) as datahdu:
        print(datahdu[1].fileinfo())
        data = np.array(datahdu[0].data)

    xdim = 63
    ydim = 63
    yind, xind = np.mgrid[0:xdim, 0:ydim]
    subdata = data[35:60, 0:25]

    init_amplitude = np.max(subdata)
    init_x_mean = 10
    init_y_mean = 40
    init_cov_matrix = 5 * np.eye(2)
    init_gaus = models.Gaussian2D(amplitude=init_amplitude, x_mean=init_x_mean, y_mean=init_y_mean, cov_matrix=init_cov_matrix)
    fitter = fitting.LevMarLSQFitter()

    gausfit = fitter(init_gaus, xind, yind, data)
    params = [gausfit.amplitude.value, gausfit.x_mean.value, gausfit.y_mean.value, gausfit.x_stddev.value, gausfit.y_stddev.value]
    # print(params)

    bkgd_mean = np.mean(data)
    bkgd_sdev = np.std(data)
    # print(bkgd_mean, bkgd_sdev, bkgd_mean + 5 * bkgd_sdev, (params[0] - bkgd_mean) / bkgd_sdev)

    if zscale:
        zmin, zmax = ZScaleInterval().get_limits(data)
    # >>>> ASK BEN should I keep same or diff scales for each img?
    # else:
    #     zmin = np.min(data)
    #     zmax = np.max(data)

    # --- FIG --- #
    fig = plt.figure(figsize=(15, 5), facecolor='white')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # --- PLOT 1 --- #
    ax1.set_title(name)
    
    ax1_data = data
    if not zscale:
        zmin = np.min(ax1_data)
        zmax = np.max(ax1_data)

    im1 = ax1.imshow(ax1_data, vmin=zmin, vmax=zmax)
    fig.colorbar(im1, ax=ax1)

    # --- PLOT 2 --- #
    datastr = r'$\mu_x = ' + '%.03f$\n'%(params[1]) + r'$\sigma_x = ' + '%.05f$\n'%(params[3]) + \
            r'$\mu_y = ' + '%.03f$\n'%(params[2]) + r'$\sigma_y = ' + '%.05f$\n'%(params[4]) + \
            r'amp = ' + '$%.03f$\n'%(params[0]) + r'$\mu_{bkgd} = ' + '%.05f$\n'%(bkgd_mean) + \
            r'$\sigma_{bkgd} = ' + '%.05f$'%(bkgd_sdev)
    ax2.text(0.6, 0.5, datastr, fontsize=12, transform=ax2.transAxes,
            bbox=dict(facecolor='#f5f5dc', alpha=0.5))
    ax2.set_title("2D Gaussian fit to planet signal")
    
    ax2_data = gausfit(xind, yind)
    if not zscale:
        zmin = np.min(ax2_data)
        zmax = np.max(ax2_data)

    im2 = ax2.imshow(ax2_data, vmin=zmin, vmax=zmax)
    fig.colorbar(im2, ax=ax2)

    # --- PLOT 3 --- #
    ax3.set_title("Residual image")
    ax3_data = data - gausfit(xind, yind)
    if not zscale:
        zmin = np.min(ax3_data)
        zmax = np.max(ax3_data)

    im3 = ax3.imshow(ax3_data, vmin=zmin, vmax=zmax)
    fig.colorbar(im3, ax=ax3)

    # --- FIG --- #
    fig.tight_layout()
    if save:
        fig.savefig(out_path)
    # plt.show()

def calc_stats(pl_data, fwhm):
    pl_loc = (12, 41)
    # st_loc = (63//2, 63//2)
    c_loc = frame_center(pl_data)
    pl_rad = np.sqrt(np.sum(np.square(np.array(c_loc) - np.array(pl_loc))))
    pl_snr = snr(pl_data, pl_loc, fwhm=fwhm, exclude_negative_lobes=True, plot=True)
    pl_sgn = significance(pl_snr, pl_rad, fwhm, student_to_gauss=True)
    print(pl_snr, pl_sgn)

def find_planet(cube: np.ndarray, angles: np.ndarray, psfn: np.ndarray,
        pl_loc: list[np.ndarray], fwhm: float, annulus_width: float=4,
        aperture_radius: float=1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    ncomp = redux_utils.numcomps
    pl_rad, pl_theta, pl_flux = firstguess(cube, angles, psfn, ncomp, pl_loc,
        fwhm, annulus_width, aperture_radius, simplex=True, plot=True,
        verbose=True)

    return pl_rad, pl_theta, pl_flux

def detect_planet(pp_frame, pl_loc, fwhm, plot=False):
    # snr, sig, snrmap, STIM map

    c_loc = frame_center(pp_frame)
    pl_rad = np.sqrt(np.sum(np.square(np.array(c_loc) - np.array(pl_loc))))

    pl_snr = snr(pp_frame, pl_loc, fwhm=fwhm, exclude_negative_lobes=True, plot=plot)
    pl_sgn = significance(pl_snr, pl_rad, fwhm, student_to_gauss=True)
    map_snr = snrmap(pp_frame, fwhm, plot=plot, approximated=False)
    # map_inv_stim = inverse_stim_map
    # map_stim = stim_map

    print(pl_snr, pl_sgn)

    return pl_snr, pl_sgn, map_snr

def ccurves(cubes, angles, psfn, fwhm, pl_loc, ncomp, nbranch, algo=pca):

    cube_0 = cubes[0]
    psfn_0 = psfn[0]
    pxscale = 0.027

    pl_rad, pl_theta, pl_flux = firstguess(cube_0, angles, psfn_0, ncomp, [pl_loc], fwhm=fwhm, simplex=True, plot=True, verbose=True)

    cubes_pf = cube_planet_free([pl_rad, pl_theta, pl_flux], cubes, angles, psfn)

    res_thru = throughput(cubes_pf, angles, psfn, fwhm, ncomp, algo=algo, nbranch=nbranch)
    thru = res_thru[0]
    plt.plot(thru[0])
    plt.show()

    if algo is pca:
        cc = contrast_curve(cubes_pf, angles, psfn, fwhm, pxscale, starphot=pl_flux, algo=pca, sigma=5, nbranch=nbranch, ncomp=ncomp, debug=True)
    elif algo is pca_annular:
        cc = contrast_curve(cubes_pf, angles, psfn, fwhm, pxscale, starphot=pl_flux, algo=pca_annular, sigma=5, nbranch=nbranch, ncomp=ncomp, radius_int=int(fwhm), debug=True)

    return cc


def _ccurves(cube, angles, psfn, fwhm, algo=pca, simplex_data=None, pl_loc=None):

    # an_dist = np.linspace(np.min(angles), np.max(angles), nbranch, endpoint=True)
    # algo_dict = {"ncomp": ncomp, "imlib": "vip-fft", "interpolation": "lanczos4"}

    if simplex_data is None:
        simplex_data = find_planet(cube, angles, psfn, pl_loc, fwhm)
    
    pl_rad, pl_theta, pl_flux = simplex_data
    ncomp = redux_utils.numcomps
    nbranch = 2
    pxscale = 0.027
    starphot = 1000


    c_loc = frame_center(cube[0,0])
    pl_loc = c_loc + pl_rad * np.array([np.cos(pl_theta * np.pi / 180), np.sin(pl_theta * np.pi / 180)])

    
    pp_cube = pca_annular(cube, angles, ncomp=ncomp)
    det = detection(pp_cube, fwhm=fwhm, psf=psfn, bkg_sigma=5, debug=False, mode='log', snr_thresh=5, plot=True, verbose=True)

    # Throughput calculation
    cube_emp = cube_planet_free([pl_rad, pl_theta, pl_flux], cube, angles, psfn)

    
    # pca_emp = pca(cube_emp, angs, ncomp, verbose=True)
    res_thru = throughput(cube_emp, angles, psfn, fwhm, ncomp, algo=algo, nbranch=nbranch)

    
    # ff pca
    cc_ff = contrast_curve(cube_emp, angles, psfn, fwhm, pxscale, starphot=pl_flux, algo=pca, sigma=5, nbranch=nbranch, ncomp=ncomp, debug=True)

    drot = 0.5
    # ann pca
    cc_ann = contrast_curve(cube_emp, angles, psfn, fwhm, pxscale, starphot=pl_flux, algo=pca_annular, sigma=5, nbranch=nbranch, delta_rot=drot, ncomp=ncomp, radius_int=int(fwhm), debug=True)
    
    an_dist = np.linspace(np.min(angles, np.max(angles), nbranch))
    algo_dict = {'ncomp': ncomp, 'imlib': 'opencv'}
    an_dist, comp_curve = completeness_curve(cube_emp, angles, psfn, fwhm, algo, an_dist=an_dist, pxscale=pxscale, ini_contrast=None, starphot=pl_flux, plot=True, nproc=None, algo_dict=algo_dict)

    # include completeness map?

if __name__ == "__main__":

    # names = [
    #     "cADI_median_00010_00090",
    #     "cADI_median_00040_00080",
    #     "cADI_mean_00010_00090",
    #     "cADI_mean_00040_00080",
    #     "cADI_mean_00040_00080_every05",
    #     "cADI_mean_00040_00080_every10",
    #     "cADI_mean_00040_00080_every20",
    #     "cADI_mean_00040_00080_every50",
    #     "vipADI_median_00010_00090",
    #     "vipADI_median_00040_00080",
    #     "vipADI_mean_00010_00090",
    #     "vipADI_mean_00040_00080",
    #     "pyn_PCA003_00040_00080_median",
    #     "pyn_PCA003_00040_00080_median_PREPPED",
    #     "pyn_PCA003_00045_00074_median",
    #     "pyn_PCA003_00045_00074_mean",
    #     "pyn_PCA020_00040_00080_median",
    # ]

    # with mp.Pool(redux_utils.numworkers) as pool:
    #     pool.map(plot_redux, names)

    lib = "vip"
    algo = "PCA"
    sub_type = "single"
    first_chnl = 45
    last_chnl = 74
    nframes = 2202
    channels = list(range(first_chnl, last_chnl))
    frames = range(0, nframes, redux_utils.everynthframe)
    ncomp = redux_utils.numcomps
    nbranch = 1

    data_path = "./data/005_center_multishift/wl_channel_%05i.fits"
    data_paths = [data_path%i for i in channels]
    pp_path = f"./out/{lib}{algo}-{sub_type}_{first_chnl}-{last_chnl}_skip{redux_utils.everynthframe}.fits"
    wavelengths_path = "data/channel_wavelengths.txt"
    angles_path = "data/parangs_bads_removed.txt"

    # cubes, wavelengths, angles = redux_utils.init(data_paths, wavelengths_path,
    #     angles_path, channels=channels, frames=frames)
    
    pp_frame = redux_utils.loadone(pp_path)

    # fwhm, psfn, opt_scal, opt_flux = prep(cubes, wavelengths)

    fwhm = 3.239077

    title = f"{algo} ({sub_type}): $\lambda$={first_chnl}-{last_chnl}, skip {redux_utils.everynthframe}"

    plot_kwargs = {}

    pl_loc = (11, 40)
    pl_snr, pl_sgn, map_snr = detect_planet(pp_frame, pl_loc, fwhm, plot=False)

    out_path = f"out/{algo}-{sub_type}_{first_chnl}-{last_chnl}_skip{redux_utils.everynthframe}.png"
    fig, ax = plot_frames(map_snr, colorbar=True, title=title, return_fig_ax=True, **plot_kwargs)
    datastr = r'S/N = $%.03f$'%(pl_snr) + '\n' + r'sig = $%.03f\sigma$'%(pl_sgn)
    ax.text(0.675, 0.85, datastr, fontsize=12, transform=ax.transAxes,
            bbox=dict(facecolor='#f5f5dc', alpha=0.5))
    fig.savefig(out_path)
    

    '''
    Planet 0: simplex result: (r, theta, f)=(21.403, 154.730, 45.632) at 
          (X,Y)=(11.65, 40.14)
    '''
    # ccurves(cubes, angles, psfn, fwhm, pl_loc, ncomp, nbranch)


