from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.visualization import ZScaleInterval
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from typing import Callable, Union

from hciplot import plot_frames, plot_cubes
from vip_hci.var import frame_center
from vip_hci.metrics import completeness_curve, contrast_curve, detection, \
    inverse_stim_map, significance, snr, snrmap, stim_map, throughput
from vip_hci.fm import cube_planet_free, firstguess
from vip_hci.psfsub import pca, pca_annular


from mypkg.redux import redux_utils
from mypkg.redux.redux_vip import prep

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

def detect_planet(pp_frame, pl_loc, fwhm, use_stim=False, plot=False, out_path: str=None, **plot_kwargs):
    # snr, sig, snrmap, STIM map

    c_loc = frame_center(pp_frame)
    pl_rad = np.sqrt(np.sum(np.square(np.array(c_loc) - np.array(pl_loc))))

    pl_snr = snr(pp_frame, pl_loc, fwhm=fwhm, exclude_negative_lobes=True, plot=plot)
    pl_sgn = significance(pl_snr, pl_rad, fwhm, student_to_gauss=True)
    map_snr = snrmap(pp_frame, fwhm, plot=plot, approximated=False)
    print(f"Potential planet detection at {pl_loc}:\nS/N = {pl_snr}\nSignificance ={pl_sgn}")
    
    if use_stim:
        # map_inv_stim = inverse_stim_map
        # map_stim = stim_map
        print("haven't implemented STIM Map yet")

    if plot or out_path is not None:
        fig, ax = plot_frames(map_snr, colorbar=True, return_fig_ax=True, **plot_kwargs)
        datastr = r'S/N = $%.02f$'%(pl_snr) + '\n' + r'sig = $%.02f\sigma$'%(pl_sgn)
        ax.text(0.6875, 0.85, datastr, fontsize=12, transform=ax.transAxes,
                bbox=dict(facecolor='#f5f5dc', alpha=0.5))
        if out_path is not None:
            fig.savefig(out_path)
        if plot:
            plt.show()


    return pl_snr, pl_sgn, map_snr

def ccurves(cubes: np.ndarray, angles: np.ndarray, psfn: np.ndarray,
            fwhm: float, pl_loc: tuple[float], ncomp: np.ndarray,
            nbranch: int, simplex_data: tuple[np.ndarray],
            algo: str="PCA", sub_type: str="single", pf_path: str=None, name_kwargs: dict={},
            algo_dict: dict={}):

    pxscale = redux_utils.pxscale
    nchnls = cubes.shape[0]
    
    cube_0 = cubes[0]
    psfn_0 = psfn[0]

    if simplex_data is None:
        pl_params = np.array(firstguess(cube_0, angles, psfn_0, ncomp,
            [pl_loc], fwhm=fwhm, simplex=True, plot=True, verbose=True))
    else:
        pl_params = np.array(simplex_data)
    pl_flux = pl_params[2,0]

    #temp
    if pf_path is None or not os.path.isfile(pf_path):
        pl_chnl_params = np.repeat([pl_params], nchnls, axis=2)
        cubes_pf = cube_planet_free(pl_chnl_params, cubes, angles, psfn)
        pf_path = "out/PF_%s.fits"%redux_utils.make_name(**name_kwargs)
        redux_utils.to_fits(cubes_pf, pf_path)
    else:
        cubes_pf = redux_utils.loadone(pf_path)


    if algo == "PCA":

        if sub_type == "annular":
            # res_thru = throughput(cubes_pf, angles, psfn, fwhm, algo=pca_annular,
            #                       ncomp=ncomp, nbranch=nbranch, **algo_dict)
            # thru = res_thru[0]
            # plt.plot(thru[0])
            # plt.show()
            cc = contrast_curve(cubes_pf, angles, psfn, fwhm, pxscale,
                                starphot=pl_flux, algo=pca_annular, sigma=5,
                                nbranch=nbranch, debug=True, **algo_dict)
        else:
            # res_thru = throughput(cubes_pf, angles, psfn, fwhm, algo=pca,
            #                       ncomp=ncomp, nbranch=nbranch, **algo_dict)
            # thru = res_thru[0]
            # plt.plot(thru[0])
            # plt.show()
            cc = contrast_curve(cubes_pf, angles, psfn, fwhm, pxscale,
                                starphot=pl_flux, algo=pca, sigma=5,
                                nbranch=nbranch, debug=True,
                                **algo_dict)
    else:
        cc = None

    return cc

# Run the Contrast Curves the same way as the reductions:
# Prep data
# have a tree of arguments for each algorithm
# run algorithms with the appropriate arg
# perform any additional analysis on curves


"""def _ccurves(cube, angles, psfn, fwhm, algo=pca, simplex_data=None, pl_loc=None):

    # an_dist = np.linspace(np.min(angles), np.max(angles), nbranch, endpoint=True)
    # algo_dict = {"ncomp": ncomp, "imlib": "vip-fft", "interpolation": "lanczos4"}
    an_dist, comp_curve = completeness_curve(cube_emp, angles, psfn, fwhm, algo, an_dist=an_dist, pxscale=pxscale, ini_contrast=None, starphot=pl_flux, plot=True, nproc=None, algo_dict=algo_dict)

    # include completeness map?
"""

if __name__ == "__main__":

    load_cubes = True
    do_prep = True
    do_snr = False
    full_output = True
    mask_rad = 10
    simplex_data = ([21.403], [154.730], [45.632])
    pl_loc = (11.65, 40.14)
    

    lib = "vip"
    algo = "PCA"
    sub_type = "double"
    first_chnl = 45
    last_chnl = 74
    nframes = 2202
    nskip_frames = redux_utils.everynthframe
    ncomp = redux_utils.numcomps
    channels = list(range(first_chnl, last_chnl + 1))
    frames = range(0, nframes, nskip_frames)
    nbranch = 1
    scaling = "temp-standard"
    nproc = redux_utils.numworkers

    name_kwargs = {"lib": lib, "algo": algo, "sub_type": sub_type,
                   "first_chnl": first_chnl, "last_chnl": last_chnl,
                   "ncomp": ncomp, "nskip_frames": nskip_frames}

    data_path = "./data/005_center_multishift/wl_channel_%05i.fits"
    data_paths = [data_path%i for i in channels]
    name = redux_utils.make_name(**name_kwargs)
    pp_path = "out/%s.fits"%name
    pf_path = "out/PF_%s.fits"%name
    wavelengths_path = "data/channel_wavelengths.txt"
    angles_path = "data/parangs_bads_removed.txt"

    if load_cubes:
        cubes, wavelengths, angles = redux_utils.init(data_paths, wavelengths_path,
            angles_path, channels=channels, frames=frames)
        pp_frame = redux_utils.loadone(pp_path)

    if do_prep:
        fwhm, psfn, opt_scal, opt_flux = prep(cubes, wavelengths)
    else:
        fwhm = 3.239077
    

    if do_snr:
        title = f"{algo} ({sub_type}): $\lambda$={first_chnl}-{last_chnl}, skip {nskip_frames}\n S/N Map"
        plot_kwargs = {"title": title}
        out_path = "out/%s.png"%name

        pl_snr, pl_sgn, map_snr = detect_planet(pp_frame, pl_loc, fwhm, plot=False, out_path=out_path)

    '''
    fwhm = 3.239077
    Planet 0: simplex result: (r, theta, f)=(21.403, 154.730, 45.632) at 
          (X,Y)=(11.65, 40.14)
    '''
    # algo_dict = {"imlib": "vip-fft", "interpolation": "lanczos4",
    #              "scale_list": opt_scal, "adimsdi": "single",
    #              "crop_ifs": False, "mask_center_px": mask_rad,
    #              "scaling": scaling, "nproc": nproc, "ncomp": ncomp,
    #              "full_output": full_output}
    
    algo_dict = {"imlib": "vip-fft", "interpolation": "lanczos4",
                "scale_list": opt_scal, "ncomp": (ncomp, ncomp),
                "adimsdi":"double", "crop_ifs": False,
                "mask_center_px": mask_rad, "scaling": scaling, "nproc": nproc,
                "full_output": full_output}
    
    # algo_dict = {"imlib": "vip-fft", "interpolation": "lanczos4",
    #             "scale_list": opt_scal, "ncomp": (ncomp, ncomp),
    #             "asize": fwhm, "n_segments": "auto",
    #             "radius_int": mask_rad, "nproc": nproc,
    #             "full_output": full_output}

    datafr, frame_fc_all, frame_no_fc, fc_map_all = ccurves(cubes, angles, psfn, fwhm, pl_loc, ncomp, nbranch,
            simplex_data, algo=algo, sub_type=sub_type, pf_path=pf_path, name_kwargs=name_kwargs,
            algo_dict=algo_dict)


    out_path0 = "out/datafr_%s.csv"%name
    out_path1 = "out/fc_all_%s.fits"%name
    out_path2 = "out/no_fc_%s.fits"%name
    out_path3 = "out/fc_map_all_%s.fits"%name

    datafr.to_csv(out_path0)
    redux_utils.to_fits(frame_fc_all, out_path1)
    redux_utils.to_fits(frame_no_fc, out_path2)
    redux_utils.to_fits(fc_map_all, out_path3)




