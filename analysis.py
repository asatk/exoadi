from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.visualization import ZScaleInterval
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np

from vip_hci.metrics import significance, snr, snrmap

import redux_utils

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
    st_loc = (63//2, 63//2)
    pl_rad = np.sqrt(np.sum(np.square(np.array(st_loc) - np.array(pl_loc))))
    pl_snr = snr(pl_data, pl_loc, fwhm=fwhm, exclude_negative_lobes=True)
    pl_sgn = significance(pl_snr, pl_rad, fwhm, student_to_gauss=True)
    print(pl_snr, pl_sgn)

if __name__ == "__main__":

    # name = "cADI_mean_00040_00080_every05"

    names = [
        "cADI_median_00010_00090",
        "cADI_median_00040_00080",
        "cADI_mean_00010_00090",
        "cADI_mean_00040_00080",
        "cADI_mean_00040_00080_every05",
        "cADI_mean_00040_00080_every10",
        "cADI_mean_00040_00080_every20",
        "cADI_mean_00040_00080_every50",
        "vipADI_median_00010_00090",
        "vipADI_median_00040_00080",
        "vipADI_mean_00010_00090",
        "vipADI_mean_00040_00080",
        "pyn_PCA003_00040_00080_median",
        "pyn_PCA003_00040_00080_median_PREPPED",
        "pyn_PCA003_00045_00074_median",
        "pyn_PCA003_00045_00074_mean",
        # "pyn_PCA005_00045_00074_median",
        # "pyn_PCA005_00045_00074_mean",
        "pyn_PCA020_00040_00080_median",
    ]

    # names = [
    #     "pyn_PCA005_00045_00074_median",
    #     "pyn_PCA005_00045_00074_mean",
    # ]

    with mp.Pool(redux_utils.numworkers) as pool:
        pool.map(plot_redux, names)
