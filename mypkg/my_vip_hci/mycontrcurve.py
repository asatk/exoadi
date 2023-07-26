"""
Module with contrast curve generation functions.

Modified: 2023.07.23 by Anthony Atkinson

"""

__author__ = "C. Gomez, O. Absil @ ULg, Anthony Atkinson"
__all__ = ["contrast_curve", "noise_per_annulus", "throughput", "aperture_flux"]

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from photutils.aperture import aperture_photometry, CircularAperture
from scipy import stats
from scipy.signal import savgol_filter
from skimage.draw import disk
from typing import Any, Callable
from vip_hci.config import time_ini, timing
from . import cube_inject_companions, frame_inject_companion, normalize_psf
from vip_hci.var import frame_center, dist


def contrast_curve(thru: tuple, fwhm: float, pxscale: float, starphot: float,
        sigma: float=5,student: bool=True, smooth: bool=True):
    
    """Computes the contrast curve at a given confidence (``sigma``) level for
    an ADI cube or ADI+IFS cube. The contrast is calculated as
    sigma*noise/throughput. This implementation takes into account the small
    sample statistics correction proposed in [MAW14]_.

    Parameters
    ----------
    thru: tuple
        Throughput result
    fwhm: int or float
        The the Full Width Half Maximum in pixels.
    pxscale : float
        Plate scale or pixel scale of the instrument (only used for plots)
    starphot : int or float
        If int or float it corresponds to the aperture photometry of the
        non-coronagraphic PSF which we use to scale the contrast. If a vector
        is given it must contain the photometry correction for each frame.
    sigma : int
        Sigma level for contrast calculation. Note this is a "Gaussian sigma"
        regardless of whether Student t correction is performed (set by the
        'student' parameter). E.g. setting sigma to 5 will yield the contrast
        curve corresponding to a false alarm probability of 3e-7.
    student : bool, optional
        If True uses Student t correction to inject fake companion.
    smooth : bool, optional
        If True the radial noise curve is smoothed with a Savitzky-Golay filter
        of order 2.

    Returns
    -------
    datafr : pandas dataframe
        Dataframe containing the sensitivity (Gaussian and Student corrected if
        Student parameter is True), the interpolated throughput, the distance in
        pixels, the noise and the sigma corrected (if Student is True).
    """

    
    vector_radd = thru[3]
    if thru[0].shape[0] > 1:
        thruput_mean = np.nanmean(thru[0], axis=0)
    else:
        thruput_mean = thru[0][0]
    
    rad_samp = vector_radd
    noise_samp = thru[1]
    res_lev_samp = thru[2]
    thruput_interp = thruput_mean
    rad_samp_arcsec = rad_samp * pxscale

    # take abs value of the mean residual fluxes otherwise the more
    # oversubtraction (negative res_lev_samp), the better the contrast!!
    res_lev_samp = np.abs(res_lev_samp)

    if smooth:
        # smoothing the noise vector using a Savitzky-Golay filter
        win = min(noise_samp.shape[0] - 2, int(2 * fwhm))
        if win % 2 == 0:
            win += 1
        noise_samp_sm = savgol_filter(noise_samp, polyorder=2,
            mode="nearest", window_length=win)
        res_lev_samp_sm = savgol_filter(res_lev_samp, polyorder=2,
            mode="nearest", window_length=win)
    else:
        noise_samp_sm = noise_samp
        res_lev_samp_sm = res_lev_samp

    # calculating the contrast
    cont_curve_samp = ((sigma * noise_samp_sm + res_lev_samp_sm) /
        thruput_interp) / starphot
    cont_curve_samp[np.where(cont_curve_samp < 0)] = 1
    cont_curve_samp[np.where(cont_curve_samp > 1)] = 1

    data_dict = {"sensitivity_gaussian": cont_curve_samp,
                 "throughput": thruput_interp,
                 "distance": rad_samp,
                 "distance_arcsec": rad_samp_arcsec,
                 "noise": noise_samp_sm,
                 "residual_level": res_lev_samp_sm}

    # calculating the Student corrected contrast
    if student:
        n_res_els = np.floor(rad_samp / fwhm * 2 * np.pi)
        ss_corr = np.sqrt(1 + 1 / n_res_els)
        sigma_corr = stats.t.ppf(stats.norm.cdf(sigma), n_res_els - 1) * ss_corr
        cont_curve_samp_corr = ((sigma_corr * noise_samp_sm + res_lev_samp_sm)/
            thruput_interp) / starphot
        cont_curve_samp_corr[np.where(cont_curve_samp_corr < 0)] = 1
        cont_curve_samp_corr[np.where(cont_curve_samp_corr > 1)] = 1
        data_dict.update({"sensitivity_student": cont_curve_samp_corr,
                "sigma corr": sigma_corr})


    return pd.DataFrame(data_dict)


# TODO: Include algo_class modifications in any tutorial using this function
def throughput(cube, angle_list, psf_template, fwhm, algo, nbranch=1, theta=0,
    inner_rad=1, fc_rad_sep=3, wedge=(0, 360), fc_snr=100, full_output=False,
    verbose=True, algo_class=None, **algo_dict):
    """Measures the throughput for chosen algorithm and input dataset (ADI or
    ADI+mSDI). The final throughput is the average of the same procedure
    measured in ``nbranch`` azimutally equidistant branches.

    Parameters
    ---------_
    cube : 3d or 4d numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : 1d numpy ndarray
        Vector with parallactic angles.
    psf_template : 2d or 3d numpy ndarray
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    algo : callable or function
        The post-processing algorithm, e.g. vip_hci.pca.pca. Third party Python
        algorithms can be plugged here. They must have the parameters: 'cube',
        'angle_list' and 'verbose'. Optionally a wrapper function can be used.
    nbranch : int optional
        Number of branches on which to inject fakes companions. Each branch
        is tested individually.
    theta : float, optional
        Angle in degrees for rotating the position of the first branch that by
        default is located at zero degrees. Theta counts counterclockwise from
        the positive x axis.
    inner_rad : int, optional
        Innermost radial distance to be considered in terms of FWHM. Should be
        >= 1.
    fc_rad_sep : int optional
        Radial separation between the injected companions (in each of the
        patterns) in FWHM. Must be large enough to avoid overlapping. With the
        maximum possible value, a single fake companion will be injected per
        cube and algorithm post-processing (which greatly affects computation
        time).
    wedge : tuple of floats, optional
        Initial and Final angles for using a wedge. For example (-90,90) only
        considers the right side of an image.
    fc_snr: float optional
        Signal to noise ratio of injected fake companions (w.r.t a Gaussian
        distribution).
    full_output : bool, optional
        If True returns intermediate arrays.
    verbose : bool, optional
        If True prints out timing and information.
    **algo_dict
        Parameters of the post-processing algorithms can be passed here,
        including e.g. ``imlib``, ``interpolation`` or ``nproc``.

    Returns
    -------
    thruput_arr : numpy ndarray
        2d array whose rows are the annulus-wise throughput values for each
        branch.
    vector_radd : numpy ndarray
        1d array with the distances in FWHM (the positions of the annuli).

    If full_output is True then the function returns: thruput_arr, noise,
    vector_radd, cube_fc_all, frame_fc_all, frame_nofc and fc_map_all.

    noise : numpy ndarray
        1d array with the noise per annulus.
    frame_fc_all : numpy ndarray
        3d array with the 3 frames of the 3 (patterns) processed cubes with
        companions.
    frame_nofc : numpy ndarray
        2d array, PCA processed frame without companions.
    fc_map_all : numpy ndarray
        3d array with 3 frames containing the position of the companions in the
        3 patterns.

    """


    # FWHM
    # - L612 min - just to test that the largest sep is not too large
    # - L630 med
    # - L695 norm psf - it appears that norm_psf just spits back out the list
    # - L776 norm psf   of FWHMs that it received. only used in flux calc...
    # - L840 aperture flux (only once, not for other)
    # for normalize_psf, only use of using fwhm is in conj w/ correct_outliers

    array = cube
    parangles = angle_list
    nproc = algo_dict.get("nproc", 1)
    imlib = algo_dict.get("imlib", "vip-fft")
    interpolation = algo_dict.get("interpolation", "lanczos4")
    scaling = algo_dict.get("scaling", None)

    if array.ndim != 3 and array.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")
    else:
        if array.ndim == 3:
            if array.shape[0] != parangles.shape[0]:
                msg = "Input parallactic angles vector has wrong length"
                raise TypeError(msg)
            if psf_template.ndim != 2:
                raise TypeError("Template PSF is not a frame or 2d array")
            maxfcsep = int((array.shape[1] / 2.0) / fwhm) - 1
            if fc_rad_sep < 3 or fc_rad_sep > maxfcsep:
                msg = "Too large separation between companions in the radial "
                msg += "patterns. Should lie between 3 and {}"
                raise ValueError(msg.format(maxfcsep))

        elif array.ndim == 4:
            if array.shape[1] != parangles.shape[0]:
                msg = "Input vector or parallactic angles has wrong length"
                raise TypeError(msg)
            if psf_template.ndim != 3:
                raise TypeError("Template PSF is not a frame, 3d array")
            if "scale_list" not in algo_dict:
                raise ValueError("Vector of wavelength not found")
            else:
                # i think this only matters if diff psf templates are computed
                # one can simply provide the psf temps beforehand.
                if algo_dict["scale_list"].shape[0] != array.shape[0]:
                    raise TypeError("Input wavelength vector has wrong length")
                if isinstance(fwhm, float) or isinstance(fwhm, int):
                    maxfcsep = int((array.shape[2] / 2.0) / fwhm) - 1
                else:
                    maxfcsep = int((array.shape[2] / 2.0) / np.min(fwhm)) - 1
                if fc_rad_sep < 3 or fc_rad_sep > maxfcsep:
                    msg = "Too large separation between companions in the "
                    msg += "radial patterns. Should lie between 3 and {}"
                    raise ValueError(msg.format(maxfcsep))

        if psf_template.shape[1] % 2 == 0:
            raise ValueError("Only odd-sized PSF is accepted")
        if not hasattr(algo, "__call__"):
            raise TypeError("Parameter `algo` must be a callable function")
        if not isinstance(inner_rad, int):
            raise TypeError("inner_rad must be an integer")
        angular_range = wedge[1] - wedge[0]
        if nbranch > 1 and angular_range < 360:
            msg = "Only a single branch is allowed when working on a wedge"
            raise RuntimeError(msg)

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
    # ***************************************************************************
    # Compute noise in concentric annuli on the "empty frame"

    argl = [attr for attr in vars(algo_class)]
    if "cube" in argl and "angle_list" in argl and "verbose" in argl:
        if "fwhm" in argl:
            frame_nofc = algo(cube=array, angle_list=parangles, fwhm=fwhm_med,
                verbose=False, **algo_dict)
            if algo_dict.pop("scaling", None):
                new_algo_dict = algo_dict.copy()
                new_algo_dict["scaling"] = None
                frame_nofc_noscal = algo(cube=array, angle_list=parangles,
                    fwhm=fwhm_med, verbose=False, **new_algo_dict)
            else:
                frame_nofc_noscal = frame_nofc
        else:
            frame_nofc = algo(cube=array, angle_list=parangles, verbose=False,
                **algo_dict)
            if algo_dict.pop("scaling", None):
                new_algo_dict = algo_dict.copy()
                new_algo_dict["scaling"] = None
                frame_nofc_noscal = algo(cube=array, angle_list=parangles,
                    verbose=False, **new_algo_dict)
            else:
                frame_nofc_noscal = frame_nofc

    if verbose:
        msg1 = "Cube without fake companions processed with {}"
        print(msg1.format(algo.__name__))
        timing(start_time)

    noise, res_level, vector_radd = noise_per_annulus(frame_nofc,
        separation=fwhm_med, fwhm=fwhm_med, wedge=wedge)
    if scaling is not None:
        noise_noscal, _, _ = noise_per_annulus(frame_nofc_noscal,
            separation=fwhm_med, fwhm=fwhm_med, wedge=wedge)
    else:
        noise_noscal = noise.copy()

    vector_radd = vector_radd[inner_rad - 1 :]
    noise = noise[inner_rad - 1 :]
    res_level = res_level[inner_rad - 1 :]
    noise_noscal = noise_noscal[inner_rad - 1 :]
    if verbose:
        print("Measured annulus-wise noise in resulting frame")
        timing(start_time)


    # TODO >>>> make this part of prep - we get fwhm from prep to begin; crop.

    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1

    psf_template = normalize_psf(psf_template, fwhm=fwhm_med,
            verbose=verbose, size=min(new_psf_size, psf_template.shape[1]))

    if cube.ndim == 3:
        n, y, x = array.shape

        # Initialize the fake companions
        angle_branch = angular_range / nbranch
        thruput_arr = np.zeros((nbranch, noise.shape[0]))
        fc_map_all = np.zeros((nbranch * fc_rad_sep, y, x))
        frame_fc_all = np.zeros((nbranch * fc_rad_sep, y, x))
        cy, cx = frame_center(array[0])

        # each branch is computed separately
        for br in range(nbranch):
            # each pattern is computed separately. For each one the companions
            # are separated by "fc_rad_sep * fwhm", interleaving the injections
            for irad in range(fc_rad_sep):
                radvec = vector_radd[irad::fc_rad_sep]
                cube_fc = array.copy()
                # filling map with small numbers
                fc_map = np.ones_like(array[0]) * 1e-6
                fcy = []
                fcx = []
                for i in range(radvec.shape[0]):
                    flux = fc_snr * noise_noscal[irad + i * fc_rad_sep]
                    cube_fc = cube_inject_companions(cube_fc, psf_template,
                        parangles, flux, rad_dists=[radvec[i]],
                        theta=br * angle_branch + theta, nproc=nproc,
                        imlib=imlib, interpolation=interpolation,
                        verbose=False)
                    y = cy + radvec[i] * np.sin(np.deg2rad(br * angle_branch + theta))
                    x = cx + radvec[i] * np.cos(np.deg2rad(br * angle_branch + theta))
                    fc_map = frame_inject_companion(fc_map, psf_template, y, x,
                        flux, imlib, interpolation)
                    fcy.append(y)
                    fcx.append(x)

                if verbose:
                    msg2 = "Fake companions injected in branch {} "
                    msg2 += "(pattern {}/{})"
                    print(msg2.format(br + 1, irad + 1, fc_rad_sep))
                    timing(start_time)

                # ***************************************************************

                arg = [attr for attr in vars(algo_class)]
                if "cube" in arg and "angle_list" in arg and "verbose" in arg:
                    if "fwhm" in arg:
                        frame_fc = algo(cube=cube_fc, angle_list=parangles,
                            fwhm=fwhm_med, verbose=False, **algo_dict)
                    else:
                        frame_fc = algo(cube=cube_fc, angle_list=parangles,
                            verbose=False, **algo_dict)
                else:
                    msg = "Input algorithm must have at least 3 parameters: "
                    msg += "cube, angle_list and verbose"
                    raise ValueError(msg)

                if verbose:
                    msg3 = "Cube with fake companions processed with {}"
                    msg3 += "\nMeasuring its annulus-wise throughput"
                    print(msg3.format(algo.__name__))
                    timing(start_time)

                # **************************************************************
                injected_flux = aperture_flux(fc_map, fcy, fcx, fwhm_med)
                recovered_flux = aperture_flux((frame_fc - frame_nofc), fcy,
                    fcx, fwhm_med)
                thruput = recovered_flux / injected_flux
                thruput[np.where(thruput < 0)] = 0

                thruput_arr[br, irad::fc_rad_sep] = thruput
                fc_map_all[br * fc_rad_sep + irad, :, :] = fc_map
                frame_fc_all[br * fc_rad_sep + irad, :, :] = frame_fc

    # normalize psf:
    # - crops template psf to 3fwhm
    # - force odd!!!

    elif cube.ndim == 4:
        w, n, y, x = array.shape
        if isinstance(fwhm, (int, float)):
            fwhm = [fwhm] * w

        # Initialize the fake companions
        angle_branch = angular_range / nbranch
        thruput_arr = np.zeros((nbranch, noise.shape[0]))
        fc_map_all = np.zeros((nbranch * fc_rad_sep, w, y, x))
        frame_fc_all = np.zeros((nbranch * fc_rad_sep, y, x))
        c = np.array(frame_center(array[0, 0]))

        # TODO >>>> multiprocessing - it says it is computed separately

        # each branch is computed separately
        for br in range(nbranch):
            # each pattern is computed separately. For each pattern the
            # companions are separated by "fc_rad_sep * fwhm"
            # radius = vector_radd[irad::fc_rad_sep]
            for irad in range(fc_rad_sep):
                radvec = vector_radd[irad::fc_rad_sep]
                len_radvec = radvec.shape[0]
                thetavec = range(int(theta), int(theta) + 360, 360 // len_radvec)
                cube_fc = array.copy()
                # filling map with small numbers
                fc_map = np.ones_like(array[:, 0]) * 1e-6
                fc_yxs = np.zeros(shape=(len_radvec, 2), dtype=np.float64)
                
                for i in range(len_radvec):
                    flux = fc_snr * noise_noscal[irad + i * fc_rad_sep]
                    cube_fc = cube_inject_companions(cube_fc, psf_template,
                        parangles, flux, rad_dists=[radvec[i]],
                        theta=thetavec[i], verbose=False, imlib=imlib,
                        interpolation=interpolation, nproc=nproc)
                    
                    ang_arg = np.deg2rad(br * angle_branch + thetavec[i])
                    fc_yxs[i] = c + np.multiply(radvec[i], [np.sin(ang_arg),
                        np.cos(ang_arg)])
                    fc_map = frame_inject_companion(fc_map, psf_template,
                        fc_yxs[i, 0], fc_yxs[i, 1], flux)

                if verbose:
                    msg2 = "Fake companions injected in branch {} "
                    msg2 += "(pattern {}/{})"
                    print(msg2.format(br + 1, irad + 1, fc_rad_sep))
                    timing(start_time)

                # **************************************************************

                arg = [attr for attr in vars(algo_class)]
                if "cube" in arg and "angle_list" in arg and "verbose" in arg:
                    if "fwhm" in arg:
                        frame_fc = algo(cube=cube_fc, angle_list=parangles,
                            fwhm=fwhm_med, verbose=False, **algo_dict)
                    else:
                        frame_fc = algo(cube=cube_fc, angle_list=parangles,
                            verbose=False, **algo_dict)
                else:
                    print("Complain... `algo` doesn't have necessary params"+\
                          " so no frame_fc will be returned")

                if verbose:
                    msg3 = "Cube with fake companions processed with {}"
                    msg3 += "\nMeasuring its annulus-wise throughput"
                    print(msg3.format(algo.__name__))
                    timing(start_time)

                # *************************************************************
                injected_flux = [aperture_flux(fc_map[i], fc_yxs[:, 0], fc_yxs[:, 1], fwhm[i])
                    for i in range(array.shape[0])]
                injected_flux = np.mean(injected_flux, axis=0)
                recovered_flux = aperture_flux(
                    (frame_fc - frame_nofc), fc_yxs[:, 0], fc_yxs[:, 1], fwhm_med)
                thruput = recovered_flux / injected_flux
                thruput[np.where(thruput < 0)] = 0

                thruput_arr[br, irad::fc_rad_sep] = thruput
                fc_map_all[br * fc_rad_sep + irad, :, :] = fc_map
                frame_fc_all[br * fc_rad_sep + irad, :, :] = frame_fc

    if verbose:
        msg = "Finished measuring the throughput in {} branches"
        print(msg.format(nbranch))
        timing(start_time)

    if full_output:
        return (thruput_arr, noise, res_level, vector_radd, frame_fc_all,
            frame_nofc, fc_map_all)
    else:
        return thruput_arr, vector_radd


def _find_coords(rad, sep, init_angle, fin_angle):
    angular_range = fin_angle - init_angle
    npoints = (np.deg2rad(angular_range) * rad) / sep  # (2*np.pi*rad)/sep
    ang_step = angular_range / npoints  # 360/npoints
    trig_arg = np.deg2rad(np.arange(int(npoints)) * ang_step + init_angle)
    yx = rad * np.array([np.sin(trig_arg), np.cos(trig_arg)])
    return yx


def noise_per_annulus(array, separation, fwhm, init_rad=None, wedge=(0, 360),
                      verbose=False, debug=False):
    """Measures the noise and mean residual level as the standard deviation
    and mean, respectively, of apertures defined in each annulus with a given
    separation.

    The annuli start at init_rad (= fwhm by default if not provided) and stop
    2*separation before the edge of the frame.

    Parameters
    ----------
    array : numpy ndarray
        Input frame.
    separation : float
        Separation in pixels of the centers of the annuli measured from the
        center of the frame.
    fwhm : float
        FWHM in pixels.
    init_rad : float
        Initial radial distance to be used. If None then the init_rad = FWHM.
    wedge : tuple of floats, optional
        Initial and Final angles for using a wedge. For example (-90,90) only
        considers the right side of an image. Be careful when using small
        wedges, this leads to computing a standard deviation of very small
        samples (<10 values).
    verbose : bool, optional
        If True prints information.
    debug : bool, optional
        If True plots the positioning of the apertures.

    Returns
    -------
    noise : numpy ndarray
        Vector with the noise value per annulus.
    res_level : numpy ndarray
        Vector with the mean residual level per annulus.
    vector_radd : numpy ndarray
        Vector with the radial distances values.

    """


    ###

    if array.ndim != 2:
        raise TypeError("Input array is not a frame or 2d array")
    if not isinstance(wedge, tuple):
        raise TypeError("Wedge must be a tuple with the initial and final " + \
                        "angles")

    if init_rad is None:
        init_rad = fwhm

    init_angle, fin_angle = wedge
    centery, centerx = frame_center(array)
    n_annuli = int(np.floor((centery - init_rad) / separation)) - 1
    noise = []
    res_level = []
    vector_radd = []
    if verbose:
        print("{} annuli".format(n_annuli))

    if debug:
        _, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(
            array, origin="lower", interpolation="nearest", alpha=0.5, cmap="gray"
        )

    for i in range(n_annuli):
        y = centery + init_rad + separation * i
        rad = dist(centery, centerx, y, centerx)
        yy, xx = _find_coords(rad, fwhm, init_angle, fin_angle)
        yy += centery
        xx += centerx

        apertures = CircularAperture(np.array((xx, yy)).T, fwhm / 2)
        fluxes = aperture_photometry(array, apertures)["aperture_sum"]

        noise_ann = np.std(fluxes)
        mean_ann = np.mean(fluxes)
        noise.append(noise_ann)
        res_level.append(mean_ann)
        vector_radd.append(rad)

        if debug:
            for j in range(xx.shape[0]):
                # Circle takes coordinates as (X,Y)
                aper = plt.Circle(
                    (xx[j], yy[j]), radius=fwhm / 2, color="r", fill=False, alpha=0.8
                )
                ax.add_patch(aper)
                cent = plt.Circle(
                    (xx[j], yy[j]), radius=0.8, color="r", fill=True, alpha=0.5
                )
                ax.add_patch(cent)

        if verbose:
            print("Radius(px) = {}, Noise = {:.3f} ".format(rad, noise_ann))

    return np.array(noise), np.array(res_level), np.array(vector_radd)


def aperture_flux(array, yc, xc, fwhm, ap_factor=1, mean=False, verbose=False):
    """Returns the sum of pixel values in a circular aperture centered on the
    input coordinates. The radius of the aperture is set as (ap_factor*fwhm)/2.

    Parameters
    ----------
    array : numpy ndarray
        Input frame.
    yc, xc : list or 1d arrays
        List of y and x coordinates of sources.
    fwhm : float
        FWHM in pixels.
    ap_factor : int, optional
        Diameter of aperture in terms of the FWHM.

    Returns
    -------
    flux : list of floats
        List of fluxes.

    Note
    ----
    From Photutils documentation, the aperture photometry defines the aperture
    using one of 3 methods:

    'center': A pixel is considered to be entirely in or out of the aperture
              depending on whether its center is in or out of the aperture.
    'subpixel': A pixel is divided into subpixels and the center of each
                subpixel is tested (as above).
    'exact': (default) The exact overlap between the aperture and each pixel is
             calculated.

    """
    n_obj = len(yc)
    flux = np.zeros((n_obj))
    ap_rad = (ap_factor * fwhm) / 2
    for i, (y, x) in enumerate(zip(yc, xc)):
        if mean:
            ind = disk((y, x), ap_rad)
            values = array[ind]
            obj_flux = np.mean(values)
        else:
            aper = CircularAperture((x, y), ap_rad)
            obj_flux = aperture_photometry(array, aper, method="exact")
            obj_flux = np.array(obj_flux["aperture_sum"])
        flux[i] = obj_flux

        if verbose:
            print("Coordinates of object {} : ({},{})".format(i, y, x))
            print("Object Flux = {:.2f}".format(flux[i]))

    return flux
