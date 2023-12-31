"""
Module with fake companion injection functions.

Modified: 2023.07.24
"""

__author__ = "Carlos Alberto Gomez Gonzalez, Valentin Christiaens, Anthony Atkinson"
__all__ = ["collapse_psf_cube",
           "normalize_psf",
           "cube_inject_companions",
           "generate_cube_copies_with_injections",
           "frame_inject_companion"]

import numpy as np
from numpy import ndarray
from photutils.aperture import aperture_photometry, CircularAperture
from photutils.centroids import centroid_com as cen_com
from scipy import stats
from typing import Callable
from vip_hci.config.utils_conf import print_precision, check_array, pool_map, \
    iterable
from vip_hci.preproc import cube_crop_frames, cube_shift, frame_crop, \
    frame_shift
from vip_hci.var import frame_center, fit_2dairydisk, fit_2dgaussian, \
    fit_2dmoffat, get_annulus_segments, get_circle

def _cube_inject_adi(array: ndarray, psf_template: ndarray, angle_list: ndarray,
                     flevel: ndarray, plsc: float, rad_dists: ndarray,
                     n_branches: int=1, theta: float=0, imlib: str='vip-fft',
                     interpolation: str='lanczos4', verbose: bool=False,
                     nproc: int=1):
    
    if np.isscalar(flevel):
        flevel = np.ones_like(angle_list)*flevel

    # ceny, cenx = frame_center(array[0])
    cenyx = np.array(frame_center(array[0]))
    sizex = array.shape[-1]
    sizey = array.shape[-2]
    nframes = array.shape[-3]
    n_rad = len(rad_dists)
    
    size_fc = psf_template.shape[-1]

    w = int(np.ceil(size_fc/2)) #fake companion width?
    if size_fc % 2:  # new convention
        w -= 1
    
    start_px = cenyx.astype(int) - w
    # sty = int(ceny) - w
    # stx = int(cenx) - w

    positions = []
    # fake companion cube
    if psf_template.ndim == 2:
        fc_fr = np.repeat(psf_template, nframes)
    else:
        fc_fr = psf_template[:nframes]

    array_out = array.copy()


    angs = (np.arange(n_branches) * 2 * np.pi / n_branches) + np.deg2rad(theta)
    # test these
    temp = np.outer([np.sin(angs), np.cos(angs)], rad_dists).flatten()
    positions = np.stack((temp[n_branches * n_rad:], temp[:n_branches * n_rad]), axis=1) + cenyx
    # posns_y = cenyx[0] + np.outer(rad_dists, np.sin(angs)).reshape(-1)
    # posns_x = cenyx[1] + np.outer(rad_dists, np.cos(angs)).reshape(-1)
    # positions = np.stack((posns_y, posns_x), axis=1)


    angs_arg = (angs - np.repeat([np.deg2rad(angle_list)], n_branches, axis=0).T).T
    # shifts_y = np.outer(rad_dists, np.sin(angs_arg))
    # shifts_x = np.outer(rad_dists, np.cos(angs_arg))

    # shifts_yx = rad * np.array([np.sin(ang_arg), np.cos(ang_arg)])
    temp = np.outer([np.sin(angs_arg), np.cos(angs_arg)], rad_dists).flatten()
    shifts_yx = np.stack((temp[n_branches * nframes:], temp[:n_branches * nframes]), axis=1)

    # integer shift (in final cube)
    yx0 = start_px + shifts_yx.astype(int)
    yxN = yx0 + size_fc
    p_yx0 = np.zeros_like(yx0)
    p_yxN = p_yx0 + size_fc
    
    idx0 = p_yx0 < 0
    p_yx0[:, 0][idx0[:, 0]] = -yx0[:, 0][idx0[:, 0]]
    yx0[:, 0][idx0[:, 0]] = 0

    p_yx0[:, 1][idx0[:, 1]] = -yx0[:, 1][idx0[:, 1]]
    yx0[:, 1][idx0[:, 1]] = 0

    size_arr = np.array([sizey, sizex])
    idxN = p_yxN > size_arr
    p_yxN[:, 0][idxN[:, 0]] -= yxN[:, 0][idxN[:, 0]] - size_arr[0]
    yxN[:, 0][idxN[:, 0]] = size_arr[0]

    p_yxN[:, 1][idxN[:, 1]] -= yxN[:, 1][idxN[:, 1]] - size_arr[1]
    yxN[:, 1][idxN[:, 1]] = size_arr[1]


    # sub-px shift (within PSF template frame)
    dsyx = shifts_yx - shifts_yx.astype(int)

    for branch in range(n_branches):
        # ang = angs[branch]

        if verbose:
            print('Branch {}:'.format(branch+1))

        for i, rad in enumerate(rad_dists):
            array_sh = np.zeros(shape=(nframes, *array.shape))

            fc_fr_rad = fc_fr.copy()
            # this can be MP'd
            if nproc > 1:
                pool_res = pool_map(nproc, frame_shift, iterable(fc_fr_rad,...))
            for fr in range(nframes):
                yx_idx = branch * nframes + fr
                yx0_fr = yx0[yx_idx]
                yxN_fr = yxN[yx_idx]
                p_yx0_fr = p_yx0[yx_idx]
                p_yxN_fr = p_yxN[yx_idx]
                dsyx_fr = dsyx[yx_idx]

                if nproc == 1:
                    fc_fr_rad = frame_shift(fc_fr_rad[fr], dsyx_fr[0], dsyx_fr[1],
                        imlib, interpolation, border_mode='constant')
                else:
                    fc_fr_rad = pool_res[fr]
                
                array_sh[fr, yx0_fr[0]:yxN_fr[0], yx0_fr[1]:yxN_fr[1]] = \
                    flevel[fr]*fc_fr_rad[p_yx0_fr[0]:p_yxN_fr[0], p_yx0_fr[1]:p_yxN_fr[1]]

            array_out += array_sh

            pos_y = positions[branch * len(rad_dists) + i, 0]
            pos_x = positions[branch * len(rad_dists) + i, 1]

            if verbose:
                rad_arcs = rad * plsc
                print("\t(X,Y)=({:.2f}, {:.2f}) at {:.2f} arcsec "
                        "({:.2f} pxs from center)".format(
                            pos_x, pos_y, rad_arcs, rad))

    return array_out, positions

def cube_inject_companions(array, psf_template, angle_list, flevel, rad_dists,
        plsc=None, n_branches=1, theta=0, imlib='vip-fft',
        interpolation='lanczos4', full_output=False, verbose=False, nproc: int=1):
    """ Injects fake companions in branches, at given radial distances.

    Parameters
    ----------
    array : 3d/4d numpy ndarray
        Input cube. This is copied before the injections take place, so
        ``array`` is never modified.
    psf_template : 2d/3d numpy ndarray
        [for a 3D input array] 2d array with the normalized PSF template, with
        an odd or even shape. The PSF image must be centered wrt to the array.
        Therefore, it is recommended to run the function ``normalize_psf`` to
        generate a centered and flux-normalized PSF template.
        It can also be a 3D array, but length should match that of ADI cube.
        [for a 4D input array] In the ADI+mSDI case, it must be a 3d array
        (matching spectral dimensions).
    angle_list : 1d numpy ndarray
        List of parallactic angles, in degrees.
    flevel : float or 1d array or 2d array
        Factor for controlling the brightness of the fake companions. If a float,
        the same flux is used for all injections.
        [3D input cube]: if a list/1d array is provided, it should have same
        length as number of frames in the 3D cube (can be used to take into
        account varying observing conditions or airmass).
        [4D (ADI+mSDI) input cube]: if a list/1d array should have the same
        length as the number of spectral channels (i.e. provide a spectrum). If
        a 2d array, it should be n_wavelength x n_frames (can e.g. be used to
        inject a spectrum in varying conditions).
    plsc : float or None
        Value of the plsc in arcsec/px. Only used for printing debug output when
        ``verbose=True``.
    rad_dists : float, list or array 1d
        Vector of radial distances of fake companions in pixels.
    n_branches : int, optional
        Number of azimutal branches.
    theta : float, optional
        Angle in degrees for rotating the position of the first branch that by
        default is located at zero degrees. Theta counts counterclockwise from
        the positive x axis.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    full_output : bool, optional
        Returns the ``x`` and ``y`` coordinates of the injections, additionally
        to the new array.
    verbose : bool, optional
        If True prints out additional information.
    nproc: int, optional
        Number of CPUs to use for multiprocessing. If None, will be 
        automatically set to half the number of available CPUs.

    Returns
    -------
    array_out : numpy ndarray
        Output array with the injected fake companions.
    positions : list of tuple(y, x)
        [full_output] Coordinates of the injections in the first frame (and
        first wavelength for 4D cubes).
    psf_trans: numpy ndarray
        [full_output & transmission != None] Array with injected psf affected
        by transmission (serves to check radial transmission)

    """

    check_array(array, dim=(3, 4), msg="array")
    check_array(psf_template, dim=(2, 3), msg="psf_template")

    nframes = array.shape[-3]

    if array.ndim == 4 and psf_template.ndim != 3:
        raise ValueError('`psf_template` must be a 3d array')

    if verbose and not np.isscalar(plsc):
        raise TypeError("`plsc` must be a scalar")
    if not np.isscalar(flevel) and len(flevel) != array.shape[0]:
            msg = "if not scalar `flevel` must have same length as array"
            raise TypeError(msg)

    rad_dists = np.asarray(rad_dists).reshape(-1)  # forces ndim=1

    if rad_dists[-1] >= array.shape[-1] / 2:
        raise ValueError('rad_dists last location is at the border (or '
                         'outside) of the field')

    # ADI case
    if array.ndim == 3:
        res = _cube_inject_adi(array, psf_template, angle_list, flevel, plsc,
                               rad_dists, n_branches, theta, imlib,
                               interpolation, verbose, nproc=nproc)
        array_out, positions = res

    # ADI+mSDI (IFS) case
    else:
        nframes_wav = array.shape[0]
        array_out = np.empty_like(array)
        if np.isscalar(flevel):
            flevel_all = flevel * np.ones([nframes_wav, nframes])
        elif flevel.ndim == 1:
            flevel_all = np.repeat(flevel, nframes_wav)
        else:
            flevel_all = flevel
        for i in range(nframes_wav):
            if verbose:
                msg = "*** Processing spectral channel {}/{} ***"
                print(msg.format(i+1, nframes_wav))
            res = _cube_inject_adi(array[i], psf_template[i], angle_list,
                                   flevel_all[i], plsc, rad_dists, n_branches,
                                   theta, imlib, interpolation,
                                   verbose=(i == 0 & verbose is True), nproc=nproc)
            array_out[i], positions = res

    if full_output:
        return array_out, positions
    else:
        return array_out


def generate_cube_copies_with_injections(array, psf_template, angle_list, plsc,
        n_copies=100, inrad=8, outrad=12, dist_flux=("uniform", 2, 500)):
    """
    Create multiple copies of ``array`` with different random injections.

    This is a wrapper around ``metrics.cube_inject_companions``, which deals
    with multiple copies of the original data cube and generates random
    parameters.

    Parameters
    ----------
    array : 3d/4d numpy ndarray
        Original input cube.
    psf_template : 2d/3d numpy ndarray
        Array with the normalized psf template. It should have an odd shape.
        It's recommended to run the function ``normalize_psf`` to get a proper
        PSF template. In the ADI+mSDI case it must be a 3d array.
    angle_list : 1d numpy ndarray
        List of parallactic angles, in degrees.
    plsc : float
        Value of the plsc in arcsec/px. Only used for printing debug output when
        ``verbose=True``.
    n_copies : int
        This is the number of 'cube copies' returned.
    inrad,outrad : float
        Inner and outer radius of the injections. The actual injection position
        is chosen randomly.
    dist_flux : tuple('method', params)
        Tuple describing the flux selection. Method can be a function, the
        ``*params`` are passed to it. Method can also be a string, for a
        pre-defined random function:

            ``('skewnormal', skew, mean, var)``
                uses scipy.stats.skewnorm.rvs
            ``('uniform', low, high)``
                uses np.random.uniform
            ``('normal', loc, scale)``
                uses np.random.normal

    Returns
    -------
    fake_data : dict
        Represents a copy of the original ``array``, with fake injections. The
        dictionary keys are:

            ``cube``
                Array shaped like the input ``array``, with the fake injections.
            ``position`` : list of tuples(y,x)
                List containing the positions of the injected companions, as
                (y,x) tuples.
            ``dist`` : float
                The distance of the injected companions, which was passed to
                ``cube_inject_companions``.
            ``theta`` : float, degrees
                The initial angle, as passed to ``cube_inject_companions``.
            ``flux`` : float
                The flux passed to ``cube_inject_companions``.

    """
    # TODO: 'mask' parameter for known companions?

    width = outrad - inrad
    yy, xx = get_annulus_segments(array[0], inrad, width)[0]
    num_patches = yy.shape[0]

    # Defining Fluxes according to chosen distribution
    dist_fkt = dict(skewnormal=stats.skewnorm.rvs,
                    normal=np.random.normal,
                    uniform=np.random.uniform).get(dist_flux[0],
                                                   dist_flux[0])
    fluxes = sorted(dist_fkt(*dist_flux[1:], size=n_copies))

    inds_inj = np.random.randint(0, num_patches, size=n_copies)

    # Injections
    for n in range(n_copies):

        injx = xx[inds_inj[n]] - frame_center(array[0])[1]
        injy = yy[inds_inj[n]] - frame_center(array[0])[0]
        dist = np.sqrt(injx**2 + injy**2)
        theta = np.mod(np.arctan2(injy, injx) / np.pi * 180, 360)

        # TODO: multiple injections?
        fake_cube, positions = cube_inject_companions(array, psf_template,
            angle_list, plsc=plsc, flevel=fluxes[n], theta=theta,
            rad_dists=dist, n_branches=1, full_output=True, verbose=False)

        yield dict(positions=positions, dist=dist, theta=theta, flux=fluxes[n],
            cube=fake_cube)


def frame_inject_companion(array, array_fc, pos_y, pos_x, flux,
                           imlib='vip-fft', interpolation='lanczos4'):
    """ 
    Injects a fake companion in a single frame (it could be a single
    multi-wavelength frame) at given coordinates, or in a cube (at the same
    coordinates, flux and with same fake companion image throughout the cube).

    Parameters
    ----------
    array : numpy ndarray, 2d or 3d
        Input frame or cube.
    array_fc : numpy ndarray, 2d
        Fake companion image to be injected. If even-dimensions, the center
        should be placed at coordinates [dim//2, dim//2] (0-based indexing),
        as per VIP's convention.
    pos_y, pos_x: float
         Y and X coordinates where the companion should be injected
    flux : int
        Flux at which the fake companion should be injected (i.e. scaling
        factor for the injected image)
    imlib : str, optional
        See documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See documentation of the ``vip_hci.preproc.frame_shift`` function.

    Returns
    -------
    array_out : numpy ndarray
        Frame or cube with the companion injected
    """
    if not (array.ndim == 2 or array.ndim == 3):
        raise TypeError('Array is not a 2d or 3d array.')
    if array.ndim == 2:
        size_fc = array_fc.shape[0]
        ceny, cenx = frame_center(array)
        ceny = int(ceny)
        cenx = int(cenx)
        fc_fr = np.zeros_like(array)
        w = int(np.floor(size_fc/2.))
        odd = size_fc % 2
        # fake companion in the center of a zeros frame
        fc_fr[ceny-w:ceny+w+odd, cenx-w:cenx+w+odd] = array_fc
        array_out = array + frame_shift(fc_fr, pos_y-ceny, pos_x-cenx, imlib,
                                        interpolation) * flux

    if array.ndim == 3:
        size_fc = array_fc.shape[1]
        ceny, cenx = frame_center(array[0])
        ceny = int(ceny)
        cenx = int(cenx)
        fc_fr = np.zeros_like(array)
        w = int(np.floor(size_fc/2.))
        odd = size_fc % 2
        # fake companion in the center of a zeros frame
        fc_fr[:, ceny-w:ceny+w+odd, cenx-w:cenx+w+odd] = array_fc
        array_out = array + cube_shift(fc_fr, pos_y - ceny, pos_x - cenx,
                                       imlib, interpolation) * flux

    return array_out


def collapse_psf_cube(array, size, fwhm=4, verbose=True, collapse='mean'):
    """ Creates a 2d PSF template from a cube of non-saturated off-axis frames
    of the star by taking the mean and normalizing the PSF flux.

    Parameters
    ----------
    array : numpy ndarray, 3d
        Input cube.
    size : int
        Size of the squared subimage.
    fwhm: float, optional
        The size of the Full Width Half Maximum in pixel.
    verbose : {True,False}, bool optional
        Whether to print to stdout information about file opening, cropping and
        completion of the psf template.
    collapse : {'mean','median'}, string optional
        Defines the way the frames are collapsed.

    Returns
    -------
    psf_normd : numpy ndarray
        Normalized PSF.
    """
    if array.ndim != 3 and array.ndim != 4:
        raise TypeError('Array is not a cube, 3d or 4d array.')

    n = array.shape[0]
    psf = cube_crop_frames(array, size=size, verbose=verbose)
    if collapse == 'mean':
        psf = np.mean(psf, axis=0)
    elif collapse == 'median':
        psf = np.median(psf, axis=0)
    else:
        raise TypeError('Collapse mode not recognized.')

    psf_normd = normalize_psf(psf, size=size, fwhm=fwhm)

    if verbose:
        print("Done scaled PSF template from the average of", n, "frames.")
    return psf_normd


def _psf_norm_2d(psf, fwhm, threshold, mask_core, full_output, verbose,
        fit_2d: Callable, imlib="vip-fft", interpolation="lanczos4"):
    # lacking interp, imlib, fit2d
    """ 2d case """
    # we check if the psf is centered and fix it if needed
    cy, cx = frame_center(psf, verbose=False)
    xcom, ycom = cen_com(psf)
    if not (np.allclose(cy, ycom, atol=1e-2) or
            np.allclose(cx, xcom, atol=1e-2)):
        # first we find the centroid and put it in the center of the array
        centry, centrx = fit_2d(psf, full_output=False, debug=False)
        if not np.isnan(centry) and not np.isnan(centrx):
            shiftx, shifty = centrx - cx, centry - cy
            psf = frame_shift(psf, -shifty, -shiftx, imlib=imlib,
                                interpolation=interpolation)

            for _ in range(2):
                centry, centrx = fit_2d(psf, full_output=False, debug=False)
                if np.isnan(centry) or np.isnan(centrx):
                    break
                cy, cx = frame_center(psf, verbose=False)
                shiftx, shifty = centrx - cx, centry - cy
                psf = frame_shift(psf, -shifty, -shiftx, imlib=imlib,
                                    interpolation=interpolation)

    # we check whether the flux is normalized and fix it if needed
    fwhm_aper = CircularAperture((cx, cy), fwhm/2)
    fwhm_aper_phot = aperture_photometry(psf, fwhm_aper,
                                            method='exact')
    fwhm_flux = np.array(fwhm_aper_phot['aperture_sum'])

    if fwhm_flux > 1.1 or fwhm_flux < 0.9:
        psf_norm_array = psf / np.array(fwhm_aper_phot['aperture_sum'])
    else:
        psf_norm_array = psf

    if threshold is not None:
        psf_norm_array[np.where(psf_norm_array < threshold)] = 0

    if mask_core is not None:
        psf_norm_array = get_circle(psf_norm_array, radius=mask_core)

    if verbose:
        print("Flux in 1xFWHM aperture: {:.3f}".format(fwhm_flux[0]))

    if full_output:
        return psf_norm_array, fwhm_flux, fwhm
    else:
        return psf_norm_array


def normalize_psf(array, fwhm='fit', size=None, threshold=None, mask_core=None,
                  model='gauss', imlib='vip-fft', interpolation='lanczos4',
                  force_odd=True, correct_outliers=True, full_output=False,
                  verbose=True, debug=False):
    """ Normalizes a PSF (2d or 3d array), to have the flux in a 1xFWHM
    aperture equal to one. It also allows to crop the array and center the PSF
    at the center of the array(s).

    Parameters
    ----------
    array: numpy ndarray
        The PSF, 2d (ADI data) or 3d array (IFS data).
    fwhm: int, float, 1d array or str, optional
        The Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data). If set to 'fit' then
        a ``model`` (assuming the PSF is centered in the array) is fitted to
        estimate the FWHM in 2D or 3D PSF arrays.
    size : int or None, optional
        If int it will correspond to the size of the centered sub-image to be
        cropped form the PSF array. The PSF is assumed to be roughly centered wrt
        the array.
    threshold : None or float, optional
        Sets to zero values smaller than threshold (in the normalized image).
        This can be used to only leave the core of the PSF.
    mask_core : None or float, optional
        Sets the radius of a circular aperture for the core of the PSF,
        everything else will be set to zero.
    model : {'gauss', 'moff', 'airy'}, str optional
        The assumed model used to fit the PSF: either a Gaussian, a Moffat
        or an Airy 2d distribution.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    force_odd : bool, optional
        If True the resulting array will have odd size (and the PSF will be
        placed at its center). If False, and the frame size is even, then the
        PSF will be put at the center of an even-sized frame.
    correct_outliers: bool, optional
        For an input 3D cube (IFS) of PSFs, if the 2D fit fails for one of the
        channels, whether to interpolate FWHM value from surrounding channels,
        and recalculate flux and normalization.
    full_output : bool, optional
        If True the flux in a FWHM aperture is returned along with the
        normalized PSF.
    verbose : bool, optional
        If True intermediate results are printed out.
    debug : bool, optional
        If True the fitting will output additional information and a diagnostic
        plot will be shown (this might cause a long output if ``array`` is 3d
        and has many slices).

    Returns
    -------
    psf_norm : numpy ndarray
        The normalized PSF (2d or 3d array).
    fwhm_flux : numpy ndarray
        [full_output=True] The flux in a FWHM aperture (it can be a single
        value or a vector).
    fwhm : numpy ndarray
        [full_output=True] The FWHM size. If ``fwhm`` is set to 'fit' then it
        is the fitted FWHM value according to the assumed ``model`` (the mean in
        X and Y is returned when ``model`` is set to 'gauss').

    """
    
    if model == 'gauss':
        fit_2d = fit_2dgaussian
    elif model == 'moff':
        fit_2d = fit_2dmoffat
    elif model == 'airy':
        fit_2d = fit_2dairydisk
    else:
        raise ValueError('`Model` not recognized')

    if array.ndim == 2:
        y, x = array.shape
        if size is not None:
            if force_odd and size % 2 == 0:
                size += 1
                msg = "`Force_odd` is True therefore `size` was set to {}"
                print(msg.format(size))
        else:
            if force_odd and y % 2 == 0:
                size = y - 1
                msg = "`Force_odd` is True and frame size is even, therefore "
                msg += "new frame size was set to {}"
                print(msg.format(size))

        if size is not None:
            if size < array.shape[0]:
                array = frame_crop(array, size, force=True, verbose=False)
            else:
                array = array.copy()
        else:
            array = array.copy()

        if fwhm == 'fit':
            fit = fit_2d(array, full_output=True, debug=debug)
            if model == 'gauss':
                fwhm = np.mean((fit['fwhm_x'], fit['fwhm_y']))
                if verbose:
                    print("\nMean FWHM: {:.3f}".format(fwhm))
            elif model == 'moff' or model == 'airy':
                fwhm = fit.fwhm.at[0]
                if verbose:
                    print("FWHM: {:.3f}".format(fwhm))

        res = _psf_norm_2d(array, fwhm, threshold, mask_core, full_output,
                          verbose, fit_2d=fit_2d, imlib=imlib, interpolation=interpolation)
        return res

    elif array.ndim == 3:
        n, y, x = array.shape
        if size is not None:
            if force_odd and size % 2 == 0:
                size += 1
                msg = "`Force_odd` is True therefore `size` was set to {}"
                print(msg.format(size))
        else:
            if force_odd and y % 2 == 0:
                size = y - 1
                msg = "`Force_odd` is True and frame size is even, therefore "
                msg += "new frame size was set to {}"
                print(msg.format(size))

        if size is not None:
            if size < array.shape[1]:
                array = cube_crop_frames(array, size, force=True, verbose=False)
            else:
                array = array.copy()

        if isinstance(fwhm, (int, float)):
            fwhm = [fwhm] * array.shape[0]
        elif fwhm == 'fit':
            fits_vect = [fit_2d(array[i], full_output=True, debug=debug) for i
                         in range(n)]
            if model == 'gauss':
                fwhmx = [fits_vect[i]['fwhm_x'] for i in range(n)]
                fwhmy = [fits_vect[i]['fwhm_y'] for i in range(n)]
                fwhm_vect = [np.mean((fwhmx[i], fwhmy[i])) for i in range(n)]
                fwhm = np.array(fwhm_vect)
                if verbose:
                    print("Mean FWHM per channel: ")
                    print_precision(fwhm)
            elif model == 'moff' or model == 'airy':
                fwhm_vect = [fits_vect[i]['fwhm'] for i in range(n)]
                fwhm = np.array(fwhm_vect)
                fwhm = fwhm.flatten()
                if verbose:
                    print("FWHM per channel:")
                    print_precision(fwhm)
            # Replace outliers if needed
            if correct_outliers and np.sum(np.isnan(fwhm)) > 0:
                for f in range(n):
                    if np.isnan(fwhm[f]) and f != 0 and f != n-1:
                        fwhm[f] = np.nanmean(np.array([fwhm[f-1],
                                                        fwhm[f+1]]))
                    elif np.isnan(fwhm[f]):
                        msg = "2D fit failed for first or last channel. "
                        msg += "Try other parameters?"
                        raise ValueError(msg)
        array_out = []
        fwhm_flux = np.zeros(n)

        for fr in range(array.shape[0]):
            restemp = _psf_norm_2d(array[fr], fwhm[fr], threshold, mask_core,
                True, False, fit_2d=fit_2d, imlib=imlib, interpolation=interpolation)
            array_out.append(restemp[0])
            fwhm_flux[fr] = restemp[1]

        array_out = np.array(array_out)
        if verbose:
            print("Flux in 1xFWHM aperture: ")
            print_precision(fwhm_flux)

        if full_output:
            return array_out, fwhm_flux, fwhm
        else:
            return array_out
