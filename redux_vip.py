import multiprocessing as mp
import numpy as np
from typing import Callable

from vip_hci.fm import normalize_psf
from vip_hci.psfsub import median_sub, pca, pca_annular
from vip_hci.preproc import find_scal_vector, frame_rescaling
from vip_hci.var import mask_circle

import redux_utils

imlib = "vip-fft"
interpolation = "lanczos4"
collapse="median"
nbranch = 2
pxscale = 0.027
drot = 0.5
rot_options = {'mask_val': 0, 'interp_zeros': True, 'imlib': 'vip-fft', 'interpolation': 'lanczos4'}

def calc_scal(cubes, wavelengths, flux_st, mask, do_opt=False):
    '''
    Iterate over wavelength channels to find opt spatial- and flux-scaling factors for each
    '''

    n_frames = cubes.shape[1]

    if do_opt:
        
        time_cubes = [cubes[:, i] for i in range(n_frames)]
        input_list = list(zip(time_cubes, np.repeat([wavelengths], n_frames, axis=0), 
                    np.repeat([flux_st], n_frames, axis=0), np.repeat([mask], n_frames, axis=0),
                    np.repeat([2], n_frames), np.repeat(["stddev"], n_frames)))

        with mp.Pool(redux_utils.numworkers) as pool:
            output = np.array(pool.starmap(find_scal_vector, input_list, chunksize=redux_utils.chunksize))

        opt_scals = output[:,0]
        opt_fluxes = output[:,1]

        opt_scal = np.median(opt_scals, axis=0)
        opt_flux = np.median(opt_fluxes, axis=0)

    else:
        opt_scal, opt_flux = find_scal_vector(np.mean(cubes, axis=1), wavelengths, flux_st, mask=mask, nfp=2, fm="stddev")

    return opt_scal, opt_flux


def _prep(cubes: np.ndarray, wavelengths: np.ndarray, mask_rad: float=10,
          do_opt: bool=False) -> tuple[float, np.ndarray, np.ndarray]:
    '''
    Prepare key arguments required in the post-processing algorithms in VIP.
    These arguments are the FWHM of the PSF before normalization, the optimal
    scaling geometrically, and the optimal scaling in flux in the images.
    '''

    # model psf - take median along time axis - beware of companion smearing
    psf = np.median(cubes, axis=1)

    # get flux and fwhm of host star in each channel
    psfn, flux_st, fwhm_list = normalize_psf(psf, fwhm="fit", full_output=True, debug=False)
    fwhm = np.mean(fwhm_list)

    #pixel diameter of star
    mask = mask_circle(np.ones_like(cubes[0,0]), mask_rad)

    opt_scal, opt_flux = calc_scal(cubes, wavelengths, flux_st, mask, do_opt=do_opt)

    return fwhm, opt_scal, opt_flux

def ASDI_vip(cubes: np.ndarray, wavelengths: np.ndarray, angles: np.ndarray,
        out_path: str=None, mask_rad: float=10, do_opt: bool=False,
        full_output: bool=False, sub_type: str="ASDI",
        **kwargs) -> tuple[np.ndarray, np.ndarray]:


    fwhm, opt_scal, opt_flux = _prep(cubes, wavelengths, mask_rad=mask_rad, do_opt=do_opt)


    # algo args
    ncomp = redux_utils.numcomps
    nproc = redux_utils.numworkers
    scaling = "temp-standard"
    an_dist = np.linspace(np.min(angles), np.max(angles), nbranch, endpoint=True)
    algo_dict = {'ncomp': ncomp, 'imlib': "vip-fft", "interpolation": "lanczos4"}
    rot_options = {"interp_zeros": True, "mask_val": 0}
    combine_fn = np.median



    # Annular ASDI kws
    if "annular" in sub_type:
        mode = "annular"

        kwkeys = kwargs.keys()
        for kw in ["asize", "delta_rot", "delta_sep", "nframes"]:
            if kw not in kwkeys:
                print("The necessary kwargs for an annuluar ADI subtraction" + \
                      "were not provided. Missing '%s' at least..."%kw)
                return (None, None)
            
        asize = fwhm if kwargs["asize"] is None else kwargs["asize"]
        delta_rot = kwargs["delta_rot"]
        delta_sep = kwargs["delta_sep"]
        nframes = kwargs["nframes"]
    

    
    if sub_type == "ASDI":
        sub = median_sub(cubes, angles, scale_list=opt_scal,
                            flux_sc_list=opt_flux, radius_int=mask_rad,
                            nproc=nproc)
    elif sub_type == "ASDI_annular":
        sub = median_sub(cubes, angles, scale_list=opt_scal,
                          flux_sc_list=opt_flux, fwhm=fwhm, asize=asize,
                          mode="annular", delta_rot=delta_rot,
                          radius_int=mask_rad, nframe=nframes, imlib=imlib,
                          collapse=collapse, interpolation=interpolation)
    elif sub_type == "ADI":
        sub_adi = median_sub(cubes, angles, imlib=imlib, collapse=collapse,
                             interpolation=interpolation)
        sub = redux_utils.combine(sub_adi)
    elif sub_type == "ADI_annular":
        sub_adi_ann = median_sub(cubes, angles, fwhm=fwhm, asize=asize,
                                 mode="annular", delta_rot=delta_rot,
                                 radius_int=mask_rad, nframe=nframes,
                                 imlib=imlib, collapse=collapse,
                                 interpolation=interpolation)
        sub = redux_utils.combine(sub_adi_ann, combine_fn=combine_fn)
    elif sub_type == "SDI":
        sub = median_sub(cubes, angles, scale_list=opt_scal, sdi_only=True,
                             radius_int=mask_rad, rot_options=rot_options)
    else:
        print("The following subtraction type '%s' is not implemented"%sub_type)
        return (None, None)

    if out_path is not None:
        redux_utils.to_fits(sub, out_path)

    return sub, fwhm


def PCA_vip(cubes: np.ndarray, wavelengths: np.ndarray, angles: np.ndarray,
        out_path: str=None, mask_rad: float=10, do_opt: bool=False,
        full_output: bool=False, sub_type: str="ASDI",
        **kwargs) -> tuple[np.ndarray, np.ndarray]:


    fwhm, opt_scal, opt_flux = _prep(cubes, wavelengths, mask_rad=mask_rad, do_opt=do_opt)



    # algo args
    ncomp = redux_utils.numcomps
    nproc = redux_utils.numworkers
    scaling = "temp-standard"
    an_dist = np.linspace(np.min(angles), np.max(angles), nbranch, endpoint=True)
    algo_dict = {"ncomp": ncomp, "imlib": "vip-fft", "interpolation": "lanczos4"}
    rot_options = {"interp_zeros": True, "mask_val": 0}

    

    # Annular ASDI kws
    if "annular" in sub_type:
        mode = "annular"

        kwkeys = kwargs.keys()
        for kw in ["asize", "delta_rot", "delta_sep", "nframes"]:
            if kw not in kwkeys:
                print("The necessary kwargs for an annuluar ADI subtraction" + \
                      "were not provided. Missing '%s' at least..."%kw)
                return (None, None)
            
        asize = fwhm if kwargs["asize"] is None else kwargs["asize"]
        delta_rot = kwargs["delta_rot"]
        delta_sep = kwargs["delta_sep"]
        nframes = kwargs["nframes"]
    
    

    if sub_type == "PCA_single":
        # Full-frame PCA-ASDI
        # Single step
        sub = pca(cubes, angles, scale_list=opt_scal, ncomp=ncomp,
                adimsdi="single", crop_ifs=False, mask_center_px=mask_rad,
                scaling=scaling, nproc=nproc, full_output=full_output,
                **rot_options)
    elif sub_type == "PCA_double":
        # Double step
        sub = pca(cubes, angles, scale_list=opt_scal, ncomp=(ncomp, ncomp),
                    adimsdi="double", crop_ifs=False, mask_center_px=mask_rad,
                    interpolation=interpolation, scaling=scaling, nproc=nproc,
                    full_output=full_output, **rot_options)
    elif sub_type == "PCA_annular":
        # Annular PCA-ASDI
        # Double step
        sub = pca_annular(cubes, angles, scale_list=opt_scal, ncomp=(ncomp, ncomp),
                    radius_int=mask_rad, asize=asize, fwhm=fwhm, nproc=nproc, full_output=full_output, **rot_options)
    else:
        print("The following subtraction type '%s' is not implemented"%sub_type)
        return (None, None)

    if out_path is not None:
        redux_utils.to_fits(sub, out_path)

    return sub, fwhm

if __name__ == "__main__":

    # --- DATA INFO --- #
    firstchannelnum = 45
    lastchannelnum = 74
    channelnums = list(range(firstchannelnum, lastchannelnum + 1))
    nchnls = len(channelnums)
    # --- DATA INFO --- #

    # --- PATHS --- #
    data_path = "./data/005_center_multishift/wl_channel_%05i.fits"
    data_paths = [data_path%i for i in channelnums]

    # outchannel_path = None
    # outchannel_paths = [outchannel_path] * nchnls
    outchannel_path = "./out/vipADI_%05i_median.fits"
    outchannel_paths = [outchannel_path%i for i in channelnums]
    outcombined_path = "./out/vipADI_%05i_%05i_median.fits"%(firstchannelnum, lastchannelnum)
    # --- PATHS --- #

    # --- DATA --- #
    cubes = redux_utils.loadall(data_paths, verbose=False)
    cubes_skipped = cubes[:,::redux_utils.everynthframe]

    angles = redux_utils.angles
    angles_skipped = angles[::redux_utils.everynthframe]

    wavelengths = redux_utils.wavelengths
    wavelengths_selected = wavelengths[channelnums]

    # ASDI_vip(cube_skipped, angles_skipped, combine_fn=np.mean, collapse_channel="median", out_path=outcombined_path, outchannel_paths=outchannel_paths)
    ASDI_vip(cubes_skipped, wavelengths_selected, angles_skipped, out_path=outcombined_path, do_opt=False, sub_type="ASDI")