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

def calc_scal(cubes, wavelengths, flux_st, mask, do_opt=False, debug: bool=False):
    '''
    Iterate over wavelength channels to find opt spatial- and flux-scaling factors for each
    '''

    n_frames = cubes.shape[1]

    if do_opt:
        
        time_cubes = [cubes[:, i] for i in range(n_frames)]
        input_list = list(zip(time_cubes, np.repeat([wavelengths], n_frames, axis=0), 
                    np.repeat([flux_st], n_frames, axis=0), np.repeat([mask], n_frames, axis=0),
                    np.repeat([2], n_frames), np.repeat(["stddev"], n_frames), np.repeat([debug], n_frames)))

        with mp.Pool(redux_utils.numworkers) as pool:
            output = np.array(pool.starmap(find_scal_vector, input_list, chunksize=redux_utils.chunksize))

        opt_scals = output[:,0]
        opt_fluxes = output[:,1]

        opt_scal = np.median(opt_scals, axis=0)
        opt_flux = np.median(opt_fluxes, axis=0)

    else:
        opt_scal, opt_flux = find_scal_vector(np.mean(cubes, axis=1), wavelengths, flux_st, mask=mask, nfp=2, fm="stddev", debug=debug)

    return opt_scal, opt_flux


def prep(cubes: np.ndarray, wavelengths: np.ndarray, mask_rad: float=10,
         psf: np.ndarray=None, ret_fwhm_list: bool=False, verbose: bool=False,
         debug: bool=False, do_opt: bool=False) -> tuple[float, np.ndarray, np.ndarray]:
    '''
    Prepare key arguments required in the post-processing algorithms in VIP.
    These arguments are the FWHM of the PSF before normalization, the optimal
    scaling geometrically, and the optimal scaling in flux in the images.
    '''

    # model psf - take median along time axis - beware of companion smearing
    if psf is None:
        psf = np.median(cubes, axis=1)

    # get flux and fwhm of host star in each channel
    psfn, flux_st, fwhm_list = normalize_psf(psf, fwhm="fit", full_output=True,
                                             verbose=verbose, debug=debug)
    fwhm = fwhm_list if ret_fwhm_list else np.mean(fwhm_list)

    #pixel diameter of star
    mask = mask_circle(np.ones_like(cubes[0,0]), mask_rad)

    opt_scal, opt_flux = calc_scal(cubes, wavelengths, flux_st, mask, do_opt=do_opt, debug=debug)

    return fwhm, psfn, opt_scal, opt_flux

def ASDI_vip(cubes: np.ndarray, wavelengths: np.ndarray, angles: np.ndarray,
        mask_rad: float=10, do_opt: bool=False, full_output: bool=False,
        sub_type: str="ASDI", **kwargs) -> np.ndarray:


    fwhm, _, opt_scal, opt_flux = prep(cubes, wavelengths, mask_rad=mask_rad, do_opt=do_opt)


    # algo args
    nproc = redux_utils.numworkers
    imlib = "vip-fft"
    interpolation ="lanczos4"
    rot_options = {"interp_zeros": True, "mask_val": 0}
    combine_fn = np.median
    nchnls = len(wavelengths)


    # Annular ASDI kws
    if "annular" in sub_type:

        kwkeys = kwargs.keys()
        for kw in ["asize", "delta_rot", "delta_sep", "nframes"]:
            if kw not in kwkeys:
                print("The necessary kwargs for an annuluar ADI subtraction" + \
                      " were not provided. Missing '%s' at least..."%kw)
                return None
            
        asize = fwhm if kwargs["asize"] is None else kwargs["asize"]
        delta_rot = (0.1, 1.0) if kwargs["delta_rot"] is None else kwargs["delta_rot"]
        delta_sep = 1.0 if kwargs["delta_sep"] is None else kwargs["delta_sep"]
        nframes = "auto" if kwargs["nframes"] is None else kwargs["nframes"]

    
    if sub_type == "ASDI":
        sub = median_sub(cubes, angles, scale_list=opt_scal,
                            flux_sc_list=opt_flux, radius_int=mask_rad,
                            nproc=nproc, full_output=full_output, **rot_options)
    # elif sub_type == "ASDI_annular":
    #     sub = median_sub(cubes, angles, scale_list=opt_scal,
    #                       flux_sc_list=opt_flux, fwhm=fwhm, asize=asize,
    #                       mode="annular", delta_rot=delta_rot, nproc=nproc,
    #                       radius_int=mask_rad, nframe=nframes, imlib=imlib,
    #                       collapse=collapse, interpolation=interpolation,
    #                       full_output=full_output, **rot_options)
    elif sub_type == "ADI":
        dict_args = {"collapse": collapse, "imlib": imlib, "interpolation": interpolation, "nproc": nproc}
        pool_args = list(zip(cubes, np.repeat([angles], nchnls, axis=0),
                    np.repeat(dict_args, nchnls, axis=0)))
        with mp.Pool(nproc) as pool:
            sub_adi = np.array(pool.starmap(median_sub, pool_args))
        sub = redux_utils.combine(sub_adi)
    elif sub_type == "ADI_annular":
        dict_args = {"collapse": collapse, "fwhm": fwhm, "asize": asize, "mode": "annular",
                     "delta_rot": delta_rot, "delta_sep": delta_sep, "radius_int": mask_rad,
                     "nframes": nframes, "imlib": imlib, "interpolation": interpolation, "nproc": nproc,
                     "full_output": full_output}
        pool_args = list(zip(cubes, np.repeat([angles], nchnls, axis=0),
                    np.repeat(dict_args, nchnls, axis=0)))
        with mp.Pool(nproc) as pool:
            sub_adi_ann = np.array(pool.starmap(median_sub, pool_args))
        sub = redux_utils.combine(sub_adi_ann, combine_fn=combine_fn)
    elif sub_type == "SDI":
        sub = median_sub(cubes, angles, scale_list=opt_scal, sdi_only=True,
                             radius_int=mask_rad, full_output=full_output,
                             **rot_options)
    else:
        print("The following subtraction type '%s' is not implemented"%sub_type)
        return None

    return sub


def PCA_vip(cubes: np.ndarray, wavelengths: np.ndarray, angles: np.ndarray,
        mask_rad: float=10, do_opt: bool=False, full_output: bool=False,
        sub_type: str="single", **kwargs) -> np.ndarray:


    fwhm, _, opt_scal, _ = prep(cubes, wavelengths, mask_rad=mask_rad, do_opt=do_opt)


    # algo args
    ncomp = redux_utils.numcomps
    nproc = redux_utils.numworkers
    kwkeys = kwargs.keys()
    scaling = "temp-standard" if "scaling" not in kwkeys else kwargs["scaling"]
    rot_options = {"interp_zeros": True, "mask_val": 0} if "rot_options" not in kwkeys else kwargs["rot_options"]
    
    # Annular PCA kws
    if "annular" in sub_type:
        
        for kw in ["asize", "delta_rot", "delta_sep", "nframes"]:
            if kw not in kwkeys:
                print("The necessary kwargs for an annuluar ADI subtraction" + \
                      "were not provided. Missing '%s' at least..."%kw)
                return None
            
        asize = fwhm if kwargs["asize"] is None else kwargs["asize"]
        delta_rot = (0.1, 1.0) if kwargs["delta_rot"] is None else kwargs["delta_rot"]
        delta_sep = (0.1, 1.0) if kwargs["delta_sep"] is None else kwargs["delta_sep"]
        n_segments = "auto" if kwargs["n_segments"] is None else kwargs["n_segments"]


    # Commons args btwn all algos: cubes, angles, scale_list, nproc, full_output
    # interpolation, imlib, rot_options. Could pass as dict.

    # For ccurves and other ops, could abstract away the args themselves.
    # Perhaps even save as config/json file.

    args_sng = {"cube": cubes, "angle_list": angles, "scale_list": opt_scal,
                "ncomp": ncomp, "adimsdi":"single", "crop_ifs": False,
                "mask_center_px": mask_rad, "scaling": scaling, "nproc": nproc,
                "full_output": full_output, **rot_options}
    
    args_dbl = {"cube": cubes, "angle_list": angles, "scale_list": opt_scal,
                "ncomp": (ncomp, ncomp), "adimsdi":"double", "crop_ifs": False,
                "mask_center_px": mask_rad, "scaling": scaling, "nproc": nproc,
                "full_output": full_output, **rot_options}
    
    args_ann = {"cube": cubes, "angle_list": angles, "scale_list": opt_scal,
                "ncomp": (ncomp, ncomp), "asize": asize, "fwhm": fwhm,
                "delta_rot": delta_rot, "delta_sep": delta_sep,
                "n_segments": n_segments, "radius_int": mask_rad,
                "nproc": nproc, "full_output": full_output, **rot_options}
    
    if sub_type == "single":
        # Full-frame PCA-ASDI
        # Single step
        pp_sng = pca(**args_sng)
        pp_res = pp_sng
    elif sub_type == "double":
        # Double step
        pp_dbl = pca(**args_dbl)
        pp_res = pp_dbl
    elif sub_type == "annular":
        # Annular PCA-ASDI
        # Double step
        pp_ann = pca_annular(**args_ann)
        if full_output:
            pp_res = pp_ann[2, 0, 1]
        else:
            pp_res = pp_ann
    else:
        print("The following subtraction type '%s' is not implemented"%sub_type)
        return None

    return pp_res

if __name__ == "__main__":

    # --- DATA INFO --- #
    nframes = 2202
    firstchannelnum = 45
    lastchannelnum = 74
    channelnums = list(range(firstchannelnum, lastchannelnum + 1))
    nchnls = len(channelnums)
    frames = range(0, nframes, redux_utils.everynthframe)
    # --- DATA INFO --- #

    # --- PATHS --- #
    data_path = "./data/005_center_multishift/wl_channel_%05i.fits"
    data_paths = [data_path%i for i in channelnums]

    angles_path = "data/parang_bads_removed.txt"
    wavelengths_path = "data/channel_wavelengths.txt"

    outcombined_path = "./out/%s.fits"%("temp_output")
    # --- PATHS --- #

    # --- DATA --- #
    cubes, wavelengths, angles = redux_utils.init(data_paths, wavelengths_path, angles_path, channels=channelnums, frames=frames)
    # --- DATA --- #
    
    ASDI_vip(cubes, wavelengths, angles, out_path=outcombined_path, do_opt=False, sub_type="ASDI")